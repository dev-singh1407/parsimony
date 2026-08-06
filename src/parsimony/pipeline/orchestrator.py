"""The orchestrator.

Modules propose; this commits (ADR-001). Five things fall out of one loop, none
of which any module has to implement: ablation, instrumentation, fidelity
gating, timing, and revert. Revert is simply not assigning.

Keep this small. Every line of coordination logic added here is a line that
cannot live in a module, and the ablation depends on modules being independent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from parsimony.core.config import ParsimonyConfig
from parsimony.core.errors import LedgerError, ModuleError, ProviderError
from parsimony.core.ledger import GateEvent, LedgerRow, StageOutcome, StageTrace
from parsimony.core.proposals import ContextPatch, NoOp, ShortCircuit
from parsimony.core.types import (
    GenParams,
    Invariants,
    Mode,
    RequestContext,
    RouteTier,
    Turn,
)
from parsimony.infra.derived import DerivedCache
from parsimony.infra.embedding import get_embedder
from parsimony.infra.ids import ulid
from parsimony.infra.memo import MemoEntry, gen_params_hash, memo_key
from parsimony.infra.nlp import RegexInvariantExtractor, RegexPiiDetector
from parsimony.infra.providers import MockProvider
from parsimony.infra.storage import MemoryBlobStore, sha256
from parsimony.infra.tokenization import get_tokenizer
from parsimony.modules.m2_cache import SemanticCache
from parsimony.modules.m4_assembler import assemble_prefix_stable, assemble_volatile_head
from parsimony.modules.m5_budgeter import OutputBudgeter
from parsimony.modules.m8_fidelity import FidelityGate
from parsimony.pipeline.assembly import prefix_survival
from parsimony.pipeline.registry import PlannedStage, StageRegistry, default_registry

DEFAULT_NUM_PREDICT = 256

# Conversations whose previous prompt is retained for prefix-survival
# measurement. Only the immediately preceding turn is ever compared against, so
# this bounds a structure that otherwise grows for the life of the process.
MAX_TRACKED_CONVERSATIONS = 256


@dataclass(frozen=True, slots=True)
class Outcome:
    response: str
    row: LedgerRow
    ctx: RequestContext
    traces: tuple[StageTrace, ...]
    generated: bool

    @property
    def served_by(self) -> str:
        return self.row.route_tier


class Pipeline:
    def __init__(
        self,
        cfg: ParsimonyConfig,
        *,
        provider=None,
        provider_large=None,
        tokenizer=None,
        cache: SemanticCache | None = None,
        registry: StageRegistry | None = None,
        gate: FidelityGate | None = None,
        sink=None,
        blobs=None,
        embedder=None,
        memo=None,
        run_id: str | None = None,
        pass_kind: str = "quality",
        corpus_hash: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer or get_tokenizer(cfg.tokenizer_id)
        self.provider = provider or MockProvider()
        self.provider_large = provider_large
        self.embedder = embedder if embedder is not None else get_embedder(cfg.embedder_id)
        self.cache = (
            cache if cache is not None else SemanticCache(cfg.cache.ttl_seconds, self.embedder, max_entries=cfg.cache.max_entries)
        )
        self.cache.attach_embedder(self.embedder)
        self.registry = registry or default_registry(self.cache)
        self.gate = gate or FidelityGate()
        self.sink = sink
        self.blobs = blobs or MemoryBlobStore()
        # Only consulted in EXPERIMENT mode: a memoised response is bit-exact
        # but takes microseconds, so it must never reach a latency measurement
        # or a served request (ADR-019).
        self.memo = memo if cfg.mode is Mode.EXPERIMENT else None
        self.run_id = run_id or ulid()
        self.pass_kind = pass_kind
        self.corpus_hash = corpus_hash
        self._extractor = RegexInvariantExtractor()
        self._pii = RegexPiiDetector()
        # Previous prompt per conversation, for prefix-survival measurement.
        # Bounded: only the most recent conversations can produce a next turn,
        # and an unbounded dict here grows for the life of the process — 400
        # conversations tracked 400 token lists in a probe.
        self._last_prompt_ids: dict[str, list[int]] = {}
        self._max_tracked_conversations = MAX_TRACKED_CONVERSATIONS

        # Fail fast: a misconfigured order costs one second here rather than six
        # unattended CPU-hours in the sweep.
        self.registry.validate(cfg)

    # -- ingestion + preprocessing -----------------------------------------

    def build_context(
        self,
        query: str,
        history: tuple[Turn, ...] = (),
        conversation_id: str | None = None,
        turn_index: int = 0,
    ) -> RequestContext:
        history = tuple(
            t.with_tokens(self.tokenizer.count(t.content)) if not t.token_count else t
            for t in history
        )
        payload = "\n".join([t.content for t in history] + [query])
        invariants: Invariants = self._extractor.extract(payload)
        ctx = RequestContext(
            request_id=ulid(),
            conversation_id=conversation_id or ulid(),
            original_query=query,
            original_history=history,
            invariants=invariants,
            query=query,
            history=history,
            system_prompt=self.cfg.system_prompt,
            context_digest=self.cfg.context_digest,
            turn_index=turn_index,
            config_hash=self.cfg.config_hash,
            corpus_hash=self.corpus_hash,
        )
        return replace(ctx, derived=DerivedCache(self.tokenizer, self.embedder))

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        query: str,
        history: tuple[Turn, ...] = (),
        conversation_id: str | None = None,
        turn_index: int = 0,
    ) -> Outcome:
        t_start = time.perf_counter_ns()
        ctx = self.build_context(query, history, conversation_id, turn_index)
        original_ctx = ctx

        traces: list[StageTrace] = []
        tokens_per_stage: dict[str, int] = {}
        gate_events: list[GateEvent] = []
        cache_stage = None
        cache_lookup_ctx: RequestContext | None = None
        cache_evidence: dict[str, Any] = {}
        cache_consulted = False
        short: ShortCircuit | None = None

        for stage in self.registry.ordered(self.cfg):
            t0 = time.perf_counter_ns()
            before_tokens = ctx.derived.token_count(ctx.text_payload())

            if isinstance(stage, PlannedStage):
                traces.append(
                    _trace(stage, StageOutcome.NOT_IMPLEMENTED, before_tokens,
                           before_tokens, t0, "scheduled for a later sprint")
                )
                continue

            if not self.cfg.enables(stage.module_id):
                traces.append(
                    _trace(stage, StageOutcome.SKIPPED, before_tokens, before_tokens,
                           t0, "module disabled")
                )
                continue

            is_cache = hasattr(stage, "remember")
            if is_cache:
                # Snapshot the context as the cache saw it: by the time the
                # response is ready, M1 may have rewritten the query, and the
                # write must use the same key the lookup used.
                cache_stage = stage
                cache_lookup_ctx = ctx
                cache_consulted = True

            try:
                if not stage.applies_to(ctx, self.cfg):
                    reason = getattr(stage, "skip_reason", None)
                    traces.append(
                        _trace(stage, StageOutcome.SKIPPED, before_tokens, before_tokens, t0,
                               reason(ctx, self.cfg) if reason else "not applicable")
                    )
                    continue
                proposal = stage.propose(ctx, self.cfg)
            except Exception as exc:  # a module must never fail the request
                traces.append(
                    _trace(stage, StageOutcome.ERROR, before_tokens, before_tokens, t0,
                           f"{type(exc).__name__}: {exc}")
                )
                continue

            if is_cache:
                cache_evidence = dict(getattr(proposal, "evidence", {}) or {})

            if isinstance(proposal, ShortCircuit):
                traces.append(
                    _trace(stage, StageOutcome.SHORT_CIRCUIT, before_tokens, 0, t0,
                           proposal.rationale, proposal.evidence)
                )
                short = proposal
                break

            if isinstance(proposal, NoOp):
                traces.append(
                    _trace(stage, StageOutcome.NOOP, before_tokens, before_tokens, t0,
                           proposal.detail or proposal.reason, proposal.evidence)
                )
                continue

            assert isinstance(proposal, ContextPatch)
            candidate = replace(ctx, **dict(proposal.fields))
            verdict = self.gate.check(ctx, candidate, proposal.kind, stage.module_id)

            if verdict.passed:
                after_tokens = ctx.derived.token_count(candidate.text_payload())
                traces.append(
                    _trace(stage, StageOutcome.APPLIED, before_tokens, after_tokens, t0,
                           proposal.rationale, proposal.evidence)
                )
                tokens_per_stage[stage.name] = after_tokens
                ctx = candidate  # commit
            else:
                gate_events.extend(verdict.events)
                traces.append(
                    _trace(stage, StageOutcome.REVERTED, before_tokens, before_tokens, t0,
                           verdict.detail, {}, tuple(verdict.events))
                )
                # revert == do nothing

        # -- generation or short circuit ------------------------------------

        # Reference denominator for the per-request input-reduction display.
        # Always the prefix-stable rendering of the ORIGINAL content, so it is
        # identical across ablation cells and only the content differs.
        # Between-cell comparison is done by the runner against the baseline
        # cell's totals, not against this.
        tokens_in_original = assemble_prefix_stable(
            original_ctx, ctx.derived.token_count
        ).total_token_count

        if short is not None:
            response = short.response
            route = short.served_by
            tokens_in_final = 0
            tokens_out = 0
            ttft = tpot = None
            gen_ns = 0
            prompt_text = ""
            prefix_survived, prefix_ratio = None, None
            early_stopped = False
            memoised = False
        else:
            # M4 produces the assembled prompt when enabled. With it ablated we
            # fall back to a volatile-head rendering, which models the very
            # common "turn N of M" preamble that silently destroys KV prefix
            # reuse while looking free in a token count (see m4_assembler).
            final_prompt = ctx.assembled or assemble_volatile_head(ctx, ctx.derived.token_count)
            prompt_text = final_prompt.full_text
            tokens_in_final = final_prompt.total_token_count

            current_ids = self.tokenizer.encode(prompt_text)
            survived, ratio = prefix_survival(
                self._last_prompt_ids.get(ctx.conversation_id), current_ids
            )
            prefix_survived, prefix_ratio = survived, ratio
            self._remember_prompt(ctx.conversation_id, current_ids)

            route = ctx.route_tier or RouteTier.MODEL_SMALL
            provider = self._provider_for(route)
            g0 = time.perf_counter_ns()
            response, ttft, tpot, early_stopped, memoised = self._generate(
                prompt_text, ctx, provider
            )
            gen_ns = time.perf_counter_ns() - g0
            tokens_out = self.tokenizer.count(response)

            if cache_stage is not None and cache_lookup_ctx is not None:
                # Redaction at the write boundary (ADR-013): the model saw the
                # user's real text; the cache — the one component with memory —
                # never does.
                cache_stage.remember(cache_lookup_ctx, self.cfg, self._pii.redact(response))

        total_ns = time.perf_counter_ns() - t_start

        row = LedgerRow(
            request_id=ctx.request_id,
            conversation_id=ctx.conversation_id,
            turn_index=ctx.turn_index,
            config_hash=self.cfg.config_hash,
            config_label=self.cfg.label,
            run_id=self.run_id,
            corpus_hash=self.corpus_hash,
            seed=self.cfg.seed,
            pass_kind=self.pass_kind,
            created_at=time.time(),
            model_name=self._provider_for(route).model_name,
            model_quantisation=self._provider_for(route).quantisation,
            model_digest=self._provider_for(route).model_digest,
            tokenizer_id=self.tokenizer.id,
            embedder_id=self.cfg.embedder_id,
            tokens_in_original=tokens_in_original,
            tokens_in_final=tokens_in_final,
            tokens_per_module=tokens_per_stage,
            tokens_out=tokens_out,
            tokens_out_budget=ctx.output_budget,
            route_tier=route.name,
            cache_consulted=cache_consulted,
            cache_hit=route in (RouteTier.CACHE_EXACT, RouteTier.CACHE_SEMANTIC),
            # Persisted even on a miss: the rejected candidates and their scores
            # are what make the threshold sweep an offline re-analysis of one
            # run rather than one CPU-bound run per threshold.
            cache_zone=cache_evidence.get("zone") if cache_consulted else None,
            cache_top_k=tuple(tuple(x) for x in cache_evidence.get("top_k", ())),
            cache_verifier=cache_evidence.get("verifier"),
            gate_fired=bool(gate_events),
            gate_events=tuple(gate_events),
            prefix_tokens_survived=prefix_survived,
            prefix_ratio=prefix_ratio,
            ttft_ns=ttft,
            tpot_ns=tpot,
            total_ns=total_ns,
            middleware_ns=total_ns - gen_ns,
            per_stage_ns={t.name: t.duration_ns for t in traces},
            generation_memoised=memoised,
            early_stopped=early_stopped,
            # Derived from wall clock and the configured package power, not
            # metered. Meaningless on a memoised row, so left None there rather
            # than reporting the energy cost of a dictionary lookup.
            joules_estimated=(
                None if memoised
                else (total_ns / 1e9) * self.cfg.energy.package_power_watts
            ),
            usd_equivalent=(
                tokens_in_final / 1e6 * self.cfg.energy.usd_per_million_input
                + tokens_out / 1e6 * self.cfg.energy.usd_per_million_output
            ),
            prompt_sha256=self.blobs.put(prompt_text) if prompt_text else "",
            response_sha256=self.blobs.put(self._pii.redact(response)),
            traces=tuple(traces),
        )

        self._emit(row)
        return Outcome(response, row, ctx, tuple(traces), short is None)

    # -- helpers -----------------------------------------------------------

    def _remember_prompt(self, conversation_id: str, ids: list[int]) -> None:
        """Keep the last prompt per conversation, LRU-bounded.

        Prefix survival only ever compares against the immediately preceding
        turn of the same conversation, so retaining every conversation forever
        buys nothing and grows without limit.
        """
        self._last_prompt_ids.pop(conversation_id, None)
        self._last_prompt_ids[conversation_id] = ids
        while len(self._last_prompt_ids) > self._max_tracked_conversations:
            self._last_prompt_ids.pop(next(iter(self._last_prompt_ids)))

    def _provider_for(self, tier: RouteTier):
        """Escalation has somewhere to go only if a large provider was supplied.

        Falling back silently would make the M6b arm look like it escalated
        while actually running the same model, so the ledger's route_tier would
        be a lie. Instead the tier is recorded as chosen and the provider used
        is recorded separately via model_name.
        """
        if tier is RouteTier.MODEL_LARGE and self.provider_large is not None:
            return self.provider_large
        return self.provider

    def _generate(self, prompt: str, ctx: RequestContext, provider=None):
        provider = provider or self.provider
        params = GenParams(
            num_predict=ctx.output_budget or DEFAULT_NUM_PREDICT,
            temperature=0.0,
            seed=self.cfg.seed,
        )
        key = None
        if self.memo is not None:
            key = memo_key(
                prompt, provider.model_digest, gen_params_hash(params, self.cfg)
            )
            cached = self.memo.get(key)
            if cached is not None:
                # Bit-exact: identical prompt, digest and params at temperature 0
                # produce identical output. Timing fields are left None rather
                # than fabricated — a memo hit has no meaningful TTFT.
                return cached.text, None, None, cached.early_stopped, True

        stopper = OutputBudgeter.stopper(self.cfg)
        pieces: list[str] = []
        first = last = None
        early_stopped = False
        t0 = time.perf_counter_ns()
        try:
            for event in provider.generate(prompt, params):
                if first is None:
                    first = event.emitted_at_ns
                last = event.emitted_at_ns
                pieces.append(event.text)
                if stopper is not None and stopper.observe(event.text):
                    early_stopped = True
                    break
        except Exception as exc:
            raise ProviderError(f"generation failed: {exc}") from exc

        text = "".join(pieces)
        ttft = (first - t0) if first is not None else None
        tpot = None
        if first is not None and last is not None and len(pieces) > 1:
            tpot = (last - first) // (len(pieces) - 1)
        if self.memo is not None and key is not None:
            self.memo.put(key, MemoEntry(text=text, early_stopped=early_stopped))
        return text, ttft, tpot, early_stopped, False

    def _emit(self, row: LedgerRow) -> None:
        if self.sink is None:
            return
        try:
            self.sink.write(row)
        except Exception as exc:
            # Opposite priorities by mode (ADR-005): in an experiment the ledger
            # IS the result, so losing a row is fatal.
            if self.cfg.mode is Mode.EXPERIMENT:
                raise LedgerError(f"ledger write failed: {exc}") from exc


def _trace(
    stage,
    outcome: StageOutcome,
    before: int,
    after: int,
    t0: int,
    rationale: str = "",
    evidence: dict[str, Any] | None = None,
    gate_events: tuple[GateEvent, ...] = (),
) -> StageTrace:
    return StageTrace(
        module_id=getattr(stage, "module_id", "--"),
        name=stage.name,
        outcome=outcome,
        tokens_before=before,
        tokens_after=after,
        duration_ns=time.perf_counter_ns() - t0,
        rationale=rationale,
        evidence=dict(evidence or {}),
        gate_events=gate_events,
    )
