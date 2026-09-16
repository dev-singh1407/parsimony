"""Long-context question answering: does question-aware compression keep the answer?

The conversation corpus cannot answer this. Its median question is six words,
and a compressor evaluated on six-word questions measures nothing but its own
floor (M1 saved 0.2%). This benchmark supplies what real long-context requests
look like -- six documents, about a thousand tokens, one question whose answer
sits in one or two sentences of one or two of them -- and asks every method to
keep the answer while removing the rest.

DESIGN, and what each choice guards against
-------------------------------------------
  * Fictional organisations. A model that already knows the answer scores well
    with the context deleted, and the benchmark would reward deleting it. The
    closed-book arm measures exactly this, and should sit near zero.
  * Distractors on purpose. Every collection repeats the attribute being asked
    about for other entities (three offices, three travel budgets), so keeping
    "a sentence about travel budgets" is not enough.
  * Five question kinds: lookup, distractor, anaphora (the answer sentence
    opens with "It ..."), negation, and two-hop (two sentences, sometimes in
    two documents). Each targets a different way extraction fails.
  * Document order shuffled per item with a fixed seed, so the answer is not
    always near the front -- which would flatter truncation.
  * A dev/test split by collection. Thresholds may be tuned on the two dev
    collections only; every reported figure is on the nine test collections.
  * Evidence spans recorded in advance. "Evidence recall" -- is every answer
    sentence still in the compressed context? -- needs no model, is
    deterministic, and runs in reproduce.py. Model accuracy is measured
    separately on the real model, because a mock cannot read.

BASELINES, each at the SAME token budget Parsimony used for that item
--------------------------------------------------------------------
  truncate   keep the context in order until the budget is spent: what every
             framework does when a prompt exceeds its window
  bm25_topk  rank sentences by BM25 and send the top ones as chunks, in rank
             order: textbook retrieval, with nothing added
  random     keep randomly chosen sentences: the floor any method must beat
  stopwords  delete stopwords everywhere (its own ratio, not matched)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from parsimony.core.config import ParsimonyConfig, full_stack
from parsimony.core.proposals import ContextPatch
from parsimony.core.types import Document, Invariants, RequestContext
from parsimony.eval import naive
from parsimony.eval.corpus import GoldItem
from parsimony.infra.nlp import split_sentences
from parsimony.modules.m1_context import ContextCompressor, bm25, build_units, ranking_terms
from parsimony.modules.m4_assembler import assemble_prefix_stable

CORPUS = Path(__file__).resolve().parents[3] / "corpus"


@dataclass(frozen=True, slots=True)
class LongItem:
    item_id: str
    collection: str
    split: str
    kind: str
    question: str
    gold: GoldItem
    evidence: tuple[str, ...]
    documents: tuple[Document, ...]


def load_longctx(root: Path | str | None = None) -> tuple[LongItem, ...]:
    base = Path(root) if root else CORPUS
    docs = {}
    for line in (base / "longctx_docs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            docs[o["doc_id"]] = Document(o["doc_id"], o["content"], o["title"])
    items = []
    for line in (base / "longctx_items.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        gold = GoldItem(o["item_id"], o["question"], o["gold_answer"], o["match"],
                        float(o.get("tolerance", 0.0)), tuple(o.get("acceptable_variants", ())))
        items.append(LongItem(o["item_id"], o["collection"], o["split"], o["kind"],
                              o["question"], gold, tuple(o["evidence"]),
                              tuple(docs[d] for d in o["doc_order"])))
    return tuple(items)


# ---------------------------------------------------------------- methods --


@dataclass(frozen=True, slots=True)
class Compressed:
    documents: tuple[Document, ...]
    detail: dict


def _sentences(item: LongItem) -> list[tuple[int, int, str]]:
    return [(d, p, s) for d, doc in enumerate(item.documents)
            for p, s in enumerate(split_sentences(doc.content))]


def _regroup(item: LongItem, kept: set[tuple[int, int]]) -> tuple[Document, ...]:
    """Kept (document, position) pairs back into documents, in original order."""
    out = []
    for d, doc in enumerate(item.documents):
        parts = [s for p, s in enumerate(split_sentences(doc.content)) if (d, p) in kept]
        if parts:
            out.append(replace(doc, content=" ".join(parts)))
    return tuple(out)


class Methods:
    """Every arm of the comparison, sharing one tokenizer, embedder and gate."""

    def __init__(self, cfg: ParsimonyConfig | None = None, *, tokenizer=None,
                 embedder=None) -> None:
        from parsimony.infra.providers import MockProvider
        from parsimony.pipeline.orchestrator import Pipeline

        self.cfg = cfg or full_stack()
        self._tokenizer, self._embedder = tokenizer, embedder
        self.pipeline = Pipeline(self.cfg, provider=MockProvider(), tokenizer=tokenizer,
                                 embedder=embedder)
        self.count = self.pipeline.tokenizer.count
        self.stage = ContextCompressor()
        # One pipeline per encoder. An arm that changes the embedder changes
        # the DERIVED cache the stage reads its vectors from, so reusing a
        # pipeline built for another encoder silently measures the wrong thing:
        # the first neural run scored sentences lexically and produced output
        # identical to the lexical arm, down to the token.
        self._by_encoder = {self.cfg.embedder_id: self.pipeline}

    def _pipeline_for(self, cfg: ParsimonyConfig):
        from parsimony.infra.embedding import get_embedder
        from parsimony.infra.providers import MockProvider
        from parsimony.pipeline.orchestrator import Pipeline

        if cfg.embedder_id not in self._by_encoder:
            self._by_encoder[cfg.embedder_id] = Pipeline(
                cfg, provider=MockProvider(), tokenizer=self._tokenizer,
                embedder=self._embedder if self._embedder is not None
                else get_embedder(cfg.embedder_id))
        return self._by_encoder[cfg.embedder_id]

    # -- the system under test -------------------------------------------

    def parsimony(self, item: LongItem, cfg: ParsimonyConfig | None = None) -> Compressed:
        cfg = cfg or self.cfg
        pipeline = self._pipeline_for(cfg)
        ctx = pipeline.build_context(item.question, conversation_id=item.item_id,
                                     documents=item.documents)
        proposal = self.stage.propose(ctx, cfg)
        if not isinstance(proposal, ContextPatch):
            return Compressed(item.documents, {"applied": False,
                                               "reason": getattr(proposal, "detail", "")})
        candidate = replace(ctx, **dict(proposal.fields))
        verdict = pipeline.gate.check(ctx, candidate, proposal.kind, "M1")
        if not verdict.passed:
            return Compressed(item.documents, {"applied": False, "gate": verdict.detail})
        return Compressed(candidate.documents, {"applied": True, **dict(proposal.evidence)})

    # -- baselines ---------------------------------------------------------

    def full(self, item: LongItem, budget: int | None = None) -> Compressed:
        return Compressed(item.documents, {})

    def closed_book(self, item: LongItem, budget: int | None = None) -> Compressed:
        return Compressed((), {})

    def truncate(self, item: LongItem, budget: int) -> Compressed:
        kept, used = set(), 0
        for d, p, s in _sentences(item):
            n = self.count(s)
            if used + n > budget and kept:
                break
            kept.add((d, p))
            used += n
        return Compressed(_regroup(item, kept), {"budget": budget})

    def random(self, item: LongItem, budget: int, seed: int = 0) -> Compressed:
        pool = _sentences(item)
        random.Random(f"{seed}:{item.item_id}").shuffle(pool)
        kept, used = set(), 0
        for d, p, s in pool:
            n = self.count(s)
            if used + n > budget and kept:
                continue
            kept.add((d, p))
            used += n
        return Compressed(_regroup(item, kept), {"budget": budget})

    def bm25_topk(self, item: LongItem, budget: int) -> Compressed:
        ctx = self.pipeline.build_context(item.question, documents=item.documents)
        c = self.cfg.compression
        units = build_units(ctx, [], self.count, c.context_term_prefix)
        scores = bm25(ranking_terms(item.question, c.context_term_prefix), units,
                      c.bm25_k1, c.bm25_b)
        ranked = sorted(range(len(units)), key=lambda i: -scores[i])
        chunks, used = [], 0
        for i in ranked:
            u = units[i]
            if used + u.tokens > budget and chunks:
                continue
            parent = item.documents[u.source[1]]
            # No title per chunk: repeating a title on every sentence would
            # charge this baseline prompt tokens Parsimony does not pay.
            chunks.append(Document(f"{parent.doc_id}#{u.position}", u.text))
            used += u.tokens
        return Compressed(tuple(chunks), {"budget": budget})

    def stopwords(self, item: LongItem, budget: int | None = None) -> Compressed:
        return Compressed(tuple(replace(d, content=naive.stopword_compress(d.content))
                                for d in item.documents), {})

    # -- measurement -------------------------------------------------------

    def prompt(self, item: LongItem, documents: tuple[Document, ...]) -> str:
        ctx = RequestContext(
            request_id="longctx", conversation_id=item.item_id,
            original_query=item.question, original_history=(), invariants=Invariants(),
            query=item.question, history=(), documents=documents,
            original_documents=documents, system_prompt=self.cfg.system_prompt)
        return assemble_prefix_stable(ctx, self.count).full_text

    def context_tokens(self, documents: tuple[Document, ...]) -> int:
        return sum(self.count(d.content) for d in documents)


def evidence_kept(item: LongItem, documents: tuple[Document, ...]) -> tuple[int, int]:
    """(evidence spans still present, spans in total)."""
    text = "\n".join(d.content for d in documents)
    return sum(e in text for e in item.evidence), len(item.evidence)


#: Parsimony with one component removed at a time -- what each is worth -- plus
#: "v1", the selector exactly as it was frozen for the first real-model run, so
#: a later change is measured against it rather than against a memory of it.
def ablations(cfg: ParsimonyConfig) -> dict[str, ParsimonyConfig]:
    from parsimony.core.config import context_v1, context_v2, neural

    c = cfg.compression
    return {
        "v1": context_v1(cfg),
        "v2": context_v2(cfg),
        # The same selector, scoring sentences with MiniLM instead of a lexical
        # encoder. Needs Ollama; costs ~9 ms per sentence (ADR-041).
        "neural": neural(cfg),
        "no_anchors": replace(cfg, compression=replace(
            c, context_anchor_guarantee=False, context_anchor_bonus=0.0)),
        "no_closure": replace(cfg, compression=replace(c, context_closure_depth=0)),
        "no_doc_prior": replace(cfg, compression=replace(c, context_doc_weight=0.0)),
        "no_mmr": replace(cfg, compression=replace(c, context_mmr_lambda=1.0)),
        "no_floor": replace(cfg, compression=replace(c, context_relevance_floor=0.0)),
    }


@dataclass(frozen=True, slots=True)
class ArmResult:
    item_id: str
    arm: str
    context_tokens: int
    full_context_tokens: int
    evidence_found: int
    evidence_total: int
    documents: tuple[Document, ...]

    @property
    def evidence_complete(self) -> bool:
        return self.evidence_found == self.evidence_total


def run_offline(methods: Methods, items, arms: tuple[str, ...] | None = None
                ) -> list[ArmResult]:
    """Every arm on every item, without a model. Deterministic."""
    variants = ablations(methods.cfg)
    wanted = arms or ("full", "parsimony", *[f"parsimony_{k}" for k in variants],
                      "bm25_topk", "truncate", "random", "stopwords", "closed_book")
    out: list[ArmResult] = []
    for item in items:
        full_tokens = methods.context_tokens(item.documents)
        ours = methods.parsimony(item)
        budget = methods.context_tokens(ours.documents)
        for arm in wanted:
            if arm == "parsimony":
                got = ours
            elif arm.startswith("parsimony_"):
                got = methods.parsimony(item, variants[arm.removeprefix("parsimony_")])
            else:
                fn: Callable = getattr(methods, arm)
                got = fn(item, budget)
            found, total = evidence_kept(item, got.documents)
            out.append(ArmResult(item.item_id, arm, methods.context_tokens(got.documents),
                                 full_tokens, found, total, got.documents))
    return out


# ------------------------------------------------------------- real model --

#: Every arm the real-model study runs, in the order it runs them.
REAL_ARMS = ("closed_book", "full", "parsimony", "parsimony_neural", "bm25_topk",
             "truncate", "random",
             "stopwords", "parsimony_no_anchors", "parsimony_no_closure",
             "parsimony_no_doc_prior", "parsimony_no_floor")

#: The confirmation run on the second held-out split: the two versions of the
#: selector against each other and against the same baselines, without the
#: component ablations (which the first split already measured).
CONFIRM_ARMS = ("closed_book", "full", "parsimony", "parsimony_neural", "parsimony_v2",
                "bm25_topk", "truncate", "random", "stopwords")


def run_real(methods: Methods, items, provider, out_path: Path, *,
             arms: tuple[str, ...] = REAL_ARMS, num_predict: int = 64,
             progress: Callable[[str], None] | None = None) -> list[dict]:
    """Ask the real model every item under every arm; append one JSON row per call.

    Resumable: rows already in `out_path` for this model digest are skipped, so
    an interrupted run -- a laptop lid closed an hour in -- continues where it
    stopped instead of starting over.

    Each prompt opens with a fresh nonce. Ollama reuses the KV cache for any
    prefix identical to the previous request's, and the arms of one item share
    long prefixes (truncation IS a prefix of the full context), so without it
    the second arm's prefill would be measured as nearly free (ADR-034).
    """
    from parsimony.core.types import GenParams
    from parsimony.eval.metrics import grade
    import secrets
    import time

    digest = provider.model_digest
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("model_digest") == digest:
                    rows.append(row)
                    done.add((row["item_id"], row["arm"]))

    variants = ablations(methods.cfg)
    params = GenParams(num_predict=num_predict, temperature=0.0, seed=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(items) * len(arms)
    with out_path.open("a", encoding="utf-8") as fh:
        for n_item, item in enumerate(items):
            ours = methods.parsimony(item)
            budget = methods.context_tokens(ours.documents)
            for n_arm, arm in enumerate(arms):
                if (item.item_id, arm) in done:
                    continue
                if arm == "parsimony":
                    got = ours
                elif arm.startswith("parsimony_"):
                    got = methods.parsimony(item, variants[arm.removeprefix("parsimony_")])
                else:
                    got = getattr(methods, arm)(item, budget)
                prompt = f"[{secrets.token_hex(3)}]\n" + methods.prompt(item, got.documents)
                if progress:
                    progress(f"{n_item * len(arms) + n_arm + 1}/{total}  {item.item_id}  {arm}")
                start = time.perf_counter()
                text, stats = provider.complete(prompt, params)
                wall_ms = (time.perf_counter() - start) * 1000
                found, spans = evidence_kept(item, got.documents)
                row = {
                    "item_id": item.item_id, "collection": item.collection,
                    "split": item.split, "kind": item.kind, "arm": arm,
                    "model": provider.model_name, "model_digest": digest,
                    "correct": grade(text, item.gold), "response": text.strip(),
                    "context_tokens": methods.context_tokens(got.documents),
                    "full_context_tokens": methods.context_tokens(item.documents),
                    "prompt_tokens": stats.get("prompt_eval_count"),
                    "prefill_ms": stats.get("prompt_eval_duration", 0) / 1e6,
                    "decode_ms": stats.get("eval_duration", 0) / 1e6,
                    "output_tokens": stats.get("eval_count"),
                    "wall_ms": round(wall_ms, 1),
                    "evidence_found": found, "evidence_total": spans,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class ArmSummary:
    arm: str
    n: int
    correct: int
    accuracy: object            # stats.Interval, percent
    context_kept_pct: float
    mean_prompt_tokens: float
    mean_prefill_ms: float
    mean_wall_ms: float
    vs_full_lost: int           # full right, this arm wrong
    vs_full_gained: int         # full wrong, this arm right
    vs_full_p: float
    by_kind: dict


def summarise_real(rows: list[dict], arms: tuple[str, ...] = REAL_ARMS) -> list[ArmSummary]:
    from parsimony.eval.stats import mcnemar_exact, wilson_interval

    by_arm: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["item_id"]] = r
    full = by_arm.get("full", {})
    out = []
    for arm in arms:
        got = by_arm.get(arm)
        if not got:
            continue
        ids = sorted(got)
        n = len(ids)
        correct = sum(got[i]["correct"] for i in ids)
        paired = [i for i in ids if i in full]
        lost = sum(full[i]["correct"] and not got[i]["correct"] for i in paired)
        gained = sum(got[i]["correct"] and not full[i]["correct"] for i in paired)
        kinds: dict[str, list[int]] = {}
        for i in ids:
            k = kinds.setdefault(got[i]["kind"], [0, 0])
            k[0] += got[i]["correct"]
            k[1] += 1
        kept = sum(got[i]["context_tokens"] for i in ids)
        full_tokens = sum(got[i]["full_context_tokens"] for i in ids) or 1
        out.append(ArmSummary(
            arm, n, correct, wilson_interval(correct, n), 100 * kept / full_tokens,
            sum(got[i]["prompt_tokens"] or 0 for i in ids) / n,
            sum(got[i]["prefill_ms"] for i in ids) / n,
            sum(got[i]["wall_ms"] for i in ids) / n,
            lost, gained, mcnemar_exact(lost, gained) if arm != "full" else 1.0,
            {k: tuple(v) for k, v in sorted(kinds.items())}))
    return out


# ----------------------------------------------------------------- tables --

#: Methods that reword rather than extract. Evidence spans cannot be found
#: verbatim in their output, so "evidence kept" is not defined for them; only
#: the model's accuracy can score them.
REWRITING_ARMS = frozenset({"stopwords"})

ARM_LABELS = {
    "closed_book": "no context (closed book)",
    "full": "full context",
    "parsimony": "Parsimony",
    "parsimony_v1": "Parsimony (as first frozen)",
    "parsimony_v2": "  with title weighting (measured, not adopted)",
    "parsimony_neural": "Parsimony + MiniLM sentence scoring",
    "parsimony_no_anchors": "  without anchors",
    "parsimony_no_closure": "  without dependency closure",
    "parsimony_no_doc_prior": "  without document prior",
    "parsimony_no_mmr": "  without redundancy penalty",
    "parsimony_no_floor": "  without relevance floor",
    "bm25_topk": "BM25 top sentences",
    "truncate": "truncate to budget",
    "random": "random sentences",
    "stopwords": "stopword removal",
}


def summarise_offline(results: list[ArmResult]) -> list[dict]:
    order: list[str] = []
    agg: dict[str, list[int]] = {}
    for r in results:
        if r.arm not in agg:
            order.append(r.arm)
            agg[r.arm] = [0, 0, 0, 0]
        a = agg[r.arm]
        a[0] += r.context_tokens
        a[1] += r.full_context_tokens
        a[2] += r.evidence_complete
        a[3] += 1
    rows = []
    for arm in order:
        kept, full_tokens, complete, n = agg[arm]
        rows.append({
            "method": arm,
            "items": n,
            "context kept %": f"{100 * kept / (full_tokens or 1):.1f}",
            "evidence kept": "n/a" if arm in REWRITING_ARMS else f"{complete}/{n}",
            "evidence kept %": "n/a" if arm in REWRITING_ARMS else f"{100 * complete / n:.1f}",
        })
    return rows


def real_rows(summaries: list[ArmSummary]) -> list[dict]:
    rows = []
    for s in summaries:
        rows.append({
            "method": s.arm,
            "items": s.n,
            "correct": s.correct,
            "accuracy %": f"{s.accuracy.point:.1f}",
            "95% CI": f"{s.accuracy.low:.1f}-{s.accuracy.high:.1f}",
            "context kept %": f"{s.context_kept_pct:.1f}",
            "prompt tokens": f"{s.mean_prompt_tokens:.0f}",
            "prefill ms": f"{s.mean_prefill_ms:.0f}",
            "wall ms": f"{s.mean_wall_ms:.0f}",
            "lost vs full": s.vs_full_lost if s.arm != "full" else "",
            "gained vs full": s.vs_full_gained if s.arm != "full" else "",
            "McNemar p": f"{s.vs_full_p:.3f}" if s.arm != "full" else "",
        })
    return rows


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
