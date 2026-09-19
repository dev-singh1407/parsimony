"""A local visualiser for what the middleware decides, and what it costs.

Served from the standard library, deliberately. A dashboard framework would
add a hundred megabytes of dependencies to a project whose whole claim is that
it runs on an ordinary laptop with no GPU and no network, and this page has to
work in a room with the wifi off. Four views, all reading the same pipeline
the terminal uses:

  Pipeline  the request crossing all eleven stages, with a transport you can
            scrub: step or play through the run and watch YOUR OWN text lose
            the sentences each stage decided it did not need, with the score
            and the reason on every one. The stack shows which layer is acting
            and how much of the prompt it hands on; the inspector shows what
            that particular layer reasoned about -- relevance bars for the
            context tier, similarity for the cache, complexity against its
            threshold for the router -- rather than one generic dump for all
            eleven. The table underneath is the terminal's own, same labels and
            same sentences, from the same `_reason()`.

            The pacing is the one invented thing and is labelled: the
            middleware takes ~100 ms, too fast to watch, so the transport holds
            each stage for a beat. Every duration shown is the stage's own
            measured one, and the segment widths are its real share of the time.
  Heatmap   paste or upload a document, ask a question, and see every sentence
            shaded by the score the encoder gave it, with the decision beside
            it. This is the encoder's judgement made visible -- including where
            it is wrong, which is the point of showing it rather than asserting
            accuracy.
  A/B       the same question with the layers off and on, against the real
            model, streaming as it generates, plotted on one chart.
            SEQUENTIALLY, not simultaneously: two generations on one CPU
            compete for the same cores, and a "race" run that way measures
            contention rather than compression.
  Demo      one screen, two counters -- tokens pruned and seconds saved this
            session -- large enough to read from the back of a room.

Nothing here is a second implementation of anything: every figure comes from
`Pipeline`, `m1_context.audit` and the runtime's own counters.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from parsimony.core.config import ParsimonyConfig, baseline, full_stack
from parsimony.core.types import split_into_documents

PAGE = Path(__file__).parent / "web_page.html"
SAMPLE = "examples/staff-handbook.md"

#: Served beside the page. Split out because a single-file page big enough to
#: hold a scrubbable pipeline view stops being reviewable, and these are still
#: local files -- the page loads nothing from a network, which is the property
#: that matters.
ASSETS = {
    "/app.css": ("web_app.css", "text/css; charset=utf-8"),
    "/app.js": ("web_app.js", "text/javascript; charset=utf-8"),
}


def sample_document() -> Path | None:
    """The handbook, whether parsimony was started from the repo or installed."""
    here = Path(__file__).resolve()
    for root in (Path.cwd(), *here.parents[2:5]):
        candidate = root / SAMPLE
        if candidate.exists():
            return candidate
    return None


@dataclass
class SessionTotals:
    """What this session has actually avoided, in tokens and in measured seconds."""

    requests: int = 0
    tokens_written: int = 0
    tokens_sent: int = 0
    seconds_measured: float = 0.0     # prefill actually spent
    seconds_full: float = 0.0         # prefill the untrimmed prompt would have cost
    priced_here: int = 0              # requests whose rate this machine actually timed
    lock: object = field(default_factory=threading.Lock)

    def record(self, *, written: int, sent: int, prefill_s: float, rate_ms: float,
               timed_here: bool) -> None:
        """One request. `timed_here` says whether the rate is this machine's own.

        A mock model, or a prompt the runtime served from its cache, gives no
        honest rate: the fallback constant is the project's recorded figure, and
        seconds derived from it are an estimate. Counting the two together and
        calling the total "measured" would be the easiest lie on the screen.
        """
        with self.lock:
            self.requests += 1
            self.priced_here += int(timed_here)
            self.tokens_written += written
            self.tokens_sent += sent
            self.seconds_measured += prefill_s
            self.seconds_full += written * rate_ms / 1000.0

    def snapshot(self) -> dict:
        with self.lock:
            pruned = self.tokens_written - self.tokens_sent
            return {
                "requests": self.requests,
                "tokens_pruned": pruned,
                "tokens_written": self.tokens_written,
                "tokens_sent": self.tokens_sent,
                "seconds_saved": max(0.0, self.seconds_full - self.seconds_measured),
                "timed_here": self.priced_here == self.requests and self.requests > 0,
                "percent": (100.0 * pruned / self.tokens_written) if self.tokens_written else 0.0,
            }


class Visualiser:
    """The pipeline behind the page. One instance, shared by every request."""

    def __init__(self, cfg: ParsimonyConfig | None = None, provider_name: str = "auto") -> None:
        from parsimony.infra.embedding import best_config
        from parsimony.infra.providers import (DEFAULT_OLLAMA_MODEL, MockProvider,
                                               OllamaProvider, make_provider)

        self.cfg = best_config(cfg or full_stack())
        if provider_name == "auto":
            live = OllamaProvider.available(model=DEFAULT_OLLAMA_MODEL)
            self.provider = make_provider("ollama") if live else MockProvider()
        else:
            self.provider = make_provider(provider_name)
        self.simulated = self.provider.model_digest.startswith("mock")
        self.totals = SessionTotals()
        # One cache for the whole session, not one per request. A Pipeline
        # builds its own when it is not given one, which meant the Memory layer
        # could never hit here however many times you asked the same question --
        # it was being handed an empty cache every time, and reporting the miss
        # honestly. Sharing it is what a real session does.
        from parsimony.modules.m2_cache import SemanticCache

        self.cache = SemanticCache(self.cfg.cache.ttl_seconds,
                                   max_entries=self.cfg.cache.max_entries)

    # -- heatmap ---------------------------------------------------------

    def compress(self, question: str, text: str, name: str = "pasted text") -> dict:
        from parsimony.modules.m1_context import audit as audit_context
        from parsimony.pipeline.orchestrator import Pipeline

        documents = split_into_documents(text, name)
        pipe = Pipeline(self.cfg, provider=self.provider, cache=self.cache)
        started = time.perf_counter()
        ctx = pipe.build_context(question, documents=documents)
        report = audit_context(ctx, self.cfg)
        took_ms = (time.perf_counter() - started) * 1000
        encoder_ms = getattr(ctx.derived, "embed_ns", 0) / 1e6
        return {
            "question": question,
            "tokens_before": report.tokens_before,
            "tokens_after": report.tokens_after,
            "removed_pct": report.removed_pct,
            "stopped_by": report.stopped_by,
            "off_topic": report.off_topic,
            "coverage": report.coverage,
            "encoder_ms": encoder_ms,
            "decide_ms": max(0.0, took_ms - encoder_ms),
            "encoder": self.cfg.embedder_id,
            "anchors": sorted(str(a) for a in report.anchors),
            "units": [
                {"text": u.text, "source": u.source_label, "score": round(u.score, 4),
                 "kept": u.kept, "tag": u.tag, "detail": u.detail, "tokens": u.tokens}
                for u in report.units
            ],
        }

    # -- the pipeline, stage by stage ------------------------------------

    #: What each stage is, in words a reader who has never seen the code can use.
    #: `role` groups them on the graph; `does` is the one-line claim the node makes
    #: before it has run, so the picture is legible while it is still empty.
    STAGE_LABELS = {
        "m6a_deterministic": ("Deterministic router", "route",
                              "Answer without the model where arithmetic or a lookup will do"),
        "m2_cache": ("Semantic cache", "reuse",
                     "Has this been asked before, closely enough to reuse the answer?"),
        "m3_history": ("History selector", "history",
                       "Which earlier turns does this question actually need?"),
        "m3_arrange": ("History arranger", "history",
                       "Order the surviving turns so the prompt prefix stays stable"),
        "m1_context": ("Context compressor", "compress",
                       "Keep only the sentences of the documents this question needs"),
        "m1_tier1": ("Tier 1 — normalise", "compress",
                     "Lossless: collapse whitespace, strip boilerplate"),
        "m1_tier2": ("Tier 2 — deduplicate", "compress",
                     "Drop sentences that restate one already kept"),
        "m1_tier3": ("Tier 3 — rewrite", "compress",
                     "Shorten phrasing, and reject any edit that costs more than it saves"),
        "m4_assembler": ("Prefix-stable assembler", "assemble",
                         "Pin invariants to the prompt head so the KV cache survives"),
        "m5_budgeter": ("Output budgeter", "budget",
                        "Classify the answer and cap how long it may run"),
        "m6b_router": ("Escalation router", "route",
                       "Is this complex enough to deserve a larger model?"),
    }

    @staticmethod
    def _split(text: str) -> list[str]:
        """Sentences, in order, as the compressor's own splitter sees them."""
        from parsimony.infra.nlp import split_sentences

        out: list[str] = []
        for line in (text or "").splitlines():
            out.extend(s.strip() for s in split_sentences(line) if s.strip())
        return out

    @classmethod
    def _life_of_every_sentence(cls, states: list[tuple[str, str]],
                                question: str = "") -> list[dict]:
        """Every sentence the request arrived with, and the stage that removed it.

        This is what makes the page show the tiers WORKING rather than reporting
        that they ran: the reader watches their own document lose sentences, one
        stage at a time, with the stage that took each one named. Matching is by
        normalised content because M1 regroups documents and M4 moves blocks, so
        a sentence that merely moved must not read as a sentence that was cut.
        """
        from collections import Counter

        def key(s: str) -> str:
            return " ".join(s.split()).casefold()

        # The payload carries the question too. This panel is about the text the
        # reader attached, and showing their own question inside it as a
        # sentence that "survived compression" is just confusing.
        asked = {key(q) for q in cls._split(question)} if question else set()
        original = [s for s in cls._split(states[0][1]) if key(s) not in asked]
        units = [{"text": s, "removed_at": None, "order": i} for i, s in enumerate(original)]
        for stage_name, text in states[1:]:
            present = Counter(key(s) for s in cls._split(text))
            for unit in units:
                if unit["removed_at"] is not None:
                    continue
                k = key(unit["text"])
                if present[k] > 0:
                    present[k] -= 1
                else:
                    unit["removed_at"] = stage_name
        return units

    def plan(self, cfg: ParsimonyConfig) -> list[dict]:
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(cfg, provider=self.provider)
        out = []
        for planned in pipe.registry.ordered(cfg):
            stage = getattr(planned, "stage", planned)
            name = getattr(stage, "name", "")
            label, role, does = self.STAGE_LABELS.get(
                name, (name.replace("_", " "), "other", ""))
            out.append({"name": name, "module": getattr(stage, "module_id", ""),
                        "label": label, "role": role, "does": does})
        return out

    @staticmethod
    def _trim(evidence) -> dict:
        """Evidence is for reading, so keep it small and JSON-safe."""
        small = {}
        for key, value in dict(evidence or {}).items():
            if isinstance(value, (int, float, bool)) or value is None:
                small[key] = value
            elif isinstance(value, str):
                small[key] = value[:200]
            elif isinstance(value, (list, tuple)):
                small[key] = [str(v)[:80] for v in list(value)[:6]]
            else:
                small[key] = str(value)[:120]
        return small

    def run_pipeline(self, question: str, text: str, emit, *, name: str = "pasted text",
                     history=()) -> dict:
        """One request, reported stage by stage as the orchestrator commits each."""
        from parsimony.infra.providers import kv_bytes_per_token
        from parsimony.pipeline.orchestrator import Pipeline
        from parsimony.surfaces.cli.live import LiveTurn

        documents = split_into_documents(text, name) if text.strip() else ()
        cfg = self.cfg
        pipe = Pipeline(cfg, provider=self.provider, cache=self.cache,
                        capture_text=True)
        stages = self.plan(cfg)
        view = LiveTurn(question, [s["name"] for s in stages], cfg=cfg, cache=pipe.cache,
                        simulated=self.simulated)
        emit("plan", {"stages": stages, "question": question,
                      "model": self.provider.model_name, "simulated": self.simulated,
                      "encoder": cfg.embedder_id})
        started = time.perf_counter()
        seen: list[dict] = []

        class _Observer:
            def begin(self, ctx, tokens_original):
                view.begin(ctx, tokens_original)
                emit("begin", {"tokens": tokens_original,
                               "documents": len(getattr(ctx, "documents", ()) or ()),
                               "history_turns": len(getattr(ctx, "history", ()) or ())})

            def stage(self, trace):
                view.stage(trace)
                payload = {
                    "name": trace.name, "module": trace.module_id,
                    "outcome": trace.outcome.value,
                    "before": trace.tokens_before, "after": trace.tokens_after,
                    "ms": trace.duration_ns / 1e6, "rationale": trace.rationale,
                    "evidence": Visualiser._trim(trace.evidence),
                    "gate_events": [g.invariant_class for g in trace.gate_events],
                }
                seen.append(payload)
                emit("stage", payload)

            def prompt(self, text_, tokens):
                view.prompt(text_, tokens)
                emit("prompt", {"tokens": tokens, "text": text_[:4000]})

            def token(self, event):
                view.token(event)
                emit("token", {"n": len(view.pieces), "text": event.text,
                               "t": time.perf_counter() - started})

            def stopped(self, reason):
                view.stopped(reason)
                emit("stopped", {"reason": str(reason)})

            def generated(self, stats):
                view.generated(stats)

        outcome = pipe.run(question, history, documents=documents, observer=_Observer())
        row = outcome.row

        # -- what each stage did to the text, after the fact --------------
        #
        # TextDelta carries before/after for every stage that PROPOSED something;
        # a stage that did nothing leaves the text where it was, so the state is
        # carried forward. The result is the payload at every stage boundary,
        # which is what the document view scrubs through.
        from parsimony.surfaces.cli.explain import _NAME, _reason

        deltas = {d.stage: d for d in outcome.text_deltas}
        first = next((d.before for d in outcome.text_deltas), "")
        states: list[tuple[str, str]] = [("__start__", first)]
        current = first
        for stage in seen:
            delta = deltas.get(stage["name"])
            if delta is not None and not delta.reverted:
                current = delta.after
            states.append((stage["name"], current))
        units = self._life_of_every_sentence(states, question) if first else []

        # The compressor's own scores, joined onto the sentences by content, so
        # each block can say why it went as well as when.
        scored: dict[str, dict] = {}
        if documents:
            try:
                from parsimony.modules.m1_context import audit as audit_context

                report = audit_context(
                    pipe.build_context(question, history, documents=documents), cfg)
                for unit in report.units:
                    scored[" ".join(unit.text.split()).casefold()] = {
                        "score": round(unit.score, 4), "tag": unit.tag,
                        "detail": unit.detail, "source": unit.source_label,
                    }
            except Exception:
                scored = {}
        for unit in units:
            unit.update(scored.get(" ".join(unit["text"].split()).casefold(), {}))

        # -- the terminal's own table, same labels and same sentences ------
        by_name = {t.name: t for t in outcome.traces}
        layers = []
        for stage in stages:
            trace = by_name.get(stage["name"])
            label, job = _NAME.get(stage["name"], (stage["name"], ""))
            if trace is None:
                layers.append({"name": stage["name"], "label": label, "job": job,
                               "outcome": "not_reached", "why": "never reached",
                               "ms": None, "delta": None, "module": stage["module"]})
                continue
            change = trace.tokens_after - trace.tokens_before
            layers.append({
                "name": stage["name"], "label": label, "job": job,
                "module": trace.module_id, "outcome": trace.outcome.value,
                "ms": trace.duration_ns / 1e6,
                "delta": change,
                "why": _reason(trace, deltas.get(stage["name"]), pipe.cache, cfg),
                "removed": sum(1 for u in units if u["removed_at"] == stage["name"]),
            })
        prefill = view.reading_seconds or 0.0
        rate = view.ms_per_token
        timed_here = not (self.simulated or view.reused_earlier_work) and bool(prefill)
        self.totals.record(written=row.tokens_in_original, sent=row.tokens_in_final,
                           prefill_s=prefill, rate_ms=rate, timed_here=timed_here)
        kv = kv_bytes_per_token(self.provider)
        removed = max(0, row.tokens_in_original - row.tokens_in_final)
        summary = {
            "answer": outcome.response.strip(),
            "tokens": {"original": row.tokens_in_original, "final": row.tokens_in_final,
                       "removed": removed, "out": len(view.pieces),
                       "budget": row.tokens_out_budget,
                       "ratio": (row.tokens_in_final / row.tokens_in_original)
                       if row.tokens_in_original else 1.0},
            "route": {"tier": row.route_tier, "served_by": outcome.served_by},
            "cache": {"consulted": row.cache_consulted, "hit": row.cache_hit,
                      "zone": row.cache_zone,
                      "top_k": [[k, round(v, 3)] for k, v in (row.cache_top_k or ())][:3]},
            "gate": {"fired": row.gate_fired,
                     "events": [g.invariant_class for g in (row.gate_events or ())]},
            "timing": {"middleware_ms": row.middleware_ns / 1e6,
                       "prefill_s": prefill, "ms_per_token": rate,
                       "timed_here": timed_here,
                       "reused": view.reused_earlier_work,
                       "saved_s": removed * rate / 1000.0,
                       "early_stopped": row.early_stopped},
            "kv": ({"per_token": kv[0], "how": kv[1],
                    "bytes_saved": kv[0] * removed} if kv else None),
            "session": self.totals.snapshot(),
            "stages": seen,
            "layers": layers,
            "units": units,
            "prompt": getattr(outcome, "prompt_text", "") or "",
        }
        emit("done", summary)
        return summary

    # -- A/B -------------------------------------------------------------

    def warm_up(self) -> bool:
        """Get the model resident before either arm is timed.

        Ollama loads the weights on the first request after an idle period, and
        that load lands on whichever arm runs first -- several seconds of disk
        and RAM that have nothing to do with compression. Measured without this,
        the first arm looks slower than it is, and since the compressed arm runs
        first the error would flatter the baseline. One throwaway token fixes it.
        """
        from parsimony.core.types import GenParams

        if self.simulated:
            return False
        try:
            self.provider.complete("hello", GenParams(num_predict=1, temperature=0.0, seed=1))
            return True
        except Exception:
            return False

    def arms(self, question: str, text: str, name: str = "pasted text"):
        """(label, config, documents) for the two arms, compressed one first."""
        documents = split_into_documents(text, name)
        return [("parsimony", self.cfg, documents), ("baseline", baseline(), documents)]

    def run_arm(self, label: str, cfg: ParsimonyConfig, documents, question: str, emit) -> dict:
        """Run one arm, emitting events as the model produces them."""
        from parsimony.pipeline.orchestrator import Pipeline
        from parsimony.surfaces.cli.live import LiveTurn

        pipe = Pipeline(cfg, provider=self.provider)
        stage_names = [s.name for s in pipe.registry.ordered(cfg)]
        view = LiveTurn(question, stage_names, cfg=cfg, cache=pipe.cache,
                        simulated=self.simulated)
        started = time.perf_counter()

        class _Observer:
            def begin(self, ctx, tokens_original):
                view.begin(ctx, tokens_original)
                emit("start", {"arm": label, "tokens_original": tokens_original})

            def stage(self, trace):
                view.stage(trace)

            def prompt(self, text_, tokens):
                view.prompt(text_, tokens)
                emit("prompt", {"arm": label, "tokens": tokens})

            def token(self, event):
                view.token(event)
                emit("token", {"arm": label, "n": len(view.pieces),
                               "t": time.perf_counter() - started,
                               "text": event.text})

            def stopped(self, reason):
                view.stopped(reason)

            def generated(self, stats):
                view.generated(stats)

        outcome = pipe.run(question, documents=documents, observer=_Observer())
        prefill = view.reading_seconds or 0.0
        rate = view.ms_per_token
        timed_here = not (self.simulated or view.reused_earlier_work) and bool(prefill)
        self.totals.record(written=outcome.row.tokens_in_original,
                           sent=outcome.row.tokens_in_final,
                           prefill_s=prefill, rate_ms=rate, timed_here=timed_here)
        result = {
            "arm": label,
            "prompt_tokens": outcome.row.tokens_in_final,
            "tokens_original": outcome.row.tokens_in_original,
            "prefill_s": prefill,
            "ttft_s": (view.t_first - view.t_prompt) if view.t_first and view.t_prompt else None,
            "answer_tokens": len(view.pieces),
            "total_s": time.perf_counter() - started,
            "answer": outcome.response.strip(),
            "reused": view.reused_earlier_work,
            "ms_per_token": rate,
        }
        emit("arm_done", result)
        return result


def make_handler(vis: Visualiser):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # quiet: the terminal belongs to the user
            pass

        # -- helpers ----------------------------------------------------
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- routes -----------------------------------------------------
        def do_GET(self) -> None:
            route = urlparse(self.path)
            if route.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route.path in ASSETS:
                name, kind = ASSETS[route.path]
                self._send((Path(__file__).parent / name).read_bytes(), kind)
            elif route.path == "/api/session":
                self._json(vis.totals.snapshot())
            elif route.path == "/api/sample":
                sample = sample_document()
                if sample is None:
                    self._json({"error": "no sample document on this machine"}, 404)
                    return
                self._json({"text": sample.read_text(encoding="utf-8"),
                            "question": "What is the annual travel budget for the "
                                        "Tallinn office?"})
            elif route.path == "/api/state":
                self._json({"model": vis.provider.model_name, "simulated": vis.simulated,
                            "encoder": vis.cfg.embedder_id})
            elif route.path == "/api/plan":
                self._json({"stages": vis.plan(vis.cfg)})
            elif route.path == "/api/pipeline":
                self._stream(parse_qs(route.query), vis.run_pipeline)
            elif route.path == "/api/race":
                self._race(parse_qs(route.query))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            route = urlparse(self.path)
            if route.path == "/api/compress":
                payload = self._body()
                question = (payload.get("question") or "").strip()
                text = payload.get("text") or ""
                if not question or not text.strip():
                    self._json({"error": "a question and some text are both required"}, 400)
                    return
                try:
                    self._json(vis.compress(question, text, payload.get("name") or "pasted text"))
                except Exception as exc:                      # a visualiser must not 500 silently
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            else:
                self._json({"error": "not found"}, 404)

        def _sse_open(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(event: str, data: dict) -> None:
                self.wfile.write(
                    f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
                self.wfile.flush()

            return emit

        def _stream(self, query: dict, fn) -> None:
            """Run one streaming job, reporting a failure to the page rather than
            dropping the connection and leaving a graph frozen mid-animation."""
            question = (query.get("question", [""])[0]).strip()
            text = query.get("text", [""])[0]
            if not question:
                self._json({"error": "a question is required"}, 400)
                return
            emit = self._sse_open()
            try:
                fn(question, text, emit)
            except Exception as exc:
                emit("failed", {"error": f"{type(exc).__name__}: {exc}"})

        def _race(self, query: dict) -> None:
            question = (query.get("question", [""])[0]).strip()
            text = query.get("text", [""])[0]
            if not question or not text.strip():
                self._json({"error": "a question and some text are both required"}, 400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(event: str, data: dict) -> None:
                chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()

            try:
                warmed = vis.warm_up()
                emit("note", {"text": "Run one after the other, not at once: two generations "
                                      "on one CPU compete for the same cores.",
                              "warmed": warmed})
                results = []
                for label, cfg, documents in vis.arms(question, text):
                    results.append(vis.run_arm(label, cfg, documents, question, emit))
                emit("done", {"arms": results, "session": vis.totals.snapshot()})
            except Exception as exc:
                emit("failed", {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8501, *, cfg: ParsimonyConfig | None = None,
          provider: str = "auto", open_browser: bool = True) -> None:
    vis = Visualiser(cfg, provider)
    server = ThreadingHTTPServer((host, port), make_handler(vis))
    url = f"http://{host}:{port}/"
    if open_browser:
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"Parsimony visualiser on {url}  (model: {vis.provider.model_name}, "
          f"encoder: {vis.cfg.embedder_id})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
