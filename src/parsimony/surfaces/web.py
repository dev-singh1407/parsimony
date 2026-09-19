"""A local visualiser for what the middleware decides, and what it costs.

Served from the standard library, deliberately. A dashboard framework would
add a hundred megabytes of dependencies to a project whose whole claim is that
it runs on an ordinary laptop with no GPU and no network, and this page has to
work in a room with the wifi off. Three views, all reading the same pipeline
the terminal uses:

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

    # -- heatmap ---------------------------------------------------------

    def compress(self, question: str, text: str, name: str = "pasted text") -> dict:
        from parsimony.modules.m1_context import audit as audit_context
        from parsimony.pipeline.orchestrator import Pipeline

        documents = split_into_documents(text, name)
        pipe = Pipeline(self.cfg, provider=self.provider)
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
