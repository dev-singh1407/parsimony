"""The Ollama seam, tested against a fake Ollama.

The real provider is exercised by the contract suite when Ollama is actually
running. That cannot be relied on: CI has no Ollama, and neither does a fresh
clone. So the wire protocol — streaming NDJSON, num_predict, error frames,
identity — is tested here against a local HTTP server that speaks Ollama's
shape. It runs everywhere, in milliseconds, and it fails when the parsing is
wrong rather than when a service is missing.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from parsimony.core.errors import ProviderError
from parsimony.core.types import GenParams
from parsimony.infra.providers import MockProvider, OllamaProvider, make_provider

# What the fake server will send, set per-test.
SCRIPT: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the test output clean
        pass

    def _send(self, code: int, payload: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, json.dumps(SCRIPT.get("tags", {"models": []})).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            self._send(200, json.dumps(SCRIPT.get("show", {})).encode())
            return
        if self.path == "/api/generate":
            SCRIPT["last_request"] = body
            if SCRIPT.get("http_error"):
                self._send(SCRIPT["http_error"], b'{"error":"boom"}')
                return
            self._send(200, SCRIPT.get("stream", b""))
            return
        self._send(404, b"{}")


@pytest.fixture()
def server():
    SCRIPT.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def tokens(*words, done=True):
    frames = [{"response": w, "done": False} for w in words]
    if done:
        frames.append({"response": "", "done": True})
    return ndjson(*frames)


class TestStreaming:
    def test_yields_one_event_per_token(self, server):
        SCRIPT["stream"] = tokens("Hello", " world", "!")
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        assert [e.text for e in events] == ["Hello", " world", "!"]

    def test_indices_are_monotonic_from_zero(self, server):
        SCRIPT["stream"] = tokens("a", "b", "c")
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        assert [e.index for e in events] == [0, 1, 2]

    def test_timestamps_are_taken_at_receipt(self, server):
        """Not back-filled from a total duration. TTFT is only real if the clock
        is read as each token arrives."""
        SCRIPT["stream"] = tokens("a", "b", "c")
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        stamps = [e.emitted_at_ns for e in events]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) > 1, "distinct arrivals must get distinct timestamps"

    def test_empty_response_frames_are_not_emitted(self, server):
        """Ollama's final frame carries an empty response; emitting it would
        inflate the output token count by one on every single request."""
        SCRIPT["stream"] = ndjson(
            {"response": "x", "done": False},
            {"response": "", "done": False},
            {"response": "", "done": True},
        )
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        assert [e.text for e in events] == ["x"]

    def test_stops_at_done_even_if_more_follows(self, server):
        SCRIPT["stream"] = ndjson(
            {"response": "a", "done": False},
            {"response": "", "done": True},
            {"response": "SHOULD NOT APPEAR", "done": False},
        )
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        assert [e.text for e in events] == ["a"]

    def test_blank_lines_are_skipped(self, server):
        SCRIPT["stream"] = b"\n" + tokens("a", "b") + b"\n"
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=64)))
        assert len(events) == 2


class TestNumPredict:
    def test_is_sent_to_the_server(self, server):
        SCRIPT["stream"] = tokens("a")
        list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=7)))
        assert SCRIPT["last_request"]["options"]["num_predict"] == 7

    def test_is_enforced_even_if_the_server_overruns(self, server):
        """A provider that overran the budget would silently corrupt M5's
        accounting, so the contract is enforced on this side too."""
        SCRIPT["stream"] = tokens(*"abcdefghij", done=False)
        events = list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=3)))
        assert len(events) == 3

    def test_temperature_and_seed_are_sent(self, server):
        """Both are what make generation memoisation bit-exact (ADR-019)."""
        SCRIPT["stream"] = tokens("a")
        list(OllamaProvider(host=server).generate(
            "hi", GenParams(num_predict=4, temperature=0.0, seed=42)))
        opts = SCRIPT["last_request"]["options"]
        assert opts["temperature"] == 0.0 and opts["seed"] == 42

    def test_stop_sequences_are_forwarded_only_when_present(self, server):
        SCRIPT["stream"] = tokens("a")
        list(OllamaProvider(host=server).generate("hi", GenParams(num_predict=4)))
        assert "stop" not in SCRIPT["last_request"]["options"]
        list(OllamaProvider(host=server).generate(
            "hi", GenParams(num_predict=4, stop=("\n\n",))))
        assert SCRIPT["last_request"]["options"]["stop"] == ["\n\n"]


class TestFailuresAreLoud:
    def test_an_error_frame_raises(self, server):
        SCRIPT["stream"] = ndjson({"error": "model not found"})
        with pytest.raises(ProviderError, match="model not found"):
            list(OllamaProvider(host=server).generate("hi", GenParams()))

    def test_a_non_json_line_raises(self, server):
        SCRIPT["stream"] = b"<html>502 Bad Gateway</html>\n"
        with pytest.raises(ProviderError, match="non-JSON"):
            list(OllamaProvider(host=server).generate("hi", GenParams()))

    def test_an_http_error_names_the_status(self, server):
        SCRIPT["http_error"] = 500
        with pytest.raises(ProviderError, match="500"):
            list(OllamaProvider(host=server).generate("hi", GenParams()))

    def test_an_unreachable_host_says_so_and_suggests_the_fix(self):
        p = OllamaProvider(host="http://127.0.0.1:1")
        with pytest.raises(ProviderError, match="ollama serve"):
            list(p.generate("hi", GenParams()))


class TestIdentity:
    def test_digest_is_prefixed_so_it_cannot_look_like_the_mock(self, server):
        SCRIPT["show"] = {"digest": "a" * 64, "details": {"quantization_level": "Q4_K_M"}}
        p = OllamaProvider(host=server)
        assert p.model_digest.startswith("ollama:")
        assert not p.model_digest.startswith("mock")

    def test_quantisation_is_read_from_the_server(self, server):
        SCRIPT["show"] = {"digest": "b" * 64, "details": {"quantization_level": "Q4_K_M"}}
        assert OllamaProvider(host=server).quantisation == "Q4_K_M"

    def test_identity_is_fetched_once(self, server):
        SCRIPT["show"] = {"digest": "c" * 64, "details": {}}
        p = OllamaProvider(host=server)
        assert p.model_digest == p.model_digest == p.model_digest
        assert p.quantisation == "unknown"

    def test_digest_falls_back_when_the_server_omits_it(self, server):
        """Older Ollama builds omit `digest`. Falling back to the model name
        alone would let the memo serve one quantisation's output for another."""
        SCRIPT["show"] = {"details": {"quantization_level": "Q4_K_M"}}
        a = OllamaProvider("m", host=server).model_digest
        SCRIPT["show"] = {"details": {"quantization_level": "Q8_0"}}
        b = OllamaProvider("m", host=server).model_digest
        assert a != b, "a quantisation change must change the digest"


class TestAvailability:
    def test_reports_down_when_nothing_listens(self):
        assert OllamaProvider.available("http://127.0.0.1:1") is False

    def test_reports_up_and_checks_the_model(self, server):
        SCRIPT["tags"] = {"models": [{"name": "qwen2.5:1.5b-instruct"}]}
        assert OllamaProvider.available(server) is True
        assert OllamaProvider.available(server, model="qwen2.5:1.5b-instruct") is True
        assert OllamaProvider.available(server, model="llama3:70b") is False


class TestFactory:
    def test_mock_by_default(self):
        assert isinstance(make_provider(), MockProvider)
        assert isinstance(make_provider("mock"), MockProvider)

    def test_unknown_name_is_refused(self):
        with pytest.raises(ProviderError, match="unknown provider"):
            make_provider("gpt4")

    def test_ollama_refuses_rather_than_falling_back_to_the_mock(self):
        """The worst failure available to this project is a run that believes
        it measured a real model and actually measured a fake one.

        Written as one attempt rather than a probe-then-call: on Windows a
        refused connection costs ~2s, and probing first paid it twice.
        """
        try:
            provider = make_provider("ollama")
        except ProviderError as exc:
            assert "nothing is listening" in str(exc)
        else:
            assert isinstance(provider, OllamaProvider), "must never be a MockProvider"
