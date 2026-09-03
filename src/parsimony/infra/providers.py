"""Model providers.

ADR-007: MockProvider is built first so the orchestrator, ledger timing fields,
the budgeter's early-stop rule and the whole ablation harness are complete and
tested before any model is installed. OllamaProvider lands in Sprint 2 as ~60
lines against this same interface, which already has a contract-test suite.

Every number produced through MockProvider is pipeline-correctness evidence,
NOT performance evidence. The ledger makes that unmistakable: model_digest is
literally "mock:v1".
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Iterator

from parsimony.core.errors import ProviderError
from parsimony.core.types import GenParams, TokenEvent

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b-instruct"

# Realistic for a 1B Q4_K_M model on an i5-class CPU. Used to synthesise
# timestamps, not to actually sleep (a sweep that really waited would take days).
MOCK_TTFT_NS = 120_000_000  # 120 ms to first token
MOCK_TPOT_NS = 65_000_000  # ~15 tokens/second

_CANNED: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("boiling", "water"),
        "Water boils at 100 degrees Celsius at sea level, where atmospheric "
        "pressure is 101.3 kPa. The boiling point falls as altitude increases "
        "because the surrounding pressure drops.",
    ),
    (
        ("capital", "australia"),
        "The capital of Australia is Canberra. It was chosen as a compromise "
        "between Sydney and Melbourne and became the seat of government in 1927.",
    ),
    (
        ("python", "list", "tuple"),
        "A list is mutable and a tuple is immutable. Lists use square brackets "
        "and support item assignment; tuples use parentheses, are hashable when "
        "their contents are, and can therefore be dictionary keys.",
    ),
    (
        ("photosynthesis",),
        "Photosynthesis is the process by which plants convert light energy into "
        "chemical energy. Chlorophyll absorbs light, water is split, and carbon "
        "dioxide is fixed into glucose, releasing oxygen as a by-product.",
    ),
)

_FILLER = (
    "The answer depends on the specific context you are working in.",
    "There are several factors that influence the outcome here.",
    "In most practical cases the standard approach is sufficient.",
    "It is worth considering the trade-offs before deciding.",
    "This behaviour is consistent across the common implementations.",
)


class MockProvider:
    """Deterministic fake model.

    Given identical prompt bytes and parameters it returns identical output,
    which mirrors the temperature-0 property that generation memoisation relies
    on (ADR-019) and makes the whole test suite reproducible.

    It deliberately becomes repetitive in its tail, so that M5's trigram-novelty
    early-stop rule has something real to fire on before Ollama exists.
    """

    def __init__(self, realtime: bool = False, verbosity: int = 6) -> None:
        self._realtime = realtime
        self._verbosity = verbosity

    @property
    def model_name(self) -> str:
        return "mock-1b"

    @property
    def quantisation(self) -> str:
        return "none"

    @property
    def model_digest(self) -> str:
        return "mock:v1"

    @property
    def tokenizer_id(self) -> str:
        return "mock"

    def _body(self, prompt: str) -> str:
        low = prompt.lower()
        for keys, answer in _CANNED:
            if all(k in low for k in keys):
                return answer
        seed = int(hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest(), 16)
        n = 3 + (seed % max(1, self._verbosity - 2))
        picked = [_FILLER[(seed + i) % len(_FILLER)] for i in range(n)]
        # Repetitive tail: small models restate. This is what M5 stops.
        picked.append(picked[0])
        picked.append(picked[0])
        return " ".join(picked)

    def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]:
        text = self._body(prompt)
        pieces = _split_pieces(text)
        start = time.perf_counter_ns()
        for i, piece in enumerate(pieces):
            if i >= params.num_predict:
                break
            if params.stop and any(s in piece for s in params.stop):
                break
            if self._realtime:
                time.sleep((MOCK_TTFT_NS if i == 0 else MOCK_TPOT_NS) / 1e9)
                emitted = time.perf_counter_ns()
            else:
                emitted = start + MOCK_TTFT_NS + i * MOCK_TPOT_NS
            yield TokenEvent(text=piece, index=i, emitted_at_ns=emitted)


def _split_pieces(text: str) -> list[str]:
    """Approximate token pieces: words with their leading space, as a BPE
    tokenizer would emit them."""
    out: list[str] = []
    for i, word in enumerate(text.split(" ")):
        out.append(word if i == 0 else " " + word)
    return out


def make_provider(name: str = "mock", *, model: str | None = None, realtime: bool = False):
    """Build a provider by name.

    Refuses rather than silently falling back to the mock: a run that believes
    it measured a real model but actually measured a fake one produces numbers
    that look real and are not, which is the single worst failure this project
    can have.
    """
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockProvider(realtime=realtime)
    if key == "ollama":
        provider = OllamaProvider(model or DEFAULT_OLLAMA_MODEL)
        if not OllamaProvider.available(provider.host):
            raise ProviderError(
                f"--provider ollama was requested but nothing is listening at "
                f"{provider.host}. Start it with `ollama serve`, or run with "
                f"--provider mock and accept simulated timings."
            )
        return provider
    raise ProviderError(f"unknown provider {name!r} (expected 'mock' or 'ollama')")


class OllamaProvider:
    """A real model over Ollama's local HTTP API.

    Deliberately stdlib-only. The project depends on numpy, tokenizers, typer,
    rich and pytest; adding `requests` or `ollama` to talk to a localhost socket
    would buy nothing and cost the dependency discipline that makes the
    CPU-only claim checkable.

    The identity fields matter more than they look. `model_digest` is what
    separates a real row from a simulated one in the ledger and what keys the
    generation memo (ADR-019) — a memo keyed on a name rather than a digest
    would serve one quantisation's output for another's.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        timeout: float = 300.0,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._info: dict | None = None
        # Server-reported timings from the most recent generate(). Ollama
        # separates prompt_eval (prefill) from eval (decode), which wall-clock
        # TTFT cannot: TTFT conflates prefill with HTTP and scheduling overhead.
        # Research gap 2 is the prefill/decode split, so the authoritative
        # numbers are worth keeping rather than re-deriving.
        self.last_stats: dict = {}

    # -- identity ----------------------------------------------------------

    def _post(self, path: str, body: dict, *, stream: bool = False, timeout=None):
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            raise ProviderError(f"ollama {path} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"cannot reach ollama at {self.host} ({exc.reason}). "
                f"Is it running? Try: ollama serve"
            ) from exc
        return resp if stream else json.loads(resp.read())

    @property
    def info(self) -> dict:
        """Model metadata, fetched once. Cheap, but not free, and the ledger
        asks for it on every row."""
        if self._info is None:
            self._info = self._post("/api/show", {"model": self.model}, timeout=30.0)
        return self._info

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def quantisation(self) -> str:
        return (self.info.get("details") or {}).get("quantization_level") or "unknown"

    @property
    def model_digest(self) -> str:
        """Content digest, not a label.

        Ollama reports the manifest digest; we prefix it so a ledger row can
        never be confused with the mock's "mock:v1", and truncate because the
        full 64 hex chars buy no discrimination in a ledger column.
        """
        raw = self.info.get("digest") or ""
        if not raw:
            # Older Ollama builds omit `digest` from /api/show. Fall back to a
            # hash of the fields that actually determine the weights, so the
            # memo key still changes when the model does.
            details = self.info.get("details") or {}
            material = json.dumps(
                {"model": self.model, **{k: details.get(k) for k in sorted(details)}},
                sort_keys=True,
            )
            raw = hashlib.blake2b(material.encode(), digest_size=16).hexdigest()
        return f"ollama:{raw[:16]}"

    @property
    def tokenizer_id(self) -> str:
        return self.model

    # -- availability ------------------------------------------------------

    @classmethod
    def available(cls, host: str = DEFAULT_OLLAMA_HOST, *, model: str | None = None) -> bool:
        """Is Ollama up, and does it have this model pulled?

        Used to skip the real-provider tests rather than fail them, so the suite
        stays green on a machine that has never installed Ollama — including CI.
        """
        try:
            with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3.0) as r:
                tags = json.loads(r.read())
        except Exception:
            return False
        if model is None:
            return True
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)

    # -- measurement -------------------------------------------------------

    def probe(self, prompt: str, *, num_predict: int = 4) -> dict:
        """One non-streaming call, returning the server's own timings.

        Separate from generate() on purpose. generate() stops yielding once
        num_predict is reached, which means it can break before the terminating
        frame arrives — and that frame is where the timings live. Draining the
        stream just to collect them would make every real generation pay for a
        measurement it does not use.

        Returns prompt_eval_* (prefill) and eval_* (decode) separately, which is
        the whole of research gap 2 and is not recoverable from wall-clock TTFT.
        """
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0.0, "seed": 0},
        }
        if self.num_ctx is not None:
            body["options"]["num_ctx"] = self.num_ctx
        out = self._post("/api/generate", body)
        if out.get("error"):
            raise ProviderError(f"ollama: {out['error']}")
        self.last_stats = {
            k: out[k]
            for k in (
                "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration",
                "load_duration", "total_duration", "done_reason",
            )
            if k in out
        }
        return self.last_stats

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, params: GenParams) -> Iterator[TokenEvent]:
        options = {
            "num_predict": params.num_predict,
            "temperature": params.temperature,
            "seed": params.seed,
        }
        if params.stop:
            options["stop"] = list(params.stop)
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        resp = self._post(
            "/api/generate",
            {"model": self.model, "prompt": prompt, "stream": True, "options": options},
            stream=True,
        )

        index = 0
        self.last_stats = {}
        try:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"ollama sent a non-JSON line: {line[:120]!r}") from exc
                if chunk.get("error"):
                    raise ProviderError(f"ollama: {chunk['error']}")

                piece = chunk.get("response", "")
                if piece:
                    # perf_counter_ns AT RECEIPT. This is the whole point of
                    # streaming rather than taking the response in one blob:
                    # TTFT and TPOT are only real if measured as tokens arrive.
                    yield TokenEvent(text=piece, index=index, emitted_at_ns=time.perf_counter_ns())
                    index += 1
                    # Ollama honours num_predict, but a provider that overran it
                    # would silently break the budgeter's accounting, so the
                    # contract is enforced here too.
                    if index >= params.num_predict:
                        break
                if chunk.get("done"):
                    self.last_stats = {
                        k: chunk[k]
                        for k in (
                            "prompt_eval_count", "prompt_eval_duration",
                            "eval_count", "eval_duration",
                            "load_duration", "total_duration", "done_reason",
                        )
                        if k in chunk
                    }
                    break
        finally:
            resp.close()
