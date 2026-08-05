"""Generation memoisation (ADR-019) and sweep resumability.

WHY THIS EXISTS
---------------
The report's experimental protocol costs an estimated ~36 days of continuous
CPU (docs/05-evaluation-harness.md). The runway allows about ten.

At temperature 0 decoding is greedy: identical prompt bytes plus an identical
pinned model digest plus identical generation parameters produce identical
output. Across 16 cells the prompt repetition rate is high -- every pair of
cells differing only in M2's flag produces byte-identical prompts on every cache
miss. So the memo returns exactly what the model would have returned. It is
bit-exact, not an approximation, and cannot bias any result.

WHY IT IS DANGEROUS, AND HOW THAT IS CONTAINED
----------------------------------------------
A memo hit takes microseconds where generation takes seconds, so a memoised run
says nothing about latency. Containment is structural rather than procedural:

  * every ledger row carries `generation_memoised`, and latency analysis filters
    on it;
  * the sweep splits into a quality pass (memo on) and a timing pass (memo off),
    and every row carries `pass_kind`;
  * the memo is only consulted in Mode.EXPERIMENT.

The key includes the early-stop configuration, not just num_predict. Two cells
can share a num_predict while differing in whether the stop rule runs, and the
stopped output differs -- keying on num_predict alone would serve one cell's
truncation to another.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from parsimony.core.config import ParsimonyConfig
from parsimony.core.types import GenParams


@dataclass(frozen=True, slots=True)
class MemoEntry:
    text: str
    early_stopped: bool


def gen_params_hash(params: GenParams, cfg: ParsimonyConfig) -> str:
    payload = {
        "num_predict": params.num_predict,
        "temperature": params.temperature,
        "seed": params.seed,
        "stop": list(params.stop),
        # The early-stop rule changes the emitted text, so it is part of the key.
        "early_stop": bool(cfg.enables("M5") and cfg.budget.early_stop),
        "novelty_window": cfg.budget.novelty_window,
        "novelty_threshold": cfg.budget.novelty_threshold,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def memo_key(prompt: str, model_digest: str, params_hash: str) -> str:
    payload = f"{hashlib.sha256(prompt.encode()).hexdigest()}:{model_digest}:{params_hash}"
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


class GenerationMemo:
    """Bit-exact memo for temperature-0 generation. EXPERIMENT mode only."""

    def __init__(self, path: Path | None = None) -> None:
        self._mem: dict[str, MemoEntry] = {}
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memo "
                "(key TEXT PRIMARY KEY, text TEXT, early_stopped INTEGER)"
            )
            self._conn.commit()

    def get(self, key: str) -> MemoEntry | None:
        entry = self._mem.get(key)
        if entry is None and self._conn is not None:
            row = self._conn.execute(
                "SELECT text, early_stopped FROM memo WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                entry = MemoEntry(row[0], bool(row[1]))
                self._mem[key] = entry
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, key: str, entry: MemoEntry) -> None:
        self._mem[key] = entry
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO memo (key, text, early_stopped) VALUES (?, ?, ?)",
                (key, entry.text, int(entry.early_stopped)),
            )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 100.0 * self.hits / total if total else 0.0

    def flush(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None


class CompletionLog:
    """Per-(cell, seed, conversation) markers so an interrupted sweep resumes.

    A 16-hour unattended run WILL be interrupted -- sleep, update, thermal
    shutdown, power. Without this, an interruption at hour 14 costs 14 hours.
    With it, it costs one conversation.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            self._done = {
                line.strip() for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        self._fh = self.path.open("a", encoding="utf-8")

    @staticmethod
    def marker(config_hash: str, seed: int, conversation_id: str) -> str:
        return f"{config_hash}:{seed}:{conversation_id}"

    def is_done(self, config_hash: str, seed: int, conversation_id: str) -> bool:
        return self.marker(config_hash, seed, conversation_id) in self._done

    def mark(self, config_hash: str, seed: int, conversation_id: str) -> None:
        key = self.marker(config_hash, seed, conversation_id)
        if key in self._done:
            return
        self._done.add(key)
        self._fh.write(key + "\n")
        self._fh.flush()  # flushed per conversation: the point is crash safety

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CompletionLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
