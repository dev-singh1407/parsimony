"""Storage: content-addressed blobs and the dual ledger sinks (ADR-005).

JsonlSink is the write path for experiments — append-only, one file per worker,
no contention, crash-safe to the last complete line. SqliteSink serves the
dashboard, which genuinely needs live SQL. `parsimony ledger import` folds JSONL
into the analysis DB after a sweep.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from parsimony.core.ledger import LedgerRow


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BlobStore:
    """Content-addressed text store.

    Prompts and responses are large and highly repetitive across ablation cells
    (the invariant zone is byte-identical by construction), so the ledger keeps
    64-char hashes and the text is stored once.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.txt"

    def put(self, text: str) -> str:
        digest = sha256(text)
        p = self._path(digest)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return digest

    def get(self, digest: str) -> str | None:
        p = self._path(digest)
        return p.read_text(encoding="utf-8") if p.exists() else None


class MemoryBlobStore:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def put(self, text: str) -> str:
        digest = sha256(text)
        self._d[digest] = text
        return digest

    def get(self, digest: str) -> str | None:
        return self._d.get(digest)


class JsonlSink:
    """Append-only. The EXPERIMENT-mode write path."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: LedgerRow) -> None:
        self._fh.write(json.dumps(row.to_dict(), separators=(",", ":")) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Columns promoted to real SQLite columns for querying. Everything else lives in
# the `extra` JSON blob, which is what keeps the schema additive-only (ADR-014):
# a new field never requires a destructive migration.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("request_id", "TEXT PRIMARY KEY"),
    ("conversation_id", "TEXT"),
    ("turn_index", "INTEGER"),
    ("config_hash", "TEXT"),
    ("config_label", "TEXT"),
    ("run_id", "TEXT"),
    ("schema_version", "INTEGER"),
    ("corpus_hash", "TEXT"),
    ("seed", "INTEGER"),
    ("pass_kind", "TEXT"),
    ("created_at", "REAL"),
    ("model_name", "TEXT"),
    ("model_digest", "TEXT"),
    ("tokenizer_id", "TEXT"),
    ("tokens_in_original", "INTEGER"),
    ("tokens_in_final", "INTEGER"),
    ("tokens_out", "INTEGER"),
    ("route_tier", "TEXT"),
    ("cache_consulted", "INTEGER"),
    ("cache_hit", "INTEGER"),
    ("cache_zone", "TEXT"),
    ("gate_fired", "INTEGER"),
    ("prefix_tokens_survived", "INTEGER"),
    ("prefix_ratio", "REAL"),
    ("ttft_ns", "INTEGER"),
    ("tpot_ns", "INTEGER"),
    ("total_ns", "INTEGER"),
    ("middleware_ns", "INTEGER"),
    ("generation_memoised", "INTEGER"),
    ("early_stopped", "INTEGER"),
    ("prompt_sha256", "TEXT"),
    ("response_sha256", "TEXT"),
    ("q_exact_match", "INTEGER"),
)


class SqliteSink:
    """WAL-mode SQLite. The SERVE-mode write path and the analysis DB."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{n} {t}" for n, t in _COLUMNS)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS ledger ({cols}, extra TEXT)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_cfg ON ledger(config_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_run ON ledger(run_id)")
        self._conn.commit()

    def write(self, row: LedgerRow) -> None:
        d = row.to_dict()
        names = [n for n, _ in _COLUMNS]
        values: list[Any] = []
        for n in names:
            v = d.get(n)
            values.append(int(v) if isinstance(v, bool) else v)
        extra = {k: v for k, v in d.items() if k not in set(names)}
        placeholders = ", ".join("?" for _ in names)
        self._conn.execute(
            f"INSERT OR REPLACE INTO ledger ({', '.join(names)}, extra) "
            f"VALUES ({placeholders}, ?)",
            (*values, json.dumps(extra, separators=(",", ":"))),
        )

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> "SqliteSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MemorySink:
    def __init__(self) -> None:
        self.rows: list[LedgerRow] = []

    def write(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def flush(self) -> None: ...

    def close(self) -> None: ...


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Crash-safe read: a truncated final line from an interrupted run is skipped
    rather than raising, so one power failure does not cost the whole sweep."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def import_jsonl(jsonl_paths: list[Path], db_path: Path) -> int:
    """Fold JSONL run files into the analysis database."""
    sink = SqliteSink(db_path)
    n = 0
    try:
        for p in jsonl_paths:
            for d in read_jsonl(p):
                names = [c for c, _ in _COLUMNS]
                values = [int(d[c]) if isinstance(d.get(c), bool) else d.get(c) for c in names]
                extra = {k: v for k, v in d.items() if k not in set(names)}
                sink._conn.execute(
                    f"INSERT OR REPLACE INTO ledger ({', '.join(names)}, extra) "
                    f"VALUES ({', '.join('?' for _ in names)}, ?)",
                    (*values, json.dumps(extra, separators=(",", ":"))),
                )
                n += 1
        sink.flush()
    finally:
        sink.close()
    return n
