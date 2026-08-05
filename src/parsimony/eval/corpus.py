"""Corpus loading and freezing.

The corpus hash goes into every ledger row. A tagged commit whose timestamp
predates the first sweep is the only real answer to 'was the corpus tuned to the
system?' — a criticism the report's own risk register anticipates (ADR-015).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "corpus" / "conversations.jsonl"


@dataclass(frozen=True, slots=True)
class Conversation:
    conversation_id: str
    cls: str
    user_turns: tuple[str, ...]
    notes: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.user_turns)


@dataclass(frozen=True, slots=True)
class Corpus:
    conversations: tuple[Conversation, ...]
    corpus_hash: str
    path: Path

    def __len__(self) -> int:
        return len(self.conversations)

    @property
    def n_requests(self) -> int:
        return sum(c.n_turns for c in self.conversations)

    def by_class(self) -> dict[str, list[Conversation]]:
        out: dict[str, list[Conversation]] = {}
        for c in self.conversations:
            out.setdefault(c.cls, []).append(c)
        return out

    def subset(self, n: int) -> "Corpus":
        """Stratified subset: proportional across classes, deterministic order.

        Used for the timing pass, which is unmemoised and therefore the
        expensive one (docs/05-evaluation-harness.md 1).
        """
        grouped = self.by_class()
        per_class = max(1, n // max(1, len(grouped)))
        picked: list[Conversation] = []
        for cls in sorted(grouped):
            picked.extend(grouped[cls][:per_class])
        return Corpus(tuple(picked[:n]), self.corpus_hash, self.path)


@dataclass(frozen=True, slots=True)
class AdversarialPair:
    pair_id: str
    operative: str
    a: str
    b: str
    answers_differ: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class GoldItem:
    gold_id: str
    question: str
    gold_answer: str
    match: str
    tolerance: float = 0.0
    acceptable_variants: tuple[str, ...] = ()


def load_adversarial(path: Path | str | None = None) -> tuple[AdversarialPair, ...]:
    p = Path(path) if path else DEFAULT_CORPUS.parent / "adversarial_pairs.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out.append(
            AdversarialPair(
                pair_id=o["pair_id"],
                operative=o["operative"],
                a=o["a"],
                b=o["b"],
                answers_differ=bool(o["answers_differ"]),
                notes=o.get("notes", ""),
            )
        )
    return tuple(out)


def load_gold(path: Path | str | None = None) -> tuple[GoldItem, ...]:
    p = Path(path) if path else DEFAULT_CORPUS.parent / "gold.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out.append(
            GoldItem(
                gold_id=o["gold_id"],
                question=o["question"],
                gold_answer=o["gold_answer"],
                match=o.get("match", "exact"),
                tolerance=float(o.get("tolerance", 0.0)),
                acceptable_variants=tuple(o.get("acceptable_variants", ())),
            )
        )
    return tuple(out)


def load_corpus(path: Path | str | None = None) -> Corpus:
    p = Path(path) if path else DEFAULT_CORPUS
    if not p.exists():
        raise FileNotFoundError(f"corpus not found: {p}")

    raw = p.read_bytes()
    corpus_hash = hashlib.sha256(raw).hexdigest()[:16]

    conversations: list[Conversation] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        user_turns = tuple(
            t["content"] for t in obj["turns"] if t.get("role") == "user" and t.get("content")
        )
        if not user_turns:
            continue
        conversations.append(
            Conversation(
                conversation_id=obj["conversation_id"],
                cls=obj.get("class", "unknown"),
                user_turns=user_turns,
                notes=obj.get("notes", ""),
            )
        )
    return Corpus(tuple(conversations), corpus_hash, p)
