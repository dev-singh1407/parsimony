"""M7 — Conversation-Mined Policy Learner.

An OFFLINE program, not a pipeline stage. It replays logs and emits a versioned
PolicyBundle that the online system loads at startup.

Gap 6: every caching result in the literature is reported on a warm cache. On
day one the cache is empty, the hit rate is zero, and the user pays full price
for questions they have asked many times before in some other chat window. A
single user's own history is a large, free, perfectly on-distribution corpus
that no optimisation system consumes.

Four artefacts:

  cache_seed   recurring question/answer pairs, pre-populated
  redundancy   phrases this user habitually writes that provably never change
               the answer, identified by counterfactual replay
  digest       standing facts re-explained every session -> M4's invariant zone
  templates    recurring query shapes -> M6 tier 0

Because the bundle is content-hashed and recorded in the ledger, "warm-started"
versus "cold" is a bundle-presence flag rather than two code paths — so Figure 6
in the report comes from one code path with one input changed, not from two
implementations that could differ for uninteresting reasons.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from parsimony.core.config import LearnerConfig
from parsimony.infra.embedding import text_similarity
from parsimony.infra.nlp import RegexPiiDetector

# Phrases worth testing for redundancy. Deliberately conservative: only material
# a user might plausibly write out of habit, never content-bearing text.
CANDIDATE_PHRASES: tuple[str, ...] = (
    "please", "kindly", "thanks", "thank you", "thanks in advance",
    "could you please", "can you please", "would you please",
    "i was wondering if you could", "i would like to know", "i'd like to know",
    "if you don't mind", "hello", "hi there", "hey", "good morning",
    "as i mentioned", "as i said before", "just to be clear",
    "if that makes sense", "hope that helps", "let me know",
)

# Standing-fact patterns: first-person statements of durable context that a user
# re-explains every session.
# `\.(?=\d)` keeps a period that sits between digits: without it "Python 3.11"
# is truncated to "Python 3", and version numbers are precisely the kind of
# durable fact a user restates every session.
_STANDING_RE = re.compile(
    r"\b(?:i am|i'm|i have|i've got|my|we are|we're|we have|our)\b"
    r"(?:[^.?!]|\.(?=\d)){5,120}",
    re.IGNORECASE,
)

_TEMPLATE_SLOT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True, slots=True)
class RedundancyFinding:
    phrase: str
    occurrences: int
    unchanged: int

    @property
    def safe_rate(self) -> float:
        return self.unchanged / self.occurrences if self.occurrences else 0.0


@dataclass(slots=True)
class PolicyBundle:
    cache_seed: list[tuple[str, str]] = field(default_factory=list)
    redundancy: list[str] = field(default_factory=list)
    digest: str = ""
    templates: list[str] = field(default_factory=list)
    findings: list[RedundancyFinding] = field(default_factory=list)
    source_conversations: int = 0

    @property
    def bundle_hash(self) -> str:
        payload = json.dumps(
            {
                "cache_seed": sorted(self.cache_seed),
                "redundancy": sorted(self.redundancy),
                "digest": self.digest,
                "templates": sorted(self.templates),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()

    def save(self, root: Path) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        pii = RegexPiiDetector()

        with (root / "cache_seed.jsonl").open("w", encoding="utf-8") as fh:
            for question, answer in self.cache_seed:
                # Redaction at the write boundary (ADR-013). The bundle persists
                # and may be shared; the cache is the one component with memory.
                fh.write(json.dumps({"q": pii.redact(question), "a": pii.redact(answer)}) + "\n")

        (root / "redundancy.txt").write_text("\n".join(sorted(self.redundancy)), encoding="utf-8")
        (root / "digest.md").write_text(pii.redact(self.digest), encoding="utf-8")
        with (root / "templates.jsonl").open("w", encoding="utf-8") as fh:
            for t in sorted(self.templates):
                fh.write(json.dumps({"template": t}) + "\n")

        manifest = "\n".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
            for name in sorted(
                ("cache_seed.jsonl", "redundancy.txt", "digest.md", "templates.jsonl")
            )
        )
        (root / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
        return root

    @staticmethod
    def load(root: Path) -> "PolicyBundle":
        root = Path(root)
        bundle = PolicyBundle()
        seed = root / "cache_seed.jsonl"
        if seed.exists():
            for line in seed.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    o = json.loads(line)
                    bundle.cache_seed.append((o["q"], o["a"]))
        red = root / "redundancy.txt"
        if red.exists():
            bundle.redundancy = [ln for ln in red.read_text(encoding="utf-8").splitlines() if ln]
        dig = root / "digest.md"
        if dig.exists():
            bundle.digest = dig.read_text(encoding="utf-8")
        tpl = root / "templates.jsonl"
        if tpl.exists():
            for line in tpl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    bundle.templates.append(json.loads(line)["template"])
        return bundle


# --------------------------------------------------------------------------
# Mining
# --------------------------------------------------------------------------


def mine_recurring_questions(
    questions: list[str], answers: dict[str, str], min_count: int = 2
) -> list[tuple[str, str]]:
    """Questions asked more than once, with their answer."""
    counts = Counter(q.strip().lower() for q in questions)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for q in questions:
        norm = q.strip().lower()
        if counts[norm] >= min_count and norm not in seen and q in answers:
            seen.add(norm)
            out.append((q, answers[q]))
    return out


_MIN_STANDING_WORDS = 4


def _group_by_prefix(surfaces: list[str]) -> list[list[str]]:
    """Group statements where one is a word-prefix of another.

    A fixed-length key cannot do this: "i am using python 3.11" is five words
    and "i am using python 3.11 for this" is seven, so any fixed cut either
    splits them or truncates past the part that distinguishes 3.11 from 3.12.
    Sorting puts prefixes adjacent and shortest-first, so one pass suffices.
    """
    groups: list[list[str]] = []
    for surface in sorted(surfaces):
        placed = False
        for group in groups:
            key = group[0]
            if surface == key or surface.startswith(key + " "):
                group.append(surface)
                placed = True
                break
        if not placed:
            groups.append([surface])
    return groups


def mine_standing_context(texts: list[str], min_count: int = 2) -> str:
    """Durable first-person facts the user restates across sessions.

    People do not restate facts verbatim, so exact-match counting would see two
    singletons and emit nothing. The shortest surface in each group is kept: it
    is the one least contaminated by whatever the user was asking that day.
    """
    surfaces: list[str] = []
    for text in texts:
        for m in _STANDING_RE.finditer(text):
            surface = re.sub(r"\s+", " ", m.group(0).strip().lower())
            if len(surface.split()) >= _MIN_STANDING_WORDS:
                surfaces.append(surface)

    groups = sorted(_group_by_prefix(surfaces), key=lambda g: (-len(g), g[0]))
    lines = [f"- {group[0]}" for group in groups[:12] if len(group) >= min_count]
    return "Standing context:\n" + "\n".join(lines) if lines else ""


def mine_templates(questions: list[str], min_count: int = 2) -> list[str]:
    """Recurring query shapes with numeric slots, which M6 tier 0 can serve."""
    counts: Counter[str] = Counter()
    for q in questions:
        shape = _TEMPLATE_SLOT_RE.sub("{n}", q.strip().lower())
        if "{n}" in shape:
            counts[shape] += 1
    return [shape for shape, n in counts.most_common(20) if n >= min_count]


def counterfactual_redundancy(
    questions: list[str],
    generate,
    embedder,
    phrases: tuple[str, ...] = CANDIDATE_PHRASES,
    similarity_floor: float = LearnerConfig().redundancy_similarity_floor,
    min_occurrences: int = 2,
) -> list[RedundancyFinding]:
    """Which habitual phrases provably never change the answer.

    For each candidate phrase, re-run every question that contains it with the
    phrase removed and compare the answers. A phrase whose removal leaves the
    answer unchanged across every occurrence is safe to strip.

    This is what makes the lexicon *provable* rather than heuristic — and it is
    why M7 is offline: it costs one generation per occurrence per candidate.
    """
    findings: list[RedundancyFinding] = []
    for phrase in phrases:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b[\s,]*", re.IGNORECASE)
        occurrences = unchanged = 0
        for q in questions:
            if not pattern.search(q):
                continue
            stripped = re.sub(r"\s{2,}", " ", pattern.sub("", q)).strip()
            if not stripped or stripped == q:
                continue
            occurrences += 1
            if text_similarity(generate(stripped), generate(q), embedder) >= similarity_floor:
                unchanged += 1
        if occurrences >= min_occurrences:
            findings.append(RedundancyFinding(phrase, occurrences, unchanged))
    return findings


def learn(
    conversations: list[list[str]],
    generate,
    embedder,
    *,
    similarity_floor: float = LearnerConfig().redundancy_similarity_floor,
    min_count: int = 2,
) -> PolicyBundle:
    """Mine a policy bundle from conversation logs.

    `conversations` is a list of user-turn lists; `generate` maps a question to
    an answer (the pipeline, or a memoised replay of it).
    """
    questions = [q for conv in conversations for q in conv]
    answers = {q: generate(q) for q in dict.fromkeys(questions)}

    findings = counterfactual_redundancy(
        questions, generate, embedder, similarity_floor=similarity_floor
    )
    return PolicyBundle(
        cache_seed=mine_recurring_questions(questions, answers, min_count),
        redundancy=[f.phrase for f in findings if f.safe_rate == 1.0],
        digest=mine_standing_context(questions, min_count),
        templates=mine_templates(questions, min_count),
        findings=findings,
        source_conversations=len(conversations),
    )
