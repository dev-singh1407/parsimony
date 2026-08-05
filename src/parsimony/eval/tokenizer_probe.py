"""Empirical probe: when does shortening text fail to reduce tokens?

M1 tier 3's negative-yield check exists because byte-pair merges are context
dependent, so a shorter string is not necessarily a cheaper one. The report
states this as an assumption. This module measures it, because an unmeasured
assumption underpinning a headline contribution is a liability.

Three regimes are probed separately, and they behave very differently:

  phrase   -- lexicon substitutions ("in order to" -> "to")
  word     -- delete one whitespace-delimited word
  subtoken -- edits that cut inside a token (drop a character, join two words)

Run: parsimony tokenprobe
"""

from __future__ import annotations

from dataclasses import dataclass

from parsimony.modules.m1_tier3 import DEFAULT_LEXICON, find_candidates, full_delta


@dataclass(frozen=True, slots=True)
class ProbeResult:
    regime: str
    tested: int
    reduced: int
    neutral: int
    increased: int
    examples: tuple[tuple[int, str, str], ...] = ()

    @property
    def wasted(self) -> int:
        """Edits that shortened the text without saving a single token."""
        return self.neutral + self.increased

    @property
    def wasted_pct(self) -> float:
        return 100.0 * self.wasted / self.tested if self.tested else 0.0


# The corpus is written in the terse register real users type, so it contains
# almost none of the lexicon's verbose phrases. These carry the phrase regime so
# that row measures something rather than reporting an empty set — and the fact
# that the natural corpus does not trigger tier 3 is itself worth reporting.
PHRASE_TEXTS: tuple[str, ...] = (
    "You need to do this in order to succeed at the task.",
    "It is important to note that the budget is 50,000 dollars.",
    "Due to the fact that a large number of users complained, we acted.",
    "Prior to the meeting, please review the majority of the documents.",
    "In the event that it fails, we are able to retry on a regular basis.",
    "With regard to your question: in order to proceed, act prior to Friday.",
    "A large number of records exist. The majority of them are stale.",
    "At this point in time we would like to proceed with the plan.",
    "In spite of the fact that it is slow, it has the ability to scale.",
    "Subsequent to the review, a sufficient amount of work remains.",
)

# Deliberately adversarial: edits that do NOT respect token boundaries.
SUBTOKEN_CASES: tuple[tuple[str, str, str], ...] = (
    ("drop a character", "running quickly", "runing quickly"),
    ("drop a character", "unbelievable result", "unbelievble result"),
    ("join two words", "in order to run", "inorder to run"),
    ("join two words", "New York City", "NewYork City"),
    ("remove a hyphen", "state-of-the-art design", "stateoftheart design"),
    ("join two words", "a very good idea", "a verygood idea"),
    ("singularise", "the categories exist", "the category exist"),
    ("drop a character", "environment variable", "enviroment variable"),
)


def probe_phrase(texts: list[str], tokenizer) -> ProbeResult:
    reduced = neutral = increased = 0
    examples: list[tuple[int, str, str]] = []
    for text in texts:
        for edit in find_candidates(text, DEFAULT_LEXICON):
            delta = full_delta(text, edit, tokenizer)
            if delta < 0:
                reduced += 1
            elif delta == 0:
                neutral += 1
                examples.append((delta, edit.matched, edit.replacement))
            else:
                increased += 1
                examples.append((delta, edit.matched, edit.replacement))
    return ProbeResult("phrase", reduced + neutral + increased, reduced, neutral,
                       increased, tuple(examples[:8]))


def probe_word(texts: list[str], tokenizer) -> ProbeResult:
    reduced = neutral = increased = 0
    examples: list[tuple[int, str, str]] = []
    for text in texts:
        base = tokenizer.count(text)
        words = text.split(" ")
        if len(words) < 2:
            continue
        for i, word in enumerate(words):
            if not word.strip():
                continue
            delta = tokenizer.count(" ".join(words[:i] + words[i + 1 :])) - base
            if delta < 0:
                reduced += 1
            elif delta == 0:
                neutral += 1
                examples.append((delta, word, ""))
            else:
                increased += 1
                examples.append((delta, word, ""))
    return ProbeResult("word", reduced + neutral + increased, reduced, neutral,
                       increased, tuple(examples[:8]))


def probe_subtoken(tokenizer) -> ProbeResult:
    reduced = neutral = increased = 0
    examples: list[tuple[int, str, str]] = []
    for _label, before, after in SUBTOKEN_CASES:
        delta = tokenizer.count(after) - tokenizer.count(before)
        if delta < 0:
            reduced += 1
        elif delta == 0:
            neutral += 1
            examples.append((delta, before, after))
        else:
            increased += 1
            examples.append((delta, before, after))
    return ProbeResult("subtoken", len(SUBTOKEN_CASES), reduced, neutral,
                       increased, tuple(examples[:8]))


def run_probe(texts: list[str], tokenizer) -> list[ProbeResult]:
    return [
        probe_phrase(list(PHRASE_TEXTS) + texts, tokenizer),
        probe_word(texts, tokenizer),
        probe_subtoken(tokenizer),
    ]
