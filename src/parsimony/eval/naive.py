"""The obvious approaches, implemented faithfully so they can be shown failing.

Every layer in this system invites the same objection: the idea is simple.
Remove unimportant words. Reuse answers to similar questions. Keep the recent
messages. Send the question to the AI. Anyone could build that.

Anyone could build the OBVIOUS version, and this module is the obvious version,
written straight so it can run beside the real one on the same input. None of
it is a straw man: each function is the design a practitioner reaches for
first, and where the literature documents a default, that default is used.

What each one gets wrong was measured, not asserted:

  stopword_compress   the standard NLTK English stopword list contains "not",
                      so "Is it not safe to mix bleach and vinegar?" becomes
                      "safe mix bleach vinegar" -- shorter, and reversed.
  abbreviate          the model's tokenizer already spends one token on common
                      words, so "information" -> "info" saves nothing and
                      "without" -> "w/o" costs an extra token.
  threshold_cache_hit similarity alone cannot tell a question from its
                      negation: the bleach pair scores 91% and is reused,
                      while a politely worded repeat scores 63% and is missed.
  keep_last           a fact stated early in a conversation is the first thing
                      dropped, however relevant it is to the question now.
"""

from __future__ import annotations

import re

import numpy as np

#: The NLTK English stopword list, the default a practitioner reaches for when
#: told to "remove the unimportant words". Reproduced rather than imported so
#: the project keeps no dependency on NLTK's corpus download.
NLTK_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
""".split())

#: Abbreviations a person would try first to "make the prompt shorter" --
#: ordinary ones found in any style guide, not chosen to fail. Each is scored
#: on the model's own tokenizer wherever it is shown.
ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("information", "info"),
    ("approximately", "approx"),
    ("please", "pls"),
    ("without", "w/o"),
    ("for example", "e.g."),
    ("government", "govt"),
)

#: Inside the 0.85-0.92 range commonly given as a safe semantic-cache threshold.
DEFAULT_CACHE_THRESHOLD = 0.85

#: One answer length for every question.
FIXED_MAX_TOKENS = 128

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def stopword_compress(text: str) -> str:
    """Drop every word on the standard stopword list."""
    return " ".join(w for w in _WORD_RE.findall(text) if w.lower() not in NLTK_STOPWORDS)


def abbreviate(text: str) -> str:
    """Replace long words with the short forms a person would use."""
    out = text
    for long, short in ABBREVIATIONS:
        out = re.sub(rf"\b{re.escape(long)}\b", short, out, flags=re.IGNORECASE)
    return out


def cosine(embedder, a: str, b: str) -> float:
    va, vb = embedder.embed([a, b])
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(va @ vb) / denom


def threshold_cache_hit(stored: str, query: str, embedder,
                        threshold: float = DEFAULT_CACHE_THRESHOLD) -> tuple[bool, float]:
    """The standard semantic cache: embed both, reuse if similar enough.

    Compares the raw questions, as such caches do, with no check of what the
    two actually say.
    """
    score = cosine(embedder, stored, query)
    return score >= threshold, score


def keep_last(history: tuple, n: int) -> tuple:
    """Keep the most recent `n` messages and drop everything older."""
    return tuple(history[-n:]) if n > 0 else ()
