"""Cross-tokenizer generalisation: does a calibration transfer?

Report §4.6 requires re-running the winning configuration on other models and
quantisation levels "without re-tuning, which measures directly whether a
calibration transfers". That is Gap 5 and Contribution 6.

Attaching three LLMs needs Ollama, which is out of scope here. But the *model*
is not the only thing a calibration depends on — the **tokenizer** is, and it is
the part that determines almost everything this project measures:

  * every token count, and therefore every reduction figure;
  * M1 tier 3's negative-yield decisions (a re-tokenisation);
  * M1 tier 1's position-0 boundary effect (ADR-030);
  * M4's prefix survival, which is measured in tokens.

So the tokenizer dimension of the generalisation study is fully answerable
today, with two genuinely different real vocabularies (Qwen2.5 at 151,665 and
GPT-2 at 50,257). If a finding calibrated on one vocabulary does not hold on the
other, that is direct evidence for the report's central claim that published
"safe" settings do not transfer — and it is evidence obtained without a single
model download.

What this does NOT cover, stated plainly: decode speed, answer quality and
quantisation effects are properties of the model, not the tokenizer. Those still
need Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from parsimony.core.config import ParsimonyConfig
from parsimony.eval.corpus import Corpus
from parsimony.eval.runner import CellResult, summarise
from parsimony.infra.tokenization import HeuristicTokenizer, get_tokenizer

DEFAULT_TOKENIZERS: tuple[str, ...] = ("Qwen/Qwen2.5-1.5B-Instruct", "gpt2")


@dataclass(slots=True)
class TokenizerArm:
    tokenizer_id: str
    vocab_size: int
    available: bool
    results: list[CellResult] = field(default_factory=list)

    @property
    def short_name(self) -> str:
        return self.tokenizer_id.split("/")[-1]

    def reduction(self, label: str) -> float | None:
        for r in self.results:
            if r.label == label:
                return r.total_reduction_pct
        return None

    def ranking(self) -> list[str]:
        """Cells ordered best-to-worst by token reduction."""
        return [r.label for r in sorted(self.results, key=lambda r: -r.total_reduction_pct)]


def sweep_across_tokenizers(
    cells: list[ParsimonyConfig],
    corpus: Corpus,
    tokenizer_ids: tuple[str, ...] = DEFAULT_TOKENIZERS,
    *,
    progress=None,
) -> list[TokenizerArm]:
    """Run the same cells under each tokenizer, with NO re-tuning.

    Re-tuning would defeat the purpose: the question is whether a calibration
    derived on one vocabulary survives being applied to another unchanged.
    """
    from parsimony.eval.runner import run_cell

    arms: list[TokenizerArm] = []
    for tid in tokenizer_ids:
        tok = get_tokenizer(tid)
        if isinstance(tok, HeuristicTokenizer):
            arms.append(TokenizerArm(tid, 0, available=False))
            continue

        arm = TokenizerArm(tid, tok._tok.get_vocab_size(), available=True)
        for cfg in cells:
            if progress is not None:
                progress(f"{tid.split('/')[-1]}: {cfg.label}")
            arm.results.append(
                run_cell(replace(cfg, tokenizer_id=tid), corpus, tokenizer=tok)
            )
        arm.results = summarise(arm.results)
        arms.append(arm)
    return arms


# --------------------------------------------------------------------------
# Does each finding hold under both vocabularies?
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransferCheck:
    claim: str
    values: dict[str, str]
    transfers: bool
    note: str = ""


def check_boundary_effect(tokenizer_ids: tuple[str, ...] = DEFAULT_TOKENIZERS) -> list[TransferCheck]:
    """ADR-030's two position-0 mechanisms, re-measured per vocabulary."""
    checks: list[TransferCheck] = []

    cases = [
        ("leading-space form is cheaper (' happened' vs 'happened')", " happened", "happened"),
        ("capitalisation costs tokens ('explain' vs 'Explain')", "explain", "Explain"),
        ("first-word deletion can cost tokens", "What happened in the 1970s?",
         "happened in the 1970s?"),
    ]
    for claim, cheap, dear in cases:
        values: dict[str, str] = {}
        holds: list[bool] = []
        for tid in tokenizer_ids:
            tok = get_tokenizer(tid)
            if isinstance(tok, HeuristicTokenizer):
                continue
            a, b = tok.count(cheap), tok.count(dear)
            values[tid.split("/")[-1]] = f"{a} vs {b}"
            holds.append(b > a)
        checks.append(
            TransferCheck(claim, values, transfers=all(holds) and len(set(holds)) == 1)
        )
    return checks


def check_tier1_yield(
    corpus: Corpus, tokenizer_ids: tuple[str, ...] = DEFAULT_TOKENIZERS
) -> list[TransferCheck]:
    """Does tier 1's zero-yield rate transfer across vocabularies?

    If the rate differs materially, the ADR-030 guard is not a Qwen artefact but
    a vocabulary-dependent quantity — which is exactly Contribution 6's point.
    """
    from parsimony.modules.m1_compressor import normalise_lossless

    values: dict[str, str] = {}
    rates: list[float] = []
    for tid in tokenizer_ids:
        tok = get_tokenizer(tid)
        if isinstance(tok, HeuristicTokenizer):
            continue
        changed = wasted = 0
        for conv in corpus.conversations:
            for text in conv.user_turns:
                out = normalise_lossless(text)
                if out == text:
                    continue
                changed += 1
                if tok.count(out) >= tok.count(text):
                    wasted += 1
        rate = 100.0 * wasted / changed if changed else 0.0
        rates.append(rate)
        values[tid.split("/")[-1]] = f"{wasted}/{changed} ({rate:.0f}%)"

    spread = max(rates) - min(rates) if len(rates) > 1 else 0.0
    return [
        TransferCheck(
            claim="tier-1 normalisations that save nothing",
            values=values,
            transfers=spread < 10.0,
            note=f"spread {spread:.0f} percentage points",
        )
    ]
