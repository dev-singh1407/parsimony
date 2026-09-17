"""Does history management keep the fact the last question needs?

M3 drops whole turns and M1's context tier shortens the ones that survive.
Both were measured in tokens and never in answers, because the conversation
corpus has no ground truth: its assistant turns come from a mock provider, so
"the answer changed" and "the answer got worse" are indistinguishable there.

`corpus/followups.jsonl` supplies the missing case. Each conversation states a
fact in its first turn -- a server's memory, an allergy, a policy number --
spends four turns on unrelated things, and ends with a question only that first
turn can answer. Written-out assistant replies make the history realistic in
length, which is what the history manager is deciding about.

The arms differ ONLY in how history is handled:

  everything          every turn verbatim: no selection, no compression
  keep the last 4     recency, the industry default
  relevance (MMR)     M3 as shipped: relevance minus redundancy
  MMR + sentences     M3, then M1's context tier inside the surviving turns
  no history          the control -- the fact is gone, so a right answer here
                      would mean the question was answerable without it

Splits: dev (6) for tuning, test (14) reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from parsimony.core.config import ParsimonyConfig, full_stack
from parsimony.core.types import Turn
from parsimony.eval.corpus import GoldItem

CORPUS = Path(__file__).resolve().parents[3] / "corpus"


@dataclass(frozen=True, slots=True)
class FollowUp:
    conversation_id: str
    split: str
    turns: tuple[Turn, ...]
    question: str
    gold: GoldItem
    evidence_turn: int
    evidence_text: str = ""

    @property
    def evidence(self) -> str:
        """The sentence the answer is in, where the corpus names one.

        Not the whole turn: sentence-level compression is supposed to remove
        the rest of it, and scoring against the turn would count that as
        losing the fact.
        """
        return self.evidence_text or self.turns[self.evidence_turn].content


def load_followups(path: Path | str | None = None) -> tuple[FollowUp, ...]:
    p = Path(path) if path else CORPUS / "followups.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        turns = tuple(Turn(f"t{i}", t["role"], t["content"])
                      for i, t in enumerate(o["turns"]))
        gold = GoldItem(o["conversation_id"], o["question"], o["gold_answer"],
                        o.get("match", "contains"), 0.0,
                        tuple(o.get("acceptable_variants", ())))
        out.append(FollowUp(o["conversation_id"], o["split"], turns, o["question"], gold,
                            int(o.get("evidence_turn", 0)), o.get("evidence_text", "")))
    return tuple(out)


# ------------------------------------------------------------------- arms --


def arms(base: ParsimonyConfig | None = None,
         gates: tuple[int, int] | None = None) -> dict[str, ParsimonyConfig | None]:
    """Configurations that differ only in how the conversation is handled.

    `gates` is (context_min_tokens, context_turn_min_tokens) for the sentence
    compression arm. The shipped values were set for attached documents, where
    a thousand tokens is normal; a conversation turn is a tenth of that, so on
    a chat they never fire. Passing the candidate values here is what makes the
    arm a measurement rather than a copy of the arm above it.
    """
    base = base or full_stack()
    c, h = base.compression, base.history
    no_context = replace(base, compression=replace(c, context_enabled=False))
    tight = base
    if gates is not None:
        tight = replace(base, compression=replace(c, context_enabled=True,
                                                  context_min_tokens=gates[0],
                                                  context_turn_min_tokens=gates[1]))
    return {
        "everything": replace(no_context, enabled_modules=base.enabled_modules - {"M3"}),
        "keep_last_4": replace(no_context, history=replace(h, strategy="recency", max_turns=4)),
        "relevance": no_context,                       # M3 as shipped, no sentence compression
        "relevance_sentences": tight,                  # and with it
        "no_history": None,                            # control: history withheld
    }


ARM_LABELS = {
    "everything": "every turn, verbatim",
    "keep_last_4": "keep the last 4 turns",
    "relevance": "relevance (MMR)",
    "relevance_sentences": "relevance + sentence compression",
    "no_history": "no history at all (control)",
}


def evidence_survives(item: FollowUp, ctx) -> bool:
    """Is the fact the question needs still in what reached the model?"""
    sent = "\n".join(t.content for t in ctx.history)
    return item.evidence in sent


@dataclass(frozen=True, slots=True)
class FollowUpRun:
    conversation_id: str
    arm: str
    correct: bool
    evidence_kept: bool
    prompt_tokens: int
    prefill_ms: float
    response: str


def run_real(items, provider, *, base: ParsimonyConfig | None = None,
             gates: tuple[int, int] | None = (100, 40),
             out_path: Path | None = None,
             progress: Callable[[str], None] | None = None) -> list[dict]:
    """Ask the real model the last question of each conversation, under every arm.

    Resumable and append-only, like the long-context study: an interrupted run
    continues rather than restarting, and every call is kept so a wrong answer
    can be read rather than counted.
    """
    import secrets
    import time

    from parsimony.core.types import GenParams
    from parsimony.eval.metrics import grade
    from parsimony.pipeline.orchestrator import Pipeline

    configured = arms(base, gates)
    digest = provider.model_digest
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if out_path is not None and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("model_digest") == digest:
                    rows.append(row)
                    done.add((row["conversation_id"], row["arm"]))

    params = GenParams(num_predict=64, temperature=0.0, seed=0)
    total = len(items) * len(configured)
    fh = out_path.open("a", encoding="utf-8") if out_path is not None else None
    try:
        for n, item in enumerate(items):
            for k, (arm, cfg) in enumerate(configured.items()):
                if (item.conversation_id, arm) in done:
                    continue
                if progress:
                    progress(f"{n * len(configured) + k + 1}/{total}  "
                             f"{item.conversation_id}  {arm}")
                history = () if cfg is None else item.turns
                config = cfg if cfg is not None else replace(
                    full_stack() if base is None else base,
                    enabled_modules=(base or full_stack()).enabled_modules - {"M3"})
                # A nonce in the always-first instruction line, so no arm is
                # measured against the key-value cache of the arm before it.
                # Two arms that select the same turns produce byte-identical
                # prompts, and the second was being timed at 0.1 s against the
                # first's 1.7 s -- a saving that belongs to the runtime, not to
                # the pipeline (ADR-034, ADR-040).
                config = replace(config,
                                 system_prompt=f"{config.system_prompt} [{secrets.token_hex(3)}]")
                pipe = Pipeline(config, provider=provider)
                ctx = pipe.build_context(item.question, history)
                prompt = ""
                start = time.perf_counter()
                outcome = pipe.run(item.question, history,
                                   conversation_id=f"{item.conversation_id}:{arm}")
                wall = (time.perf_counter() - start) * 1000
                stats = dict(getattr(provider, "last_stats", {}) or {})
                row = {
                    "conversation_id": item.conversation_id, "split": item.split, "arm": arm,
                    "model": provider.model_name, "model_digest": digest,
                    "correct": grade(outcome.response, item.gold),
                    "evidence_kept": evidence_survives(item, outcome.ctx),
                    "prompt_tokens": stats.get("prompt_eval_count") or outcome.row.tokens_in_final,
                    "tokens_in_final": outcome.row.tokens_in_final,
                    "prefill_ms": stats.get("prompt_eval_duration", 0) / 1e6,
                    "wall_ms": round(wall, 1),
                    "response": outcome.response.strip(),
                    "question": item.question, "gold": item.gold.gold_answer,
                }
                del prompt, ctx
                rows.append(row)
                if fh is not None:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return rows


def summarise(rows: list[dict], split: str | None = None) -> list[dict]:
    from parsimony.eval.stats import mcnemar_exact, wilson_interval

    chosen = [r for r in rows if split is None or r["split"] == split]
    by_arm: dict[str, dict[str, dict]] = {}
    for r in chosen:
        by_arm.setdefault(r["arm"], {})[r["conversation_id"]] = r
    reference = by_arm.get("everything", {})
    out = []
    for arm in ARM_LABELS:
        got = by_arm.get(arm)
        if not got:
            continue
        ids = sorted(got)
        correct = sum(got[i]["correct"] for i in ids)
        paired = [i for i in ids if i in reference]
        lost = sum(reference[i]["correct"] and not got[i]["correct"] for i in paired)
        gained = sum(got[i]["correct"] and not reference[i]["correct"] for i in paired)
        interval = wilson_interval(correct, len(ids))
        out.append({
            "method": ARM_LABELS[arm], "arm": arm, "conversations": len(ids),
            "correct": f"{correct}/{len(ids)}",
            "accuracy %": f"{interval.point:.1f}",
            "95% CI": f"{interval.low:.1f}-{interval.high:.1f}",
            "fact kept": f"{sum(got[i]['evidence_kept'] for i in ids)}/{len(ids)}",
            "prompt tokens": f"{sum(got[i]['prompt_tokens'] for i in ids) / len(ids):.0f}",
            "prefill ms": f"{sum(got[i]['prefill_ms'] for i in ids) / len(ids):.0f}",
            "lost vs everything": "" if arm == "everything" else lost,
            "gained": "" if arm == "everything" else gained,
            "McNemar p": "" if arm == "everything" else f"{mcnemar_exact(lost, gained):.3f}",
        })
    return out
