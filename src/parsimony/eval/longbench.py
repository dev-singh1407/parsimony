"""The same compressor, on a benchmark we did not write.

`corpus/longctx_*.jsonl` is ours: we chose the documents, the distractors, the
question kinds and the evidence spans. That is a legitimate instrument -- it can
ask things no public set asks, like "does the tier survive an off-topic
question" -- but it cannot answer the one question an examiner asks first. *Did
you only do well because you set the exam?*

So this runs the shipped configuration, unchanged, on **LongBench** [Bai et al.,
ACL 2024], the set the compression literature actually reports: same tasks, same
metric, same answers, none of them ours. It is the weakest position the project
can be measured from, which is why it is worth having.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Comparable: the arms here are compared against each other, on the same items,
through the same model, with the same budget. That comparison is sound.

NOT comparable: the absolute F1 against published LongBench tables. Those are
produced by 7B-70B models; this is qwen2.5-1.5b-instruct on a laptop CPU, which
scores far lower on every task before compression is involved at all. A number
from this file should never be placed in a column beside a number from a paper.
The full-context arm is the only baseline that means anything here, and it is
run on every item for exactly that reason.

THE METRIC IS THEIRS
--------------------
`qa_f1` below is LongBench's own scorer for the QA tasks -- SQuAD-style token F1
after their normalisation, maximised over the reference answers. Rewriting the
metric to suit the system under test is the easiest way to win a benchmark and
the fastest way to make it worthless, so this is a transcription, and the tests
pin it against worked examples.

DATA
----
Not vendored: LongBench is ~110 MB and its tasks carry the licences of the
datasets they are built from. `--data` points at an extracted copy, and a run
records the file's SHA-256 so a result can be tied to the bytes it came from.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import re
import string
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from parsimony.core.config import ParsimonyConfig
from parsimony.core.types import Document

#: The QA tasks whose answers are short enough for token F1 to mean something.
#: Summarisation (gov_report, multi_news) is scored with ROUGE in LongBench and
#: is deliberately out of scope: a 1.5B model's summaries are not what this
#: project is measuring, and half-implementing ROUGE would be worse than not.
TASKS = ("2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper")


# ------------------------------------------------------------- the metric --


def _normalise(text: str) -> str:
    """LongBench's normalisation: lowercase, strip articles, punctuation, space."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction: str, answers: list[str]) -> float:
    """Token F1 against the best reference. LongBench's `qa_f1_score`, transcribed."""
    best = 0.0
    for answer in answers:
        pred_tokens = _normalise(prediction).split()
        gold_tokens = _normalise(answer).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


# --------------------------------------------------------------- the data --


@dataclass(frozen=True, slots=True)
class BenchItem:
    item_id: str
    task: str
    question: str
    answers: tuple[str, ...]
    documents: tuple[Document, ...]
    context_words: int


def _as_documents(context: str, item_id: str) -> tuple[Document, ...]:
    """LongBench ships one context string; the passages inside it are the units.

    Multi-document tasks concatenate passages with blank lines, and a passage
    usually opens with its title. Splitting there gives the compressor the same
    document boundaries a real caller would have, without inventing any: where
    there are no blank lines the whole context stays as one document, which is
    the honest representation of a single-document task.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", context) if b.strip()]
    if len(blocks) < 2:
        return (Document(f"{item_id}_d0", context.strip(), ""),)
    out = []
    for i, block in enumerate(blocks):
        first, _, rest = block.partition("\n")
        title = first.strip() if rest and len(first) < 120 else ""
        out.append(Document(f"{item_id}_d{i}", block, title))
    return tuple(out)


def load(task: str, data_dir: Path, limit: int | None = None) -> tuple[BenchItem, ...]:
    """The first `limit` items of a task, in file order.

    File order, never sampled: picking items by length would be choosing the
    exam again, and this file exists to stop the project doing that.
    """
    path = Path(data_dir) / f"{task}.jsonl"
    items = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if limit is not None and len(items) >= limit:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(BenchItem(
            item_id=row.get("_id") or f"{task}_{i}",
            task=task,
            question=row["input"].strip(),
            answers=tuple(row["answers"]),
            documents=_as_documents(row["context"], row.get("_id") or f"{task}_{i}"),
            context_words=len(row["context"].split()),
        ))
    return tuple(items)


def data_digest(task: str, data_dir: Path) -> str:
    """SHA-256 of the task file, so a result names the bytes it came from."""
    return hashlib.sha256((Path(data_dir) / f"{task}.jsonl").read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------- the arms --

#: LongBench's own prompt for the multi-document QA tasks, kept verbatim. The
#: compressor is allowed to change the CONTEXT and nothing else; rewriting the
#: instruction would make this a different experiment.
PROMPT = ("Answer the question based on the given passages. Only give me the answer and do not "
          "output any other words.\n\nThe following are given passages.\n{context}\n\n"
          "Answer the question based on the given passages. Only give me the answer and do not "
          "output any other words.\n\nQuestion: {question}\nAnswer:")

ARMS = ("full", "parsimony", "truncate")


def _render(documents: tuple[Document, ...], question: str) -> str:
    return PROMPT.format(context="\n\n".join(d.content for d in documents), question=question)


def run(task: str, data_dir: Path, provider, out_path: Path, *, limit: int = 20,
        cfg: ParsimonyConfig | None = None, arms: tuple[str, ...] = ARMS,
        num_predict: int = 48, progress: Callable[[str], None] | None = None) -> list[dict]:
    """Every arm of every item, one JSON row per model call. Resumable.

    Refuses rather than records when a prompt does not fit the server's window:
    a truncated full-context arm would hand compression a win it did not earn
    (ADR-045).
    """
    from parsimony.core.types import GenParams
    from parsimony.eval.longctx import Methods
    from parsimony.infra.providers import ProviderError, window_overflow

    methods = Methods(cfg)
    items = load(task, data_dir, limit)
    digest = provider.model_digest
    params = GenParams(num_predict=num_predict, temperature=0.0, seed=0)

    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("model_digest") == digest and row.get("task") == task:
                    rows.append(row)
                    done.add((row["item_id"], row["arm"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(items) * len(arms)
    with out_path.open("a", encoding="utf-8") as fh:
        for n_item, item in enumerate(items):
            position = answer_position(item)
            ours = _compress(methods, item, cfg)
            budget = sum(methods.count(d.content) for d in ours)
            for n_arm, arm in enumerate(arms):
                if (item.item_id, arm) in done:
                    continue
                if arm == "full":
                    documents = item.documents
                elif arm == "parsimony":
                    documents = ours
                elif arm == "truncate":
                    documents = _truncate(methods, item, budget)
                else:
                    raise ValueError(f"unknown arm {arm!r}")

                # A fresh nonce per prompt. The arms of one item share almost
                # all of their text -- truncation IS a prefix of the full
                # context -- and Ollama reuses its key-value cache for any
                # identical prefix, so without this the second arm's prefill is
                # measured as nearly free. It showed up immediately on the
                # first run of this harness: an arm that had not compressed at
                # all reported 0.1 s for 7,152 tokens, because the previous arm
                # had just read the same bytes (ADR-034, reintroduced here and
                # caught by the number being impossible).
                prompt = f"[{secrets.token_hex(3)}]\n" + _render(documents, item.question)
                sent = methods.count(prompt)
                if progress:
                    progress(f"{n_item * len(arms) + n_arm + 1}/{total}  {item.item_id[:12]}  "
                             f"{arm}  {sent:,} tok")
                started = time.perf_counter()
                text, stats = provider.complete(prompt, params)
                wall_ms = (time.perf_counter() - started) * 1000
                if window_overflow(sent, stats):
                    raise ProviderError(
                        f"{item.item_id}/{arm}: {sent:,} prompt tokens exceed the "
                        f"{stats.get('num_ctx'):,}-token window, so the runtime dropped the front "
                        f"of the prompt. Raise num_ctx rather than recording this row.")

                row = {
                    "task": task, "item_id": item.item_id, "arm": arm,
                    # A property of the item, stored per row so the analysis in
                    # `by_position` survives without the 110 MB of source data.
                    "answer_position": (round(position, 4)
                                        if position is not None else None),
                    "model": provider.model_name, "model_digest": digest,
                    "f1": round(qa_f1(text, list(item.answers)), 4),
                    "response": text.strip()[:400],
                    "answers": list(item.answers),
                    "context_words": item.context_words,
                    "context_tokens": sum(methods.count(d.content) for d in documents),
                    "full_context_tokens": sum(methods.count(d.content) for d in item.documents),
                    "prompt_tokens": stats.get("prompt_eval_count"),
                    "prefill_ms": stats.get("prompt_eval_duration", 0) / 1e6,
                    "decode_ms": stats.get("eval_duration", 0) / 1e6,
                    "wall_ms": round(wall_ms, 1),
                    "num_ctx": stats.get("num_ctx"),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
    return rows


def _compress(methods, item: BenchItem, cfg: ParsimonyConfig | None):
    """The shipped context tier, through the gate, exactly as `ask` would run it."""
    from dataclasses import replace as _replace

    from parsimony.core.proposals import ContextPatch

    config = cfg or methods.cfg
    pipeline = methods._pipeline_for(config)
    ctx = pipeline.build_context(item.question, conversation_id=item.item_id,
                                 documents=item.documents)
    proposal = methods.stage.propose(ctx, config)
    if not isinstance(proposal, ContextPatch):
        return item.documents
    candidate = _replace(ctx, **dict(proposal.fields))
    verdict = pipeline.gate.check(ctx, candidate, proposal.kind, "M1")
    return candidate.documents if verdict.passed else item.documents


def _truncate(methods, item: BenchItem, budget: int) -> tuple[Document, ...]:
    """Keep the context in order until the budget is spent: what a framework does."""
    from parsimony.infra.nlp import split_sentences

    kept, used = [], 0
    for doc in item.documents:
        parts = []
        for sentence in split_sentences(doc.content):
            n = methods.count(sentence)
            if used + n > budget and (parts or kept):
                break
            parts.append(sentence)
            used += n
        if parts:
            kept.append(Document(doc.doc_id, " ".join(parts), doc.title))
        if used >= budget:
            break
    return tuple(kept) or item.documents


# ------------------------------------------------------------- reporting --


def answer_position(item: BenchItem) -> float | None:
    """How far through the context the answer string first appears, 0.0-1.0.

    A property of the DATA, computed without reference to any result, so it can
    split the items without being a fishing expedition. `None` when no reference
    answer occurs literally in the context -- which happens, since the answers
    are not guaranteed to be extractive.
    """
    context = _normalise("\n\n".join(d.content for d in item.documents))
    total = len(context.split())
    best = None
    for answer in item.answers:
        needle = _normalise(answer)
        if not needle:
            continue
        idx = context.find(needle)
        if idx >= 0:
            fraction = len(context[:idx].split()) / max(1, total)
            best = fraction if best is None else min(best, fraction)
    return best


def evidence_kept(item: BenchItem, documents: tuple[Document, ...]) -> bool | None:
    """Does the answer still appear in what this arm sent?

    `None` when the answer never appeared in the full context either -- those
    items say nothing about a compressor, because there was nothing to keep.
    Matching is on the normalised text, the same way `qa_f1` compares, so a
    difference in punctuation or casing is not mistaken for a loss.
    """
    if answer_position(item) is None:
        return None
    sent = _normalise("\n\n".join(d.content for d in documents))
    return any(_normalise(a) and _normalise(a) in sent for a in item.answers)


def recall_report(task: str, data_dir: Path, *, limit: int = 40,
                  cfg: ParsimonyConfig | None = None,
                  budget_from: str = "parsimony") -> list[dict]:
    """Evidence recall per arm, recomputed offline. No model calls.

    Deterministic, so it is cheap to re-run and cannot disagree with itself
    between runs the way an accuracy number can.
    """
    from parsimony.eval.longctx import Methods

    methods = Methods(cfg)
    items = load(task, data_dir, limit)
    kept: dict[str, list[bool]] = {"full": [], "parsimony": [], "truncate": []}
    skipped = 0
    for item in items:
        ours = _compress(methods, item, cfg)
        budget = sum(methods.count(d.content) for d in ours)
        arms = {"full": item.documents, "parsimony": ours,
                "truncate": _truncate(methods, item, budget)}
        verdicts = {name: evidence_kept(item, docs) for name, docs in arms.items()}
        if verdicts["full"] is None:
            skipped += 1
            continue
        for name, verdict in verdicts.items():
            kept[name].append(bool(verdict))
    out = []
    for name, flags in kept.items():
        if flags:
            out.append({"arm": name, "n": len(flags), "kept": sum(flags),
                        "recall_pct": round(100 * sum(flags) / len(flags), 1)})
    if skipped:
        out.append({"arm": "(excluded: answer not literally in the full context)",
                    "n": skipped, "kept": 0, "recall_pct": 0.0})
    return out


def by_position(rows: list[dict], items: dict[str, BenchItem] | None = None, *,
                cutoff: float = 0.20) -> dict[str, list[dict]]:
    """The same summary, split by whether keeping the front keeps the answer.

    Truncation's whole strategy is "keep the front". On a benchmark whose
    documents arrive in their natural order, that is a coin toss it sometimes
    wins -- which is why it scores level with a compressor overall, and why the
    overall number alone hides what is happening. Splitting on where the answer
    actually sits separates "truncation worked" from "truncation got lucky".

    `cutoff` defaults to the budget the compressed arms are held to, because
    that is the threshold that makes the split mean something: below it,
    truncation keeps the answer by construction.

    Positions come from the rows themselves where a run recorded them, so this
    runs on a results file alone; `items` is only needed for rows written before
    that field existed.
    """
    positions: dict[str, float | None] = {}
    for row in rows:
        if "answer_position" in row:
            positions[row["item_id"]] = row["answer_position"]
    for item_id, item in (items or {}).items():
        positions.setdefault(item_id, answer_position(item))

    early, late = [], []
    for item_id, position in positions.items():
        (early if position is not None and position < cutoff else late).append(item_id)
    return {
        "early": summarise([r for r in rows if r["item_id"] in early]),
        "late": summarise([r for r in rows if r["item_id"] in late]),
        "counts": [{"group": "early", "n": len(early)}, {"group": "late", "n": len(late)}],
    }


def summarise(rows: list[dict]) -> list[dict]:
    """One row per arm: F1, what it sent, and what reading it cost."""
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    full = {r["item_id"]: r for r in by_arm.get("full", [])}
    out = []
    for arm in ARMS:
        got = by_arm.get(arm)
        if not got:
            continue
        n = len(got)
        f1 = sum(r["f1"] for r in got) / n
        ctx = sum(r["context_tokens"] for r in got) / n
        prefill = sum(r["prefill_ms"] for r in got) / n
        denom = sum(full[r["item_id"]]["context_tokens"] for r in got if r["item_id"] in full)
        out.append({
            "arm": arm, "n": n,
            "f1": round(100 * f1, 1),
            "context_kept_pct": round(100 * sum(r["context_tokens"] for r in got) / denom, 1)
            if denom else 100.0,
            "context_tokens": round(ctx),
            "prefill_s": round(prefill / 1000, 2),
        })
    return out
