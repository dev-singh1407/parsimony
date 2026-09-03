"""Does what M7 learns transfer to conversations it has never seen?

The project is titled "a stacked, *self-improving* optimisation layer". M7 is
built — it mines a PolicyBundle from logs, and `warm_start` loads one into a
live pipeline — but until now nothing measured it, so the word "self-improving"
rested on code rather than on a number.

The measurement that matters is not "does replaying a bundle on the
conversations it was mined from save tokens". That is guaranteed and worthless:
the cache seed contains those exact questions, so it measures memorisation. The
question is whether a bundle mined from one set of conversations helps on a
DISJOINT set. That can legitimately come back zero, and a zero is the honest
answer to report.

Split is by conversation and stratified by class, so no conversation
contributes to both halves and both halves have the same mix of question types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from parsimony.core.config import ParsimonyConfig
from parsimony.eval.corpus import Conversation, Corpus
from parsimony.eval.runner import CellResult, run_cell
from parsimony.modules.m7_learner import PolicyBundle, learn
from parsimony.infra.embedding import get_embedder
from parsimony.pipeline.orchestrator import Pipeline


def stratified_split(corpus: Corpus, *, fraction: float = 0.5) -> tuple[Corpus, Corpus]:
    """Split into (mine, held_out), disjoint by conversation.

    Deterministic and stratified: within each class the first `fraction` go to
    the mining half. Deterministic because a bundle mined from a different
    random half is a different experiment, and the result must be re-derivable
    from the corpus hash alone.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")

    mine, held = [], []
    for cls in sorted(corpus.by_class()):
        convs = corpus.by_class()[cls]
        cut = max(1, int(len(convs) * fraction)) if len(convs) > 1 else 0
        mine.extend(convs[:cut])
        held.extend(convs[cut:])

    return (
        Corpus(tuple(mine), corpus.corpus_hash, corpus.path),
        Corpus(tuple(held), corpus.corpus_hash, corpus.path),
    )


def recurrence_rate(corpus: Corpus) -> float:
    """Fraction of user turns that repeat a question asked earlier.

    The ablation corpus sits at 1.9%: it was authored for ablation diversity —
    six classes of deliberately distinct conversations — which is the right
    shape for measuring M1/M2/M3/M5 and precisely the wrong shape for measuring
    a module that learns from repetition.
    """
    turns = [q for c in corpus.conversations for q in c.user_turns]
    return ((len(turns) - len(set(turns))) / len(turns) * 100) if turns else 0.0


def deployment_trace(
    corpus: Corpus,
    *,
    recurrence: float,
    n_conversations: int = 120,
    hot_fraction: float = 0.15,
    seed: int = 0,
) -> Corpus:
    """A synthetic traffic trace with a controlled recurrence rate.

    SYNTHETIC IN ITS REPETITION STRUCTURE, and labelled as such wherever it is
    reported. Every question is a real corpus question and every answer is a
    real pipeline answer; what is imposed is how often questions recur.

    Modelled as a hot set plus a long tail, because that is the shape real
    assistant traffic takes — a support desk, a docs bot and a classroom tool
    all see a small number of questions over and over against a tail of
    one-offs. Uniform repetition would be easier to generate and would not
    resemble any deployment.
    """
    if not 0.0 <= recurrence < 1.0:
        raise ValueError(f"recurrence must be in [0, 1), got {recurrence}")

    rng = random.Random(seed)
    by_question = {q: c.cls for c in corpus.conversations for q in c.user_turns}
    # Distinct, because the base corpus already contains a few duplicate turns
    # and counting those as "fresh" would put a floor under the recurrence axis.
    pool = list(dict.fromkeys(q for c in corpus.conversations for q in c.user_turns))
    if not pool:
        raise ValueError("cannot build a trace from an empty corpus")

    n_hot = max(1, int(len(pool) * hot_fraction))
    hot, tail = pool[:n_hot], pool[n_hot:] or pool

    # Tail draws are WITHOUT replacement. Sampling it with replacement makes the
    # birthday paradox manufacture repeats on its own — a trace built with
    # recurrence=0 measured 25% actual recurrence, which silently detached the
    # x-axis from what it claimed to vary.
    fresh = tail[:]
    rng.shuffle(fresh)

    convs = []
    for i in range(n_conversations):
        want_repeat = rng.random() < recurrence
        if want_repeat or not fresh:
            q = rng.choice(hot)
        else:
            q = fresh.pop()
        convs.append(
            Conversation(
                conversation_id=f"trace_{recurrence:.2f}_{i:04d}",
                cls=by_question.get(q, "factual"),
                user_turns=(q,),
                notes=f"synthetic trace, target recurrence {recurrence:.0%}",
            )
        )
    return Corpus(tuple(convs), corpus.corpus_hash, corpus.path)


def mine_bundle(
    corpus: Corpus, cfg: ParsimonyConfig, *, provider=None, embedder=None
) -> PolicyBundle:
    """Mine a bundle by actually running the pipeline over the mining half.

    `learn` needs a question -> answer map, and the honest source for that is
    the same pipeline that would serve them, not a synthetic stub.
    """
    pipeline = Pipeline(cfg, provider=provider)
    embedder = embedder if embedder is not None else get_embedder(cfg.embedder_id)
    answers: dict[str, str] = {}

    def generate(q: str) -> str:
        if q not in answers:
            answers[q] = pipeline.run(q).response
        return answers[q]

    return learn(
        [list(c.user_turns) for c in corpus.conversations],
        generate,
        embedder,
    )


@dataclass(frozen=True, slots=True)
class LearningStudy:
    bundle: PolicyBundle
    n_mined: int
    n_held_out: int
    cold: CellResult
    warm: CellResult

    @property
    def transfer_pp(self) -> float:
        """Extra input-token reduction on unseen conversations, in points.

        Positive means the bundle helped where it had not been trained.
        """
        return _reduction(self.warm) - _reduction(self.cold)

    @property
    def extra_cache_hits(self) -> int:
        return self.warm.cache_hits - self.cold.cache_hits

    @property
    def extra_gate_fires(self) -> int:
        """The safety side of the question.

        A seeded cache can serve an answer mined from a *different* question.
        If warm-starting buys tokens by increasing wrong answers, that is not a
        win, and the gate firing more often is the first place it would show.
        """
        return self.warm.gate_fires - self.cold.gate_fires

    @property
    def verdict(self) -> str:
        if self.extra_cache_hits == 0 and abs(self.transfer_pp) < 0.05:
            return "no transfer"
        return "transfers" if self.transfer_pp > 0 else "harms"


def _reduction(r: CellResult) -> float:
    base = r.tokens_in_baseline
    return ((base - r.tokens_in_final) / base * 100) if base else 0.0


def run_learning_study(
    corpus: Corpus,
    cfg: ParsimonyConfig,
    *,
    fraction: float = 0.5,
    provider=None,
    embedder=None,
    progress=None,
) -> LearningStudy:
    """Mine from one half, measure on the other, cold against warm.

    Both arms run the identical config over the identical held-out
    conversations. The ONLY difference is whether the pipeline starts with the
    bundle loaded, which is what makes the delta attributable to M7 rather than
    to any other change.
    """
    note = progress or (lambda _: None)

    mine_half, held_out = stratified_split(corpus, fraction=fraction)
    if not len(held_out):
        raise ValueError("split left no held-out conversations to measure on")

    note(f"mining from {len(mine_half)} conversations")
    bundle = mine_bundle(mine_half, cfg, provider=provider, embedder=embedder)

    note(f"cold arm over {len(held_out)} unseen conversations")
    cold = run_cell(replace(cfg, label="cold"), held_out, provider=provider,
                    embedder=embedder)

    note(f"warm arm over the same {len(held_out)}")
    warm_cfg = replace(
        cfg,
        label="warm",
        context_digest=bundle.digest,
        bundle_hash=bundle.bundle_hash,
    )
    warm = run_cell(warm_cfg, held_out, provider=provider, embedder=embedder,
                    warm_start=bundle)

    return LearningStudy(
        bundle=bundle,
        n_mined=len(mine_half),
        n_held_out=len(held_out),
        cold=cold,
        warm=warm,
    )


DEFAULT_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


@dataclass(frozen=True, slots=True)
class RecurrencePoint:
    target: float
    actual: float
    study: LearningStudy


def recurrence_sweep(
    corpus: Corpus,
    cfg: ParsimonyConfig,
    *,
    rates: tuple[float, ...] = DEFAULT_RATES,
    n_conversations: int = 120,
    provider=None,
    embedder=None,
    progress=None,
) -> list[RecurrencePoint]:
    """How much M7 is worth, as a function of how repetitive the traffic is.

    A single number for "does learning transfer" is not answerable, because the
    answer depends entirely on the traffic. This turns the question into a
    curve, which is the same deliverable the project promises everywhere else:
    a calibration, not a headline percentage.
    """
    note = progress or (lambda _: None)
    points = []
    for r in rates:
        note(f"recurrence {r:.0%}")
        trace = deployment_trace(
            corpus, recurrence=r, n_conversations=n_conversations
        )
        points.append(
            RecurrencePoint(
                target=r,
                actual=recurrence_rate(trace),
                study=run_learning_study(
                    trace, cfg, provider=provider, embedder=embedder
                ),
            )
        )
    return points
