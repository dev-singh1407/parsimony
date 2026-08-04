# Parsimony — Corpus Specification and Authoring Guide

**This is the one deliverable with zero technical dependencies. All three of you can start today, in
parallel, before a single line of code exists.** It is also the one artefact that cannot be fixed later:
per ADR-015 the corpus must be frozen and hashed *before* the first sweep, and the gold answers must be
written *before* anyone inspects a system output.

---

## 1. What is being built

| Artefact | Size | Purpose |
|---|---|---|
| `conversations.jsonl` | 150 conversations, 6 classes × 25 | The evaluation workload |
| `adversarial_pairs.jsonl` | 50 pairs | Measures false cache hits (Gap 3 / Contribution 2) |
| `gold.jsonl` | 40 Q&A | The **only** non-proxy quality measure in the project |
| `MANIFEST.sha256` | — | The freeze; its hash goes in every ledger row |

---

## 2. `conversations.jsonl`

One JSON object per line.

```json
{
  "conversation_id": "fact_007",
  "class": "factual",
  "author": "dev",
  "turns": [
    {"role": "user", "content": "What is the boiling point of water at sea level?"},
    {"role": "assistant", "content": null},
    {"role": "user", "content": "And at 3000 metres?"},
    {"role": "assistant", "content": null}
  ],
  "notes": "tests unit retention and a follow-up that depends on turn 1"
}
```

`assistant.content` is `null` — responses are *generated*, never authored. Authoring assistant turns would
make the corpus a fixture rather than a workload.

### The six classes, 25 each

| Class | Turns | What it must exercise | Watch for |
|---|---|---|---|
| **single-turn factual** | 1 | Short queries where tier 2 compression has nothing to remove | Establishes the honest floor — do not pad these to make compression look good |
| **multi-turn follow-up** | 4–8 | Anaphora ("it", "that one", "the second"), so M3 selection and the cache context chain are genuinely tested | At least 10 must have a follow-up that is *unanswerable* without a specific earlier turn |
| **arithmetic & short reasoning** | 1–3 | Router tier 0 (exact arithmetic, units, dates) and the reasoning escalation | Split ~15 tier-0-answerable / ~10 requiring the model — do **not** make them all tier-0-friendly |
| **code explanation** | 1–3 | Long prompts with code blocks; markdown scaffolding for M1 tier 1; fenced-block early stopping for M5 | Include ≥5 with numeric literals in code that M8 must protect |
| **summarisation** | 1–2 | Long inputs where M1 tier 2 has real redundancy to remove | Source text must be licence-clean — write it or use public-domain text |
| **paraphrased repeats** | 2–4 | Semantic cache true-hit rate: the same question asked differently | Vary phrasing *only*, never meaning — meaning changes belong in the adversarial set |

### Authoring rules

1. **Write queries you would actually type.** The corpus is the workload; artificially verbose prompts
   inflate compression numbers and a reviewer will spot it.
2. **Number and entity density must be realistic.** M8's fire rate is a reported result. A corpus with no
   numerals makes the fidelity gate look free; one stuffed with numerals makes it look prohibitive.
3. **Include ~20 % prompts with natural redundancy** — restating, "as I mentioned", conversational padding.
   That is what real chat looks like and it is what M1 tier 2 exists for.
4. **No PII, real or synthetic-realistic.** The corpus ships with the code.
5. **English only** for v1. Note it as a limitation.
6. **Record `notes`** — what each conversation is meant to exercise. In week 12 you will not remember, and
   these notes become the corpus description section of the report.

### Split across the team

| Person | Classes | Count |
|---|---|---|
| Arrsh | factual, follow-up | 50 |
| Alok | arithmetic, code | 50 |
| Dev | summarisation, paraphrased repeats | 50 |

**Cross-review before freeze.** Each person reviews another's 50 against the rules above. Two hours, and it
is the difference between a corpus you can defend and one you cannot.

---

## 3. `adversarial_pairs.jsonl` — the sharpest instrument in the project

50 pairs of questions that differ by **exactly one operative token** and have **materially different correct
answers**. This subset alone carries Contribution 2.

```json
{
  "pair_id": "adv_012",
  "a": "Is it safe to mix bleach and vinegar?",
  "b": "Is it not safe to mix bleach and vinegar?",
  "operative": "negation",
  "answers_differ": true,
  "notes": "negation flip; a semantic cache at cosine 0.96 will match these"
}
```

**Required distribution** — this is what makes the result interpretable rather than a single mixed number:

| `operative` | Count | Example |
|---|---|---|
| `negation` | 15 | "is X safe" / "is X **not** safe" |
| `number` | 15 | "convert 100 **km**" / "convert 100 **m**" |
| `entity` | 10 | "capital of **Australia**" / "capital of **Austria**" |
| `modifier` | 10 | "**minimum** temperature" / "**maximum** temperature" |

Negations and numbers dominate deliberately: they are exactly what a compressor strips and exactly what a
384-dimensional embedding is worst at distinguishing. A pair that a cosine similarity easily separates
teaches nothing — **a good adversarial pair is one you expect the cache to get wrong.**

Aim for pairs whose cosine similarity exceeds 0.90. If you can check that cheaply once the embedder exists,
do — pairs below 0.85 should be replaced with harder ones.

---

## 4. `gold.jsonl` — the only ground truth

40 questions with human-written reference answers. Everything else in the evaluation is a proxy.

```json
{
  "gold_id": "gold_003",
  "question": "How many minutes are there in 3.5 hours?",
  "gold_answer": "210",
  "match": "numeric",
  "tolerance": 0.0,
  "acceptable_variants": ["210 minutes"]
}
```

`match` ∈ `exact` | `numeric` | `set` | `contains`. Grading rules are declared **per item, in advance** —
deciding what counts as correct after seeing the output is how gold subsets stop being ground truth.

**Composition:** 15 arithmetic/unit (unambiguous), 15 short factual (single verifiable value), 10 multi-turn
where the answer depends on an earlier turn. Skew toward objectively checkable answers — a "gold answer" for
an open-ended question is just another opinion.

**Rule, in bold because it is the one that will be broken accidentally: write these before running the
system on them.** Git history is the evidence, so commit them as a separate commit before the first sweep.

---

## 5. Freeze procedure

Once complete, before any sweep:

```bash
cd corpus
sha256sum conversations.jsonl adversarial_pairs.jsonl gold.jsonl > MANIFEST.sha256
git add -A && git commit -m "Freeze evaluation corpus v1"
git tag corpus-v1
```

`corpus_hash` (the hash of `MANIFEST.sha256`) is written into every ledger row. Any post-freeze edit changes
the hash and makes the affected rows visibly incomparable — which is the point. A tagged commit whose
timestamp predates the first sweep is the only real answer to "was the corpus tuned to the system?", and the
report's own risk register anticipates that question.

**If the corpus must change after freeze** (a genuine error, not a convenient one): bump to `corpus-v2`,
re-run affected cells, and report both. Never edit in place.

---

## 6. Suggested schedule

| When | What |
|---|---|
| **Now → 9 Aug** | 30 conversations (5 per class) → corpus v0, unblocks Sprint 0's baseline |
| 17 Aug → 30 Aug | 50 adversarial pairs (needed for Sprint 2's cache work) |
| 31 Aug → 13 Sep | Scale to 150; write the 40 gold answers; cross-review; **freeze** |

The gold answers land before Sprint 4, which is before any results exist to be tempted by.
