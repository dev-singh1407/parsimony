# Viva Questions — the hard ones

**Prepared.** 24 October 2026
**Companion to** [`11-demo-runbook.md`](11-demo-runbook.md) §3, which covers the first-pass
questions (real model or simulated, why prefill, why not keep the last few turns, the encoder
regression, the `localhost` deduction). This sheet covers what comes **after** those, and the
questions a hostile examiner asks.

Every answer names where the evidence is. If you cannot point at it, do not claim it.

---

## A. The base papers

### "Which papers is this based on?"

> Two. **CPC** (Liskavets et al., AAAI-25) for compression — sentence-level, question-aware,
> scored by a sentence encoder, which is M1's design arrived at independently. And **vCache**
> (Schroeder et al., ICLR 2026) for reuse — it argues a single static similarity threshold
> gives a semantic cache no formal correctness guarantee.
>
> LLMLingua (EMNLP 2023) and GPTCache (NLP-OSS 2023) founded the two strands and are related
> work; LLMLingua contributes the budget controller that survives as M5.

### ⚠ "vCache already found that thresholds don't work. So what is new here?"

**This is the hardest question on the sheet. Do not get defensive — the answer is good.**

> Three things.
>
> **One, we reached it independently and earlier.** E3 was measured before that paper was
> read. Two groups converging on the same diagnosis from different directions is stronger
> evidence than either alone — it is the difference between a finding and an anecdote.
>
> **Two, the remedies differ, and ours suits a different machine.** vCache estimates how
> wrong a threshold is likely to be and bounds the error statistically, learning a threshold
> per cached prompt online. We refuse **categorically**: compare numbers, entities,
> negations, operative modifiers and what kind of answer is wanted, and reject if any differ.
> Four set comparisons, microseconds, no second neural pass, no per-entry state. A statistical
> bound suits high-volume serving. A categorical refusal suits one user on a CPU.
>
> **Three, we measured the side they leave open.** vCache raises hit rate under an error
> bound. It does not ask how much *legitimate* reuse a safe cache permits. We did: **4 of 40
> reworded questions, rising to 31 of 40**, with zero wrong answers at every step.

*Evidence:* paper §E3, §E7; `docs/15-base-papers.md`.

### "Did you implement CPC or vCache and compare against them?"

> **No, and that is a real limitation.** We compare against an uncompressed baseline and
> against our own ablations, not against their implementations. CPC needs a contrastively
> trained encoder we did not train; vCache needs the online threshold learner. The honest
> claim is that our design agrees with CPC's and our diagnosis agrees with vCache's, not that
> we beat either. Implementing them is the obvious next work.

---

## B. The cache result

### "78 % reuse — why not higher?"

> Because the next four points cost two wrong answers, and we publish the table that shows
> it. At `tau_lo` 0.50 reuse goes to 30 of 40 and two near-misses get served. We stopped at
> the last operating point that gives up nothing.

*Evidence:* paper §E7, the joint sweep table.

### ⚠ "You tuned thresholds until the number looked good."

> The opposite, and the table shows it. Reuse went from 4 to 31 across five changes, and
> **only one of them moved a threshold** — from 0.70 to 0.60, as far as a corpus authored
> before this work allowed. The other four fixed mechanisms: the encoder was being handed a
> sentence fragment, the overlap floor required 55 % of words to be *identical* which a
> paraphrase by definition is not, "what" was compared as if it stated a kind, and the entity
> extractor had two bugs.
>
> We re-ran the threshold sweep after the mechanism fixes. **The same point was still
> optimal.** The ceiling rose because a mechanism started reading its input correctly, not
> because a gate was opened.

### "How do you know you haven't overfitted to your own 80 pairs?"

> Because the safety side is judged by two corpora this work did not author —
> `adversarial_pairs` and `interactive_pairs`, both frozen. Every change had to leave both
> unchanged. One candidate rule gained three rewordings and broke a frozen pair; it was
> discarded. The 80 pairs can only ever flatter recall; they cannot flatter safety.
>
> We also kept the 80 **separate** from the frozen corpora rather than adding them, because
> adding questions to a frozen instrument after seeing a failure is not measurement.

---

## C. The fidelity gate

### ⚠ "Your gate let through 55 % of meaning-changing rewrites. Isn't that the whole architecture failing?"

**Expect this. It is the most damaging-sounding fact in the project, and the answer is the
best thing in it.**

> It was, and we found it ourselves, by measuring. The gate asked one question — *was any
> extractable value lost?* — and a value that was never there cannot be missing. So it was
> blind to anything **added**. It admitted "visitors must sign in" becoming "must **not**
> sign in", and "maximum" becoming "minimum".
>
> It is now **3 of 40**, while refusing none of twelve legitimate rewrites. And the fix
> needed no new research: the semantic cache's verifier already checked those things. It had
> been built for M2 and never given to M8, which makes the stronger claim.
>
> The reason it survived so long is the reason worth telling you: it had passed every test
> written for it. There were eight of them.

*Evidence:* paper §E8; ADR-060; `tests/unit/test_gate_meaning.py`.

### "What does it still miss?"

> Three, and they are in the paper rather than hidden. **Attribution** — "the handbook states
> that X" becoming "X". **Word order** — "designed in Tallinn and built in Leeds" reversed,
> because both entities are present on both sides and a set comparison cannot see an order.
> And a qualifying **phrase** rather than a word — "allowed with manager approval" becoming
> "allowed". All three need structure the gate does not model. There is a test that **fails
> if they ever start being caught**, so the moment that changes we find out.

---

## D. Method

### "Why 60 architecture decision records?"

> Because several of them record the project being wrong, and those are the ones worth
> keeping. ADR-050 records six mechanisms that each worked correctly and changed recall by
> zero items, because a floor downstream was discarding what they found. ADR-055 records our
> own first measurement reading "4 of 5" when the honest figure was 45 %.

### ⚠ "Give me one thing you measured, believed, and later found was wrong."

**Have this ready. It is asked to see whether you will admit anything.**

> The cache reuse number. We measured five end-to-end pairs, got 4 of 5, and wrote it into
> two decision records. On eighty pairs written properly the real figure was **45 %, and two
> near-misses were being served** — wrong answers nobody had looked for. Both facts were
> invisible at five pairs.
>
> That is why the conclusion of the paper says a component is not measured until it has been
> given a set written to disagree with it.

### "What is your overhead?"

> **4.06 ms** per request, mean, measured. Against a prefill cost of about 8.5 ms per input
> token, the middleware pays for itself if it removes a single token.

---

## E. Trap questions

| Question | Short answer |
|---|---|
| **"Isn't this just RAG?"** | No. RAG *fetches* context; we *reduce* context the user already supplied. Both can run — M1 would compress what a retriever returned. |
| **"Why not fine-tune the model instead?"** | Fine-tuning changes the model; this changes the prompt. It needs no training, no GPU, no model internals, and works on any model behind any API. |
| **"Why not just use a bigger model?"** | Measured. ADR-037 escalated to a second model: **36/40 against 36/40**, item for item identical, for 16 % more wall clock and twice the memory. The router still classifies; escalation is switched off because it earned nothing. |
| **"Why a 1.5B model?"** | Because that is what runs on a CPU, and the whole premise is that prefill dominates there. The cost structure we exploit is strongest exactly where the hardware is weakest. |
| **"What happens if Ollama is down?"** | It degrades toward *keeping* context, not discarding it. That was a bug until ADR-059 — an outage used to make the compressor throw whole documents away. Every component now fails toward the missed saving rather than the wrong answer. |
| **"Does this generalise past one model?"** | Partly, and we measured which part. Per-module reductions replicate within 0.09 pp under a second tokenizer vocabulary; the *explanations* for them do not all transfer. A threshold belongs to the encoder it was set against. |

---

## F. Limitations — say these before you are asked

Volunteering a limitation costs nothing and buys credibility for everything else.

1. **Energy is not measured.** The estimator is degenerate — it attributes 0.0 J to one
   configuration and reports the full stack as 39 % *worse* than baseline, contradicting the
   directly measured prefill reduction. **No energy claim appears anywhere in the paper.** It
   needs RAPL or socket-level instrumentation.
2. **No comparison against CPC or vCache implementations.** See §A.
3. **One model family, one machine.** Qwen2.5-1.5B on one laptop CPU.
4. **The corpus is ours.** 151 conversations we authored. LongBench 2wikimqa is the external
   check, and it found three failures our own corpus could not.
5. **Nine rewordings still refused**, and fourteen of forty; the paper says which and why.
6. **Thirty references lack author lists** — being fixed.

---

## G. If you are asked something you do not know

> "I don't know — it isn't measured. What *is* measured is ⟨nearest thing⟩, and that says ⟨X⟩."

Never guess a number in a viva. Every figure on this sheet regenerates from logs by one
command, and an examiner who catches one invented number will doubt the other forty.
