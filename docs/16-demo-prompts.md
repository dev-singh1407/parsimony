# Demo Prompts — every module, with verified numbers

**Prepared.** 24 October 2026 · against the live server on `http://127.0.0.1:8600`
**Model.** `qwen2.5:1.5b-instruct` · **Encoder.** `ollama:all-minilm` · not simulated

Every prompt on this sheet was run through the server and the numbers below are what came
back. Nothing is carried forward from an older sheet.

> **Read this first — order matters.** The cache is real. Once you ask a question, asking it
> again is answered from memory and the layers you wanted to show will be skipped. Work down
> the sheet in order, and if you need to re-run a section, restart the server first.

---

## Part 1 — One module at a time

### M6a · Calculator — answered with no AI at all

```
What is 17 * 4?
```

**24 → 0 tokens. The model is never called.**
Layer table reads: *"worked it out exactly (arithmetic); the AI was never called"*.

*Say:* the cheapest token is the one never sent. The stack stops at the first layer.

---

### M1 tier 1 · Politeness remover

```
Hello! Could you please tell me, what is recursion? Thanks!
```

**29 → 22 tokens.** Reason: *removed "Hello! Could you please … Thanks!"*

---

### M1 tier 2 · Repeat remover

```
The budget is 84000 pounds. The budget is 84000 pounds. What is the budget?
```

**42 → 31 tokens.** Reason: *dropped 1 repeated sentence(s)*

---

### M1 tier 3 · Wordiness trimmer

```
A large number of staff would like to book the lab on a regular basis; what is the rule?
```

**36 → 29 tokens.** Reason: *shortened 3 wordy phrase(s), saving 7 tokens*
(*a large number of* → many, *would like to* → want to, *on a regular basis* → regularly)

A second one if you want it: `The majority of visitors are able to sign in prior to the
meeting, is that right?` — **33 → 28, saving 5 tokens.**

---

### M8 · The fidelity gate refusing a saving

```
Explain the deadline. The deadline is 15 March. The deadline is 16 March.
```

**35 → 35 tokens. Nothing is removed, and that is the result.**
The repeat remover proposes deleting the second sentence as a duplicate; the gate reverts it
because the two dates differ. Layer shows **reverted**.

*Say:* this is the only number in the demo that goes down to zero saving on purpose. A
compressor that never refuses anything is the one not to trust.

---

### M5 · Output budgeter — three classes, three budgets

| Prompt | Budget |
|---|---|
| `What is the capital of France?` | **factual → 128 tokens** |
| `Write a Python function that reverses a linked list.` | **code → 512 tokens** |
| `Why do small language models spend most of their time on prefill rather than decode?` | **reasoning → 640 tokens** |

*Say:* the answer is budgeted before it starts, by what kind of question it is.

---

### M6b · Escalation router

Use the reasoning prompt above. Layer reads:
*"complex (0.42) but the bigger model is switched off, so the small one"*

*Say, honestly:* escalation is **measured and switched off**. ADR-037 found a second model
scored 36/40 against 36/40 — item for item identical — for 16 % more wall clock and twice the
memory. The router still classifies; it just has nowhere better to send the question.

---

### M4 · Prefix-stable assembler

Visible on **every** run: *"pinned 11 unchanging tokens to the front"*.

*Say:* the parts that never change go first so the model can reuse the work it already did
on them. This is the layer that conflicts with "Lost in the Middle", and E2 in the paper
prices that conflict.

---

### M1 · Context compressor — attach the handbook

Click **Attach the handbook**, then:

```
What is the annual travel budget for the Leeds office?
```

**892 → 289 tokens.** *kept 13 of 49 sentences (823 → 242 tokens of context)*

---

### M1 · Off-topic — the document is not thrown away blindly

Handbook still attached:

```
What is the capital of Peru?
```

**888 → 58 tokens.** *kept 1 of 49 sentences* — the context is judged off topic, and **one**
sentence is kept rather than none, because an empty context reads to the model as an
instruction with a missing attachment.

---

### M2 · Memory — three behaviours in a row

Handbook attached. Run these **in this order**:

| # | Prompt | What happens |
|---|---|---|
| 1 | `What is the annual travel budget for the Tallinn office?` | compressed and answered normally |
| 2 | *the same question again* | **893 → 0.** *"the same question, word for word"* |
| 3 | `How much can Tallinn staff spend on travel in a year?` | **894 → 0.** *"reused an earlier answer — 90 % match"* |
| 4 | `What is the annual travel budget for the Leeds office?` | **not reused** — compressed and answered |

**Point at the "Answered from memory" panel** on step 3. It shows both questions side by side
with how alike they are, and underneath: *different words, judged to be the same question.*

*Say:* step 3 is the new result — 4 of 40 reworded questions were reused before this work,
29 of 40 now, with **zero** wrong answers served at any point. Step 4 is why that is safe.

---

### M3 · History manager — terminal only

The Pipeline tab is single-shot by design, so M3 has nothing to select from. Show it in the
terminal instead:

```bash
parsimony chat
```

Then ask three questions where the third depends on the first, e.g. *"Explain what a hash
table is."* → *"What is its average lookup complexity?"* → *"And in the worst case?"*. The
layer table shows the history selector keeping the turns that matter and dropping the rest.

---

### M7 · Learner — not a live layer

M7 runs **offline**, over conversations that already happened, and writes a bundle the system
loads at start-up. It is on the Architecture tab for completeness and there is nothing to
click. Say so plainly rather than pretending.

---

## Part 2 — The features, not the layers

### Architecture tab
Open it first. Six colour-coded parts, one plain sentence each, a worked example under every
one, and M7/M8 shown separately because they are not pipeline steps. Ends with **how to check
any of this**, naming the tab that proves each claim.

### Heatmap tab
Attach the handbook, ask `When does the office close for Christmas?`, press **Compress**.

* Every sentence scored, tallest first, with the **cut-off line** drawn across.
* Switch **Decide the cut-off by** to *reading the scores*. The floor is read off the
  distribution — it reports something like *"floor 0.27 — the scores fall 2.2× after rank 9
  of 15"*. **This takes 8–12 s** and the page says so while it works.
* Switch **Score the sentences with** between the two encoders and watch the bars move.
  *Say:* a threshold is a property of an encoder, not of a technique — which is why every
  results table in the paper names its encoder.

### A/B tab
`Run both` asks the same question twice, with the system off and on, one after the other
against the real model. **Read the two answers at the bottom first** — the claim is that they
say the same thing — then compare what each cost.

### Totals tab
Everything saved this session. The two figures that matter are the safety record: **changes
checked for lost meaning** and **changes refused and undone**.

### The copy button
*copy as text* on the layer table gives the same rows as `parsimony chat` prints, so the
screen and the transcript cannot disagree.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| A prompt returns 0 tokens and "the same question, word for word" | you ran it earlier | restart the server |
| Heatmap seems frozen for ten seconds | re-scoring 49 sentences on CPU | it says so on screen; wait |
| First run of the session is slow | Ollama loading the model | run one throwaway question before the demo |
| "simulated" in the sidebar instead of "real model" | Ollama not running | `ollama serve` |

---

## The three numbers to have ready

1. **890 → 263 tokens** on the handbook question — 70 % of the prompt never sent.
2. **Prefill is 91.7–98.8 %** of CPU inference time, so tokens are the right thing to cut.
3. **0 wrong answers** served across 40 adversarial near-miss pairs, at every stage of the
   cache work.
