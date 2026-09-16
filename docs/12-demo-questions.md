# Demo Questions

Every question below was run through the pipeline and the modules it fires were **measured**, not
guessed. Numbers are input tokens before → after.

Use them with:

```powershell
.\demo.ps1 ask
```

Type the question, press Enter. You get a numbered walkthrough: only the steps that actually did
something, each showing the sentence before and after, the words that were deleted, and the tokens
saved. Add `--detail` if you also want the engineering trace underneath.

---

## What you will see

Each turn is drawn **as it runs**: every layer fills in with its own duration and what it did, the
prompt bar shrinks as cuts are committed, a clock runs while the model reads the prompt, and the answer
streams in against the budget the answer limiter chose. This is a live capture, not a drawing:

```
Parsimony  Hello, I was wondering if you could please explain what recursion
is? Thanks!

Prompt  ██████████████████████░░░░░░░░░░░░░░  17 tokens   11 removed (39%)

    Layer                   Time   Tokens  What happened                    
✓   Calculator            0.3 ms        -  not a sum or a date question     
✓   Memory                1.2 ms        -  nothing stored yet to compare    
                                           against                          
✓   History trimmer      <0.1 ms        -  no earlier turns to work with yet
✓   History arranger     <0.1 ms        -  no earlier turns to work with yet
✓   Context selector     <0.1 ms        -  no documents attached and no long
                                           earlier answers                  
✓   Politeness remover    1.2 ms      -10  removed 10 tokens of greeting or 
                                           formatting                       
✓   Repeat remover        0.2 ms        -  no repeated sentence             
✓   Wordiness trimmer     3.1 ms        -  no wordy phrase worth shortening 
✓   Prompt arranger       0.5 ms        -  pinned 8 unchanging tokens to the
                                           front                            
✓   Answer limiter        0.2 ms        -  factual question, so up to 128   
                                           tokens                           
✓   Model chooser         0.1 ms        -  simple enough (0.19) for the     
                                           small model                      

   ✓ read the prompt (simulated: no real reading time)                  
   ✓ wrote 53 tokens of 128 allowed   stopped early: restated a sentence
╭─ Answer ─────────────────────────────────────────────────────────────────╮
│ It is worth considering the trade-offs before deciding. This behaviour   │
│ is consistent across the common implementations. The answer depends on   │
│ the specific context you are working in. There are several factors that  │
│ influence the outcome here. In most practical cases the standard         │
│ approach is sufficient. It is worth considering the trade-offs before    │
│ deciding.                                                                │
╰──────────────────────────────────────────────────────────────────────────╯╭─ What the AI actually received  struck-through text was removed before s─╮
│ Instructions        You are a concise, accurate assistant.   (always     │
│                     first, never changes)                                │
│ Your question       Hello, I was wondering if you could please explain   │
│                     what recursion is? Thanks!                           │
│ sent as             Explain what recursion is?                           │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Measured ───────────────────────────────────────────────────────────────╮
│ Prompt: 28 tokens as written -> 17 sent (11 removed).                    │
│ Reading time is not measured here - this is the simulated model. At the  │
│ 8.5 ms per token measured on the real one, 17 tokens is about 0.1 s      │
│ against 0.2 s.                                                           │
│ The answer was cut short because it restated a sentence.                 │
│ Simulated AI: the layers are real, the timings are not. Start Ollama for │
│ real ones.                                                               │
╰──────────────────────────────────────────────────────────────────────────╯
This conversation: 1 question(s)  |  17 of 28 prompt tokens sent (39% 
removed)  |  the AI spent 0.0 s reading (simulated) instead of about 0.2 s
```

Read it top to bottom and it explains itself:

- **The bar** is the prompt: solid for what is being sent, hollow for what was removed.
- **Every layer is listed, every turn**, including the ones that did nothing - each says *why*, and how
  long it took. A layer that declines is still accounted for.
- **The model's own time** is separated into reading the prompt and writing the answer, because on this
  CPU reading is 92-99% of it. Against a real model both come from the runtime's counters.
- **What the AI actually received** shows the text that went, with everything removed struck through -
  so the claim in the bar can be checked by eye.
- **Measured** converts the tokens into seconds, and says plainly when a figure is an estimate.

Type `compare` at any point and the same question is asked again with every layer switched off, from
cold, on the same model - two measurements side by side rather than an estimate.

Inside `ask`, type `help` to list the layers and `memory` to see what has been remembered.

---

## Part 1 — One tier at a time

### M1 tier 1 · strips politeness and boilerplate

```
Hello, I was wondering if you could please explain what recursion is? Thanks!
```

**28 → 17 tokens.** Removes `"Hello, I was wondering if you could please"` and `"Thanks!"`.
Say: *"Politeness carries no instruction. It costs tokens and teaches the model nothing."*

### M1 tier 2 · drops a sentence that repeats a fact

```
Summarise this. The server runs Ubuntu Linux. The server runs Ubuntu Linux and needs a restart. Please advise.
```

**36 → 26 tokens.** The duplicate sentence goes; the one carrying new information stays.

### M1 tier 3 · shortens wordy phrasing

```
Please provide me with an explanation of how in order to configure the settings you would go about doing it.
```

**33 → 30 tokens.** Fires tier 1 *and* tier 3 — "in order to" and "go about doing" are the tier-3 targets.
This is the smallest effect of the three tiers, which is itself the honest finding: tier 3 contributes
least, and the ablation says so.

### M8 · the fidelity gate refusing an edit

```
Explain the deadline. The deadline is 15 March. The deadline is 16 March.
```

**32 → 32 tokens — deliberately unchanged.** Tier 2 proposes deleting the "duplicate" sentence; the gate
blocks it because the two carry different dates. The report reads *"Blocked because: rewrite dropped
number:1"*.

Also works:

```
Check the price. The price is 40 dollars. The price is 45 dollars.
```

**This is the most important question in the set.** It is the one that shows the system declining to
optimise.

### M6 · answered with no model at all

```
What is 847 * 23?
```

**23 → 0 tokens**, served by `DETERMINISTIC`. A calculator answered it; the model was never called.

```
How many days between 2026-01-15 and 2026-08-05?
```

**40 → 0 tokens.** Same tier, different handler.

### M5 · the output budget changes with the question

```
Why does quicksort degrade to quadratic time, and how would you avoid it?
```

Budget **640 tokens** — classified as reasoning. Compare with the deterministic-tier question above, which
is classified arithmetic and budgeted **48**. Same module, a 13× difference in what it allows.

### M2 · the memory (semantic cache)

The memory is the layer people ask about most, so it has the most questions. Every result below is what
the pipeline actually does — measured, not expected. Each needs one unrelated question in between
(`What is a pointer?` works), so the reuse is not just the previous turn.

**1. The same question again — reused.**

```
Explain recursion.
```
```
What is a pointer?
```
```
Explain recursion.
```

The memory panel says *"the same wording as before — no similarity needed"*. The AI is never called.

**2. The same question, worded differently — reused.** This is the interesting one:

| Asked first | Asked later | Result |
|---|---|---|
| `Hello, could you please explain what recursion is? Thanks!` | `What is recursion?` | **reused** — 100% match |
| `How does a hash table work?` | `Explain how hash tables work` | **reused** — 100% match |
| `What causes rain?` | `Why does it rain?` | **reused** — 100% match |

The memory compares the questions with courtesy ("could you please… thanks"), request words ("explain",
"tell me") and plurals set aside. All three pairs used to **miss** — the polite one scored 63%, the
plural 71%, and "Why does it rain?" was never compared at all, because the word *it* made the system
treat it as a follow-up. All three were fixed and are now tested.

**3. A question that only LOOKS the same — refused.** This is the safety property, and the best thing to
show someone:

```
Is it safe to mix bleach and vinegar?
```
```
What is a pointer?
```
```
Is it not safe to mix bleach and vinegar?
```

The two are **91% similar** — close enough that a similarity-only cache would hand back the opposite
answer. The memory panel shows the safety checks, with **"not / never DIFFER"** in red, and refuses.

**4. Same words, a different kind of question — refused.**

```
What is Java?
```
```
What is a pointer?
```
```
Where is Java?
```

These score **100%** — both reduce to the one subject word "Java". But one asks *what* (the language) and
the other *where* (the island). The panel reads *"earlier asked what, this asks where (DIFFERENT)"* and
refuses. Before this was fixed, the memory answered the where-question with the what-answer.

**5. A follow-up — correctly not reused from elsewhere.**

```
Explain what a hash table is.
```
```
Why does that affect lookup time?
```

*"that"* points back at something earlier, so the question only means something inside this conversation.
The panel names the word: *"treated as a follow-up because of the word 'that'"*.

The whole set of rephrasings the memory is checked against — 37 pairs, half of which must **not** be
reused — is in `corpus/interactive_pairs.jsonl`, and every pair is a test.

### M3 · the history manager

Needs enough conversation to have something worth dropping. Measured turn by turn:

| turn | history | M3 |
|---|---|---|
| 1 | 0 turns | skipped — not applicable |
| 2–4 | 2–6 turns | no-op — *all turns fit the budget* |
| **5** | **8 turns** | **applied — `mmr: kept 6 of 8 turns`** |
| 6+ | 10+ turns | applied — keeps 6, drops the rest |

**So ask five questions before expecting M3 to do anything.** Turns 2–4 reporting "no-op" is the module being
honest, not broken — there was nothing to trim.

Any five questions work; these are convenient because they build on each other:

```
Explain what a hash table is.
```
```
How does it handle collisions?
```
```
What is a good load factor?
```
```
Why does that affect lookup time?
```
```
How do I choose a hash function?
```

---

## Part 2 — Five questions that fire several tiers at once

### 1. Tier 1 + tier 2 — **43 → 25 tokens, −42%**

```
Hello! Could you please summarise this for me? The server runs Ubuntu Linux. The server runs Ubuntu Linux and needs a restart. Thanks in advance!
```

Fires **T1, T2, M4, M5, M6**. Politeness stripped *and* the duplicate sentence dropped.

### 2. Tier 1 + tier 3 — **42 → 36 tokens**

```
Hi there, I was wondering whether or not you could please tell me how in order to reverse a string I would go about doing it? Thanks!
```

Fires **T1, T3, M4, M5, M6**. Boilerplate at the edges, wordiness in the middle.

### 3. All three tiers **and** the gate — **66 → 55 tokens**

```
Hello! I would like to know whether or not you could please summarise this for me. The service runs on a laptop. The service runs on a laptop with 8 GB of RAM. In order to deploy it I use systemd. Thanks so much in advance!
```

Fires **T1, T2, T3, M4, M5, M6 — and the gate blocks one edit.** Six modules acting plus a refusal, in a
single question.

**This is the best single question in the whole set.** The two sentences look duplicated, but the second
carries "8 GB of RAM" — so the gate refuses that deletion while the other tiers still do their work. It
shows the whole pipeline cooperating *and* disagreeing.

### 4. Tier 1 + the gate — **41 → 34 tokens**

```
Hello, could you please explain the deadline to me? The deadline is 15 March. The deadline is 16 March. Thanks!
```

Fires **T1, M4, M5, M6, with T2 blocked.** Cleaner than #3 if you want the gate to be the story: boilerplate
is removed, and the one edit that would have cost information is refused.

### 5. Tier 1 + tier 2 + a large budget — **45 → 27 tokens, −40%**

```
Hi! Could you please explain why quicksort degrades to quadratic time? The pivot choice matters. The pivot choice matters a great deal here. Thanks in advance!
```

Fires **T1, T2, M4, M5, M6**, and M5 sets a **640-token** budget because the question is reasoning-class.
Shows the compressor shrinking the input while the budgeter *grants* more output — the two modules pulling
in opposite directions, correctly.

---

## A sixth, if you want to show stage order matters

```
Hello there, I was wondering if you could please tell me what 847 * 23 is? Thanks!
```

The same arithmetic as the deterministic-tier question, wrapped in politeness — and it does **not** reach
the deterministic tier. It is served by the model instead.

The reason is stage order: the deterministic handler runs *before* the compressor, so it sees the verbose
text and its pattern does not match. Strip the politeness first and it would.

This is the same class of effect as research gap 3 (what the cache sees depends on where it sits), and it is
an honest thing to show rather than hide: **the order of the stages changes the result, which is exactly why
this project makes stage order configuration rather than code.**

---

## Part 3 — A document, where compression is worth seconds

Everything above is a question of a few dozen tokens, so the most any compressor can save is a few dozen
tokens. Attach a file and the same layers have something to work with:

```powershell
.\demo.ps1 ask --file examples/staff-handbook.md
```

The handbook is 817 tokens in seven sections. Every question below is answered from it, and the context
selector decides which sentences the model reads. Typical: **888 tokens as written, about 320 sent.**

| Ask this | What happens, checked against a live run |
|---|---|
| How many people does the Tallinn office employ? | 888 → 327 tokens. The answer sentence is "It employs 58 people." and it never says Tallinn: it is found because it inherits the name from the sentence before it, and that sentence is kept so "It" still refers to something. The Onboarding section goes entirely. |
| For how many days can an oscilloscope be borrowed? | 890 → 161 tokens, the largest cut of the five. Every office section disappears; both loan periods (14 days and 7) stay, because both sentences are about loans and only the model can pick between them. |
| Which office has more employees, Porto or Tallinn? | 889 → 337 tokens, and **both** counts survive — "It employs 58 people" and "The office employs 41 people" — from two different sections. A two-part question needs both, and dropping either would make it unanswerable. |
| What is the annual travel budget for the Tallinn office? | 890 → 318 tokens. All three offices' travel budgets survive here, not just Tallinn's: each is a sentence about travel budgets, and the selector ranks rather than decides. What it does guarantee is that the Tallinn section is represented — the budget sentence there says only "The office", so a name match alone would have missed it. |
| What is the capital of Peru? | **The honest one to show.** Nothing in the handbook bears on the question, and the selector still keeps about 40% of it: scores are relative, so "least irrelevant" still wins a place. Say so — it is a named limitation, not a surprise, and the fix (an absolute relevance floor) is measurable rather than a matter of taste. |

Then type `compare`. The same question is asked again with every layer switched off, from cold, on the
same model — on this laptop that is **8.9 s of reading against 3.0 s**, for the same answer.

To see the decision itself rather than its effect, without needing the model at all:

```powershell
.\demo.ps1 sentences
```

Every sentence of six documents marked kept or removed, the answer sentences in bold, and a line saying
what stopped the selection, which names were guaranteed a sentence, and how many sentences were kept only
so a following "It" makes sense.

---

## What to say while these run

- **On a no-op:** *"That module reports doing nothing, and says why. A stage that stayed silent would be
  unauditable — the trace never hides one."*
- **On the gate:** *"It found a saving and refused to take it. That is the module that makes the other seven
  safe to use."*
- **On the deterministic tier:** *"Zero tokens. The cheapest request is the one you never send."*
- **On the net-effect line:** *"Every edit passed the gate, so the meaning is intact. The saving is free."*
