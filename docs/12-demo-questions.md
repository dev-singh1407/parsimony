# Demo Questions

Every question below was run through the pipeline and the modules it fires were **measured**, not
guessed. Numbers are input tokens before → after.

Use them with:

```powershell
.\demo.ps1 ask
```

Type the question, press Enter. You get a panel per module plus a written report of what each one did.

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

### M2 · the semantic cache

Needs two turns, because the cache is scoped to a conversation:

```
What is the capital of Australia?
```
```
reset
```
```
What is the capital city of Australia?
```

**`CACHE_SEMANTIC` — the model was never called.** Different wording, same question, and the verifier
confirmed the two agree on every number, entity, negation and modifier.

### M3 · the history manager

Needs a conversation. Ask any four questions in a row; by the third or fourth turn M3 has history worth
trimming and reports `mmr: kept 6 of 8 turns`.

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

## What to say while these run

- **On a no-op:** *"That module reports doing nothing, and says why. A stage that stayed silent would be
  unauditable — the trace never hides one."*
- **On the gate:** *"It found a saving and refused to take it. That is the module that makes the other seven
  safe to use."*
- **On the deterministic tier:** *"Zero tokens. The cheapest request is the one you never send."*
- **On the net-effect line:** *"Every edit passed the gate, so the meaning is intact. The saving is free."*
