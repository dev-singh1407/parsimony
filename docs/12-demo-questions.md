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

Ask something self-contained, ask two other things, then ask the first one again:

```
Explain recursion.
```
```
What is the capital of Australia?
```
```
What is a pointer?
```
```
What is the capital of Australia?
```

Turn 4 comes back **`CACHE_EXACT` — the model was never called.**

This only started working after ADR-039. Before it, every cache entry was tagged with a hash of the last two
turns, and that hash changes every turn — so an answer stored at turn 2 was filed under a label that no
longer existed at turn 4, and the lookup reported *"cache miss (no candidates)"*. Not "too dissimilar": no
candidate at all. The tag now applies only to questions that actually depend on the conversation.

A follow-up still misses, correctly:

```
Why does that affect lookup time?
```

*"that"* refers to something earlier, so its answer is scoped to this conversation and cannot be reused.

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

## What to say while these run

- **On a no-op:** *"That module reports doing nothing, and says why. A stage that stayed silent would be
  unauditable — the trace never hides one."*
- **On the gate:** *"It found a saving and refused to take it. That is the module that makes the other seven
  safe to use."*
- **On the deterministic tier:** *"Zero tokens. The cheapest request is the one you never send."*
- **On the net-effect line:** *"Every edit passed the gate, so the meaning is intact. The saving is free."*
