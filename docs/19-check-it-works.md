# Check It Works — copy-paste prompts

**Prepared.** 25 October 2026 · verified against `http://127.0.0.1:8600` on a **cold server**
**Model.** `qwen2.5:1.5b-instruct` · **Encoder.** `ollama:all-minilm`

Forty-one prompts. Every expected result below is what the server actually returned, not what
it ought to return. Work **top to bottom** — the cache is real, and several checks depend on
what was asked before them.

### Before you start

1. Sidebar foot reads **running · real model**, green dot. Amber means `ollama serve` first.
2. **Restart the server** so the cache is cold: `Ctrl-C`, then `parsimony web`.
3. Run one throwaway question so the model is loaded.

Token counts may differ by ±2 from the numbers here; the *behaviour* in the right-hand column
is what you are checking.

---

## A · No AI at all — the calculator (M6a)

Paste each. All four must show **→ 0 tokens** and *"the AI was never called"*.

```
What is 17 * 4?
```
```
What is 256 / 8?
```
```
What is 1234 + 5678?
```
```
What is 15% of 200?
```

| ✓ | Expect |
|---|---|
| ☐ | 24 → **0**, *worked it out exactly (arithmetic)* |
| ☐ | 25 → **0**, arithmetic |
| ☐ | 29 → **0**, arithmetic |
| ☐ | 27 → **0**, *worked it out exactly (**percent**)* |

---

## B · Politeness and filler (M1 tier 1)

```
Hello! Could you please tell me, what is recursion? Thanks!
```
```
Hi there, I was just wondering what a hash table is. Thanks in advance!
```
```
Good morning. Kindly explain what an index is. Many thanks.
```

| ✓ | Expect |
|---|---|
| ☐ | 29 → **22** |
| ☐ | 32 → **27** |
| ☐ | 29 → **21** |

Open **All of it as a table** and check the *Politeness remover* row names what it removed.

---

## C · Repeated sentences (M1 tier 2)

```
The budget is 84000 pounds. The budget is 84000 pounds. What is the budget?
```

| ✓ | Expect |
|---|---|
| ☐ | 42 → **31**, *dropped 1 repeated sentence(s)* |

```
Probation lasts six months. Probation lasts six months. How long is probation?
```

| ✓ | Expect |
|---|---|
| ☐ | 32 → **32** — **reverted**. The gate blocks this one, because the repeat-remover wanted to take the question itself. A good accidental demonstration. |

---

## D · Wordy phrasing (M1 tier 3)

```
A large number of staff would like to book the lab on a regular basis; what is the rule?
```
```
The majority of visitors are able to sign in prior to the meeting, is that right?
```

| ✓ | Expect |
|---|---|
| ☐ | 36 → **29**, *shortened 3 wordy phrase(s), saving 7 tokens* |
| ☐ | 33 → **28**, *shortened 3 wordy phrase(s), saving 5 tokens* |

---

## E · The gate refusing (M8) — **the important one**

```
Explain the deadline. The deadline is 15 March. The deadline is 16 March.
```
```
Explain the limit. The limit is 7 days. The limit is 14 days.
```

| ✓ | Expect |
|---|---|
| ☐ | 35 → **35**. *Repeat remover* **reverted** — *REFUSED, it would change the meaning* |
| ☐ | 34 → **34**. Reverted on the number *14* |

```
Explain the rule. Encryption is required. Encryption is required.
```

| ✓ | Expect |
|---|---|
| ☐ | 27 → **23** — a *genuine* duplicate, so this one is allowed through. Shows the gate is not simply blocking everything. |

---

## F · Answer-length budget (M5)

```
What is the capital of France?
```
```
Write a Python function that reverses a linked list.
```
```
Why do small language models spend most of their time on prefill rather than decode?
```

| ✓ | Expect — read the *Answer limiter* row |
|---|---|
| ☐ | **factual question, so up to 128 tokens** |
| ☐ | **code question, so up to 512 tokens** |
| ☐ | **reasoning question, so up to 640 tokens** |

---

## G · The document (M1 context) — **click "Attach the handbook" first**

```
What is the annual travel budget for the Leeds office?
```
```
How many people work in the Porto office?
```
```
What is the daily meal allowance?
```
```
How long is a standard equipment loan?
```
```
Who is the Tallinn site manager?
```

| ✓ | Expect |
|---|---|
| ☐ | 892 → **289** · kept **13 of 49** sentences |
| ☐ | 890 → **363** · kept **17 of 49** |
| ☐ | 888 → **51** · kept **1 of 49** |
| ☐ | 889 → **150** · kept **7 of 49** |
| ☐ | 889 → **330** · kept **15 of 49** |

*The number kept varies with the question. That is the point — it is question-aware.*

---

## H · Off topic — the document cannot answer (handbook attached)

```
What is the capital of Peru?
```
```
Who won the 1998 World Cup?
```

| ✓ | Expect |
|---|---|
| ☐ | 888 → **58** · kept **1 of 49** — context judged off topic |
| ☐ | 892 → **62** · kept **1 of 49** |

*One sentence is kept, not none: an empty context reads to the model as an instruction with a
missing attachment.*

---

## I · Memory, exact repeat (M2)

Re-ask two questions from §G and §F **word for word**:

```
What is the daily meal allowance?
```
```
What is the capital of France?
```

| ✓ | Expect |
|---|---|
| ☐ | 888 → **0** · *the same question, word for word* |
| ☐ | 22 → **0** · same |

---

## J · Memory, reworded — **the headline result**

Handbook attached. These were never asked, but §G asked them *differently*:

```
How much do staff get for meals each day?
```
```
What is the headcount of the Porto office?
```
```
What is the loan period for standard equipment?
```

| ✓ | Expect |
|---|---|
| ☐ | 891 → **0** · *reused an earlier answer — **71 % match*** |
| ☐ | 891 → **0** · ***87 % match*** |
| ☐ | 890 → **0** · ***90 % match*** |

**Point at the "Answered from memory" panel.** Both questions side by side. This is the newest
work in the project: 4 of 40 reworded questions were reused before it, **29 of 40** now.

---

## K · Memory must **refuse** — the safety check

```
What is the annual travel budget for the Tallinn office?
```
```
How many people work in the Leeds office?
```
```
What is the nightly hotel limit?
```

| ✓ | Expect — **none of these may be reused** |
|---|---|
| ☐ | 893 → **266** · kept 12 of 49 — *Tallinn ≠ Leeds* |
| ☐ | 890 → **335** · kept 16 of 49 — *Leeds ≠ Porto* |
| ☐ | 888 → **212** · kept 9 of 49 |

> **If any of these three returns 0 tokens and "reused an earlier answer", stop.** That is a
> wrong answer being served, and it is the one failure this project treats as unacceptable.

---

## L · Two different questions about the same office

The pattern that found a real bug on 25 October. Ask **in this order**:

```
What is the annual travel budget for the Leeds office?
```
```
How many people work in the Leeds office?
```

| ✓ | Expect |
|---|---|
| ☐ | first: compressed and answered |
| ☐ | second: **compressed and answered — NOT reused** |

*Same office, both asking for a number, almost no words in common. Until ADR-061 the second
was served with the first's answer. If it is reused again, the overlap floor has moved.*

---

## M · The Heatmap controls

Load the sample handbook, ask `When does the office close for Christmas?`, press **Compress**.

| ✓ | Expect |
|---|---|
| ☐ | Chart of 49 bars, tallest first, dashed cut-off line |
| ☐ | **Decide the cut-off by → reading the scores**: takes **8–12 s**, says so on screen, then reports a floor read off the distribution |
| ☐ | **Score the sentences with →** the other encoder: **the bars move** |
| ☐ | Drag **Cut-off score**: the line and the kept/dropped colours follow |

---

## N · Totals

Open **Totals** after all of the above.

| ✓ | Expect |
|---|---|
| ☐ | **tokens removed** and **seconds saved** are non-zero |
| ☐ | **answered without the model** ≥ 6 (four calculator + the cache hits) |
| ☐ | **changes refused and undone** ≥ 2 — the gate said no |

---

## If something is wrong

| Symptom | Likely cause |
|---|---|
| Everything returns 0 tokens | cache warm from an earlier session — restart the server |
| Sidebar says **simulated** | Ollama not running |
| §K or §L reuses an answer | **a real defect** — do not demo; the floor has regressed |
| First question takes 30 s | model loading; run a throwaway first |
| Heatmap seems frozen | re-scoring 49 sentences; it says so, wait |
