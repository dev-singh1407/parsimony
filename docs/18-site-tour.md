# A Tour of the Site

**Prepared.** 25 October 2026 · `http://127.0.0.1:8600`
**Companion to** [`16-demo-prompts.md`](16-demo-prompts.md), which is the prompt list alone.
This walks the screen itself — every tab, every panel, what it is and what it proves.

> **Before you start.** The sidebar foot must read **running · real model** with a green dot.
> Amber and "simulated" means Ollama is not up: run `ollama serve` and reload. And run one
> throwaway question before the audience arrives, so the model is loaded and the first real
> answer is not the slow one.

---

## The frame — what is on screen at all times

**The sidebar, top to bottom.**

| Element | What it is |
|---|---|
| **Parsimony / prompt middleware** | Under it, one sentence saying what the project does: *a small AI model is slow because it has to read every word you send it.* |
| **Five tabs** | Each with a line saying what that screen shows |
| **model** | `qwen2.5:1.5b-instruct` — what is actually answering |
| **encoder** | `ollama:all-minilm` — what is scoring sentences and matching questions |
| **running** | green **real model**, or amber **simulated** |

*Say:* the page always knew which model it was talking to. It now says so, because a results
screen that cannot tell you whether a model was involved is not evidence of anything.

**The header bar** names the tab you are on and repeats the model and encoder, so a
screenshot of any single tab carries its own provenance.

---

## Tab 1 — Architecture · *start here*

A still picture. Nothing runs on this tab, and that is deliberate: it explains before
anything moves.

**"Where it sits"** — three boxes: **You** → **Parsimony** → **The AI model**. Underneath,
the sentence the whole project rests on: *without this in the middle, the model reads every
word you send, one at a time, before it writes anything back. That reading is most of the
wait.*

**"The six parts, in the order they run"** — six colour-coded cards. Each has its code (M1–M6),
a plain name, one sentence of what it does, and a **worked example** under a rule.

| Card | What to point at |
|---|---|
| **M6** Shortcut router | answers arithmetic itself — the AI is never called |
| **M2** Memory of past answers | "ask *Who manages the Porto office?*, then *Who is the site manager in Porto?*" — and the measured figure: **26 of 40 reworded questions reused, none of 40 near-misses** |
| **M3** Conversation trimmer | twenty turns become the four that touch your question |
| **M1** Document compressor | *the big one* — 49 sentences in, 12 out, answer still right |
| **M4** Prompt builder | unchanging parts first, so the model reuses earlier work |
| **M5** Answer-length budget | a one-line answer does not need five hundred words |

**"The two parts that are not steps"** — **M8**, the safety check, and **M7**, the offline
learner. Shown apart because they are not pipeline stages. M8's example is the one to read
aloud: *two sentences give two different dates; the repeat-remover asks to delete one; M8
refuses, because the dates differ. The saving is lost on purpose.*

**"How to check any of this"** — names the tab that proves each claim. Use it to move on.

---

## Tab 2 — Pipeline · *the main demonstration*

One question, start to finish. Top to bottom:

**1. Ask the system something** — the question box, pre-filled so the page is one press from
a run. **Run**, and **Attach the handbook** which loads the 4 KB sample document.

**2. The "Try:" chips** — five one-click demonstrations:

| Chip | Shows |
|---|---|
| compress a document | M1 on the handbook |
| never call the model | M6a answering arithmetic |
| **refuse a saving, on purpose** | **M8 reverting an edit** |
| let it choose its own threshold | the adaptive floor (sends you to Heatmap) |
| reuse an answer | M2, run it twice |

Each chip writes a note under the box explaining what you are about to see.

**3. The result, in words** — appears after a run, above the numbers:

> *The model read **263** tokens instead of **890**: **627** of them never reached it. That is
> about **11.3 s** of reading this laptop did not have to do, timed on this machine.*

*Say:* one sentence, before any figure. If an examiner reads nothing else on this page, that
is the result.

**4. "Answered from memory"** — *appears only when the cache fires.* Both questions side by
side, how alike they are, and underneath what was checked before reusing: **numbers, names,
negations, and what kind of thing is being asked for.** This is the newest work on the page.

**5. The five headline numbers** — tokens in the prompt → sent to the model → removed before
sending → % of the prompt survived (the ring) → seconds not waited.

**6. Prompt bar** — the same thing as a bar: green sent, red removed.

**7. The transport** — ◀ ▶ Play ▶| and a speed menu. **Play steps through the layers one at a
time.** Below it the track: each segment is one layer, its width that layer's measured share
of middleware time. Click a segment to jump to it.

*Say:* this is the control that turns a claim into a demonstration. Every layer has to
explain itself, in order, including the ones that did nothing.

**8. "Your question, moving down the stack"** — the architecture with your question in it.
M1–M6 badges down the left, words moving through channels, **red words are leaving**. Click
any box to inspect it.

**9. "Your text, losing what the question does not need"** *(expandable)* — the handbook with
sentences disappearing as the layer that removed each one runs.

**10. "A closer look at one step"** *(expandable)* — what the layer you clicked actually read
and decided.

**11. "All of it as a table"** *(expandable)* — every layer, its module, time, token delta,
sentences removed, and **why**. The *copy as text* button gives the identical rows that
`parsimony chat` prints, so the screen and the transcript cannot disagree.

**12. "What the AI was actually sent"** *(expandable)* — the assembled prompt, with anything
struck through marked `[removed]` — it never left this machine.

**13. "What this run cost"** and **"What it answered"** — the measured list in the same words
the command line uses, and the model's answer with its token count.

---

## Tab 3 — Heatmap · *why each sentence was kept*

**Load the sample handbook**, ask `When does the office close for Christmas?`, press
**Compress**.

**"What it decided"** — tokens in the document, sent to the model, % removed, sentences kept,
and the two timings: scoring the sentences, and choosing between them.

**"Every sentence, best score first"** — the chart. Bars left of the dashed line were sent,
bars right of it were not. Colours: kept, dropped, **kept below the line** (a sentence the
question named, or one held so a pronoun still refers to something), and **dropped above it**
(the budget ran out).

**Two controls worth demonstrating:**

- **Decide the cut-off by** — *a number I set* or **reading the scores**. Switch to the
  second and it reports something like *floor 0.27 — the scores fall 2.2× after rank 9 of 15*.
  **This takes 8–12 seconds**; the page says so while it works, and dims the old chart.
- **Score the sentences with** — the two encoders. Switch and **the bars move**.

*Say, on the encoder switch:* a threshold is a property of an encoder, not of a technique.
That is why every results table in the paper names which encoder produced it.

**Two dials** — *Cut-off score* and *Most of the document it may send*. Drag either and the
chart follows. Moving them changes this page only; the shipped system keeps its own setting.

**The notes underneath** are the argument — why a rule beats a number, and why the encoder
choice is not cosmetic. Leave them for a reader, do not read them aloud.

**"The document itself"** — every sentence shaded by score, struck through if dropped. Hover
any sentence for the decision behind it.

---

## Tab 4 — A/B · *the honest comparison*

Ask a question, **Run both**. The same question twice against the real model: once with the
system off, once on, **one after the other**, never at once — two generations on one CPU
would measure contention rather than compression.

- **"Words coming out, second by second"** — while a line is flat the AI is still *reading*.
  The sooner a line lifts off, the sooner you saw an answer.
- **"The two runs, side by side"** — the measured table.
- **"The two answers"** — **scroll here first.** The claim is that they say the same thing.

*Say:* this is the tab that could embarrass us, and it is the one we put the answers at the
bottom of on purpose. Read them before the timings.

⏱ Two real generations on a CPU. Allow a couple of minutes.

---

## Tab 5 — Totals · *the safety record*

Everything this session has saved, added up — **tokens removed** and **seconds saved**, in
figures readable from the back of a room.

Underneath, five smaller figures. Two of them are the ones to point at:

| Figure | Why it matters |
|---|---|
| **changes checked for lost meaning** | how often M8 was consulted |
| **changes refused and undone** | **how often it said no** |

*Say:* that second number is the one a compression project is least likely to show you.

---

## A ten-minute route

1. **Architecture** — what it is. *(1 min)*
2. **Pipeline**, chip *never call the model* — 24 → 0 tokens, no AI. *(1 min)*
3. **Pipeline**, chip *refuse a saving, on purpose* — the gate reverting. *(2 min)*
4. **Pipeline**, attach handbook, ask the Tallinn budget question — press **Play**. *(3 min)*
5. **Ask it again reworded** — the memory panel, 90 % match. *(1 min)*
6. **Heatmap** — switch the cut-off to *reading the scores*. *(2 min)*
7. **Totals** — end on *changes refused and undone*. *(30 s)*

Skip A/B unless you have the minutes — it is the slowest thing on the site.
