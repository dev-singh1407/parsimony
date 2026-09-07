# Demo Runbook

**Parsimony** · mini demo and final review
Every command below was executed and timed on the demo machine before this was written.

---

## 0. Fifteen minutes before

Run these. They are not part of the demo — they remove the two things that can make a live demo look broken.

```bash
cd "C:\Users\dev singh\OneDrive\Desktop\final year\parsimony"
```

**1. Wake the model.** The first query after a reboot spends ~10 seconds loading 1 GB into RAM. Every one
after that takes ~2 seconds. Do not let your guide watch the first one.

```bash
python -m parsimony.surfaces.cli.main chat "hello" --provider ollama --no-trace
```

**2. Confirm Ollama is up.** If this prints two models, you are ready.

```bash
ollama list
```

**3. Widen the terminal.** The tables are drawn to fit the window. Maximise it, and set the font large
enough to read from across a desk — 14pt or more. A beautiful table nobody can read is worse than no table.

**4. Have this open in a second tab:** `docs/09-findings.md`. It is your answer sheet for any number you are
asked about.

---

## 1. The demo — five acts, twelve minutes

### Act 1 — What the system does · 2 minutes

```bash
python -m parsimony.surfaces.cli.main compare "How do I reverse a string in Python?" --turns 8
```

**What appears:** the same question sent twice — once with the middleware off, once on — to a real model
running on this laptop, with no internet.

> "Same question, same model, same machine. The only difference is whether our middleware ran. It removed
> about 27% of the input, and the model spent about 20% less time reading the prompt. That last number is
> the server's own measurement, not my stopwatch."

**Point at the `prefill` row specifically.** That is the honest number. If asked why not wall clock: *"Wall
clock on a single request also carries how long the answer happened to be, and that varies by more than the
effect. Prefill is the part input tokens actually control."*

**If output tokens went up:** *"M5 right-sizes the output budget per question type. A code question gets 512
tokens instead of the 256 default — the baseline was being cut off mid-answer. Across the whole corpus it
still reduces every class."*

---

### Act 2 — Why you can trust it · 2 minutes

This is the strongest thing in the demo. Do not rush it.

```bash
python -m parsimony.surfaces.cli.main chat "Explain the deadline. The deadline is 15 March. The deadline is 16 March." --text
```

**What appears:** two panels — **kept (what the model sees)** and **refused (what it would have lost)**.

> "The compressor found two near-identical sentences and tried to delete one. But one carries a date the
> other doesn't. The fidelity gate blocked the edit and the pipeline kept the longer, correct text. You are
> watching a wrong answer being prevented."

Then say the sentence that separates this project from a demo:

> "Every module in this system *proposes* a change. Only the orchestrator commits one, and only after the
> gate has checked it. That is why we can switch any module off independently — which is what makes the
> ablation possible at all."

---

### Act 3 — The finding nobody else has · 3 minutes

```bash
python -m parsimony.surfaces.cli.main calibrate
```

Runs in ~2 seconds.

> "We built 45 adversarial pairs — questions one word apart with opposite answers — plus 45 controls that
> mean the same thing and should match.
>
> The negation pair sits at cosine 0.924. That is **higher than every genuine paraphrase in our set**. The
> thresholds published as safe across the caching literature are 0.85 to 0.92 — so the standard setting
> would accept it and serve the opposite answer.
>
> No threshold fixes this, because the adversarial pairs sit *above* the real paraphrases. What fixes it is
> a verifier: four cheap set comparisons that took our false-answer rate from 26.7% to zero."

If they want the three checks: **operative modifiers** (min/max — changes no number, entity or negation),
**morphological negation** (possible/impossible), and **alphanumeric identifiers** (pandas vs Panda3D).

> "As far as our survey of 52 papers found, no published semantic cache performs those three."

---

### Act 4 — The science · 3 minutes

```bash
python reproduce.py --out figures
```

Takes about 100 seconds. **Talk while it runs** — this is your architecture slot, not dead air:

> "This is rebuilding every table in my report from raw logs. Eight modules, a 2⁴ factorial ablation, 151
> conversations, 263 requests, 17 configurations, bootstrap confidence intervals.
>
> While it runs — the reason it *can* run is that stage order is configuration rather than code. Modules
> propose, the orchestrator commits, every decision is written to a ledger. So switching a module off
> genuinely removes its effect, and the factorial cell means what it says."

When it finishes:

> "Fourteen CSV files and a full report, about a hundred seconds. I am not showing you screenshots — you can
> ask me about any number in this report and I will regenerate it in front of you."

Then the headline:

> "The main result is that **savings do not compound**. Turn on four optimisations individually and their
> savings total more than running all four together. We measured that shortfall with a confidence interval.
> Nobody has published this, because nobody runs these four techniques in one pipeline."

---

### Act 5 — What we got wrong · 2 minutes

Counter-intuitive, and it is what separates a project from a report. Pick **one**:

**Option A — our own explanation was half wrong.**
> "We claimed compression sometimes costs tokens because of two effects in the tokenizer. We tested that
> against a second vocabulary — GPT-2 against Qwen. The reduction percentages transferred almost exactly.
> The *explanation* did not: one of our two mechanisms is specific to Qwen and doesn't exist in GPT-2. We
> found that ourselves and wrote it up."

**Option B — we attacked our own safety component.**
> "We fuzzed our own cache verifier and found we could defeat it with an invisible character — a zero-width
> space inside the word 'not' hides the negation, and the verifier passes a question against its own
> opposite. Seven variants worked. We fixed it, and no result moved."

**Option C — a number moved against us.**
> "We improved our encoder, and it made our headline result weaker — the cache got better, so the modules
> overlapped less, so the shortfall shrank and its interval now touches zero. We kept the better encoder and
> reported both. Choosing a worse component because it gives a nicer number is the failure mode this whole
> project is written against."

---

## 2. If you have more time

| Command | Time | Shows |
|---|---|---|
| `... main learning` | 4 s | "Self-improving" measured: +0.00 pp at 0% traffic repetition, +17.83 pp at 57% |
| `... main generalise` | 42 s | Does a calibration transfer to another vocabulary? Ratios yes, mechanisms no |
| `... main gap3` | 9 s | Research gap 3: what compression does to the cache |
| `... main tokenprobe` | 3 s | When shortening text fails to reduce tokens |
| `... main corpus` | 1 s | Corpus composition and its freeze hash |
| `... main demo` | 19 s | The older scripted walkthrough, six sections |

**Do not run live:** `latency` (3–4 min) and `judge` (5–10 min). Quote their numbers from §8 of the findings
instead, or run them beforehand and show the scrollback.

---

## 3. The questions you will be asked

**"Did you test on a real model, or just simulate?"**
> Real. `qwen2.5:1.5b-instruct`, 4-bit, running on this laptop's CPU, offline. On our 40 gold questions it
> scores 92.5% without the pipeline and 97.5% with it — and **not one answer the baseline got right was
> lost**. Everything before that was measured against a deterministic stand-in, and every log row records
> which one produced it, so the two can never be confused.

**"How do you know the cache is safe?"**
> 45 adversarial pairs, false-answer rate from 26.7% to 0%. Then we fuzzed it and found a bypass our own
> corpus couldn't reveal, and fixed that too.

**"Why is prefill the number you keep quoting?"**
> On CPU, reading the prompt is 92 to 99% of the time — about 8.5 milliseconds per input token, measured.
> Writing the answer is almost free by comparison. That is why input tokens are the thing worth cutting, and
> it is the opposite of the GPU-datacentre assumption most of the literature is written under.

**"What's novel here? These techniques all exist."**
> Individually, yes — and our survey covers 52 papers on them. What doesn't exist is a measurement of what
> happens when you **compose** them. Every paper measures one technique against an uncompressed baseline.
> We measured all four together and found the savings don't add. We also found a direct conflict between two
> well-cited results: *Lost in the Middle* says put relevant content at the prompt edges; prefix caching
> needs a stable prompt head. Doing the first destroys the second — 212 milliseconds against 18,914 for the
> same content.

**"What's left to do?"**
> Four things, in priority order, in the roadmap. The largest is replacing our lexical encoder with a neural
> one — we've quantified exactly what that would buy, so it's a costed decision rather than a wish.

**"How much of this is your own work?"**
> The corpus is ours — 151 conversations, 45 adversarial pairs, 40 gold answers, authored and hashed. The
> architecture, all eight modules, the evaluation harness and 700 tests are ours. The techniques are from the
> literature; the composition, the measurement instrument, and every finding are ours.

---

## 4. If something goes wrong

| Symptom | Fix |
|---|---|
| First model query hangs ~10 s | Normal — it is loading. You should have warmed it in §0. |
| `nothing is listening at localhost:11434` | Ollama stopped. Run `ollama list` to restart it, then retry. |
| Tables wrap badly | Terminal too narrow. Maximise, or add `--turns 4` to shorten. |
| A number differs slightly from the report | Say so plainly: wall-clock and middleware-ms vary run to run; token counts do not. Point at the token column. |
| Something errors outright | Fall back to `python reproduce.py --out figures`, which regenerates everything, and to `docs/09-findings.md`. |

**The general rule:** if a number surprises you live, say *"that's run-to-run variance, the token counts are
deterministic"* and move to the token column. Never guess at an explanation in the room.

---

## 5. If you have sixty seconds

> "It's a middleware layer that sits between an application and a small language model running on a normal
> laptop — no GPU, no internet, no API cost. Eight modules that each cut token usage a different way, and a
> gate that blocks any edit which would change the meaning.
>
> It cuts about a third of the tokens with no loss of answer accuracy. But the real result is that the
> savings **don't add up** — techniques that each save 10% don't save 40% together, and we measured by how
> much they fail to. Nobody had, because nobody runs them in one pipeline.
>
> Everything regenerates from raw logs with one command, and there are 700 tests."
