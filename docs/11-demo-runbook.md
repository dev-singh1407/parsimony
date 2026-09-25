# Demo Runbook

**Parsimony** · mini demo and final review
Every command below was executed and timed on the demo machine before this was written.

---

## 0. Start here — the launcher

Two things break a live demo before it starts: being in the wrong folder, and using the wrong Python. Both
happened in rehearsal, so `demo.ps1` removes them — it sets the folder and the interpreter itself.

**Open PowerShell and paste this one line first. Nothing else works until you do.**

```powershell
cd "C:\Users\dev singh\OneDrive\Desktop\final year\parsimony"
```

Then check everything is ready:

```powershell
.\demo.ps1 check
```

You want five healthy lines: folder, python, ollama installed, server responding, package importable.

**Fifteen minutes before your guide arrives, warm the model.** The first query after a reboot spends about
ten seconds loading 1 GB into RAM; every one after takes about two. Do not let anyone watch the first one.

```powershell
.\demo.ps1 warmup
```

Then maximise the terminal and raise the font to 14pt or more. The tables are drawn to fit the window, and a
table nobody can read from across a desk is worse than no table.

Keep `docs/09-findings.md` open in a second window. It is your answer sheet for any number you are asked
about.

---

## 1. The demo — six acts, fifteen minutes

### Act 0 — A document, live · 3 minutes

**Run this first. It is the strongest thing you have, and the only act where the guide watches the system
work rather than reads about it afterwards.**

```powershell
.\demo.ps1 doc
```

An 817-token staff handbook is attached, and one question is asked about it. The screen fills in as the
request moves:

1. **Each layer appears as it runs**, with its real duration and what it did — the calculator declining, the
   memory with nothing stored yet, the context selector removing 546 tokens in about 50 ms.
2. **The prompt bar shrinks** from 888 tokens to about 320.
3. **A clock runs while the model reads the prompt.** That wait is the cost the layers exist to remove.
4. **The answer streams in**, counted against the budget the answer limiter chose.
5. **The prompt the model actually received** is printed with every removed sentence struck through — six
   document sections, mostly deleted, the answer sentence kept.
6. Then the same question is asked again **from cold with every layer switched off**, and the two runs are
   put side by side.

> "Same question, same model, same laptop, one after the other. Without the layers it reads 907 tokens and
> takes 8.9 seconds before it can start answering. With them it reads 326 and takes 3.0. The answer is the
> same sentence. Neither number is mine — both come from the runtime's own counters."

**If asked why the comparison is run twice rather than reusing the first answer:** *"Because the runtime
caches the prompt it just processed. Re-sending it would have been measured as 64 milliseconds, which would
have flattered us by a factor of fifty. Each side is run from cold, with its own nonce, so neither can reuse
the other's work."*

**Then show which sentences survived and why:**

```powershell
.\demo.ps1 sentences
```

Every sentence of six documents, marked kept or removed, with the answer sentences in bold — and the line
underneath saying what stopped the selection, which names were guaranteed a sentence, and how many sentences
were kept only so a following "It" still refers to something.

---

### Act 1 — What the system does · 2 minutes

```powershell
.\demo.ps1 1
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

```powershell
.\demo.ps1 2
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

```powershell
.\demo.ps1 3
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

> "As far as our survey of 54 papers found, no published semantic cache performs those three."

---

### Act 4 — The science · 3 minutes

```powershell
.\demo.ps1 4
```

Takes about 100 seconds. **Talk while it runs** — this is your architecture slot, not dead air:

> "This is rebuilding every table in my report from raw logs. Eight modules, a 2â´ factorial ablation, 151
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
> overlapped less, so the shortfall shrank and its interval now barely clears zero. We kept the better encoder and
> reported both. Choosing a worse component because it gives a nicer number is the failure mode this whole
> project is written against."

---

## 2. If you have more time

| Command | Time | Shows |
|---|---|---|
| `.\demo.ps1 longctx` | 2 s | Parsimony against truncation, retrieval, stopwords and random at the same budget |
| `.\demo.ps1 followups` | 3 s | Which history strategy keeps the fact the last question needs |
| `.\demo.ps1 ask --file examples/staff-handbook.md` | live | Attach a file and ask freely; type `compare` at any point |
| `.\demo.ps1 learning` | 4 s | "Self-improving" measured: +0.00 pp at 0% traffic repetition, +17.83 pp at 57% |
| `.\demo.ps1 generalise` | 42 s | Does a calibration transfer to another vocabulary? Ratios yes, mechanisms no |
| `.\demo.ps1 gap3` | 9 s | Research gap 3: what compression does to the cache |
| `.\demo.ps1 tokenprobe` | 3 s | When shortening text fails to reduce tokens |
| `.\demo.ps1 corpus` | 1 s | Corpus composition and its freeze hash |
| `.\demo.ps1 demo` | 19 s | The older scripted walkthrough, six sections |
| `.\demo.ps1 web` | live | The visualiser in a browser: sentence heatmap, A/B against the real model, demo counters |
| `.\demo.ps1 proof` | 10 s | A PDF of the compressed prompt, every removal struck through and labelled |
| `.\demo.ps1 floor` | 90 s | The relevance floor set, lowered, and read off the score distribution (ADR-051) |

**The visualiser, if the room has a projector.** `.\demo.ps1 web` opens a local page — no install, no
network. Paste or load a document, ask a question, and every sentence is shaded by the score the encoder
gave it, with the branch that decided it on hover: `[KEEP: ANCHOR]`, `[DROP: FLOOR]`. The A/B tab runs the
same question with the layers off and on against the real model and plots both as they generate; say out
loud that they run **one after the other**, because two generations on one CPU would measure contention
rather than compression, and that the model is warmed first so neither arm pays the weight load. The Demo
tab is two counters large enough to read from the back — and the seconds counter says *estimated* unless
this machine timed the rate itself, which is worth pointing at rather than hiding.

**The strongest ninety seconds on that page.** Open the Heatmap, pick the chip **read the floor off the
scores**, press Compress, then switch *The floor is* from **a constant you choose** to **read off the
scores**. The chart keeps its bars; a green band appears between two of them, labelled with how steeply the
score fell there, and the line moves from 0.15 to 0.35. The page states the difference in a sentence: three
sentences fewer, 56 tokens fewer, with the answer still in the prompt.

> “The threshold here was 0.15. We measured that no single value is right — it suits a question whose
> evidence is concentrated and refuses the second hop of a multi-hop one, which scores low against the
> question by construction. So we stopped setting it. The rule looks at the sorted scores, finds where they
> fall off a cliff, and cuts there; where there is no cliff it uses the constant, because a smooth ramp
> tells you nothing. On 81 held-out items it keeps exactly the same evidence as the constant and sends less
> context than either the constant or the lower one — and on questions the documents cannot answer, a fifth
> less. It is still *off by default*, because the saving is 0.4 points and that is not a reason to move a
> default four ADRs of measurement sit behind.”

**The best question they can ask is why it is off by default, and the answer is the strongest thing here:**
the mechanism wins decisively against *the remedy we ourselves published* — ADR-050 told an operator to
lower the constant, and under the encoder we ship that buys nothing on this corpus and costs 6.8 points of
context. So it is documented as the thing to reach for instead of that remedy, not as a new default.

Run `.\demo.ps1 floor` for the table, and run it twice — `--encoder lexical` gives different numbers,
because the encoder produces the scores the floor is read from. The two encoders disagree about what is
wrong with the constant and agree about the fix, and saying so out loud is worth more than either table.

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
> corpus couldn't reveal, and fixed that too. And when we finally swapped in a neural encoder, it broke the
> safety design — which is the most useful thing we found all month.

**"You upgraded the encoder and it made the system less safe. Explain that."** - *they will see `all-minilm` in the header, so this is the question, not the old one about lexical scoring.*
> We did, and it is installed: MiniLM, 45 MB, served by the same local runtime, no PyTorch. It fixes what a
> lexical encoder cannot see, and it took our false-answer rate from **0% to 17.8%**. The reason is the part
> worth hearing: our design skipped verification when similarity was overwhelming, and the lexical encoder
> simply never scored a trick pair that high. MiniLM scores *"is it safe"* against *"is it NOT safe"* at
> **0.996**. No threshold fixes that — a negation is a smaller edit than a rephrasing in any embedding space,
> so a better space makes it worse. We now verify every hit, which costs microseconds: 0% false answers under
> both encoders, and reuse rises from 26.7% to 37.8%. The thresholds are recalibrated per encoder, because a
> threshold belongs to the encoder it was set against.

**"How much of your speed comes from the layers rather than from tuning?"**
> One honest deduction: while measuring the encoder we found that resolving `localhost` cost **two seconds
> per call** on this machine — it tries IPv6 first, Ollama listens on IPv4, and the attempt has to time out.
> At `127.0.0.1` the same call takes 31 ms. Every wall-clock figure we had measured carried that constant, on
> both sides of every comparison, and none of the prefill numbers could see it because the runtime's counters
> start after the connection. It is fixed, and it is written up rather than quietly removed.

**"Why is prefill the number you keep quoting?"**
> On CPU, reading the prompt is 92 to 99% of the time — about 8.5 milliseconds per input token, measured.
> Writing the answer is almost free by comparison. That is why input tokens are the thing worth cutting, and
> it is the opposite of the GPU-datacentre assumption most of the literature is written under.

**"Why not just keep the last few turns? That is what everyone does."**
> Because on the conversations where it matters it answers **none** of them. We wrote 14 conversations that
> state a fact first -- a server's memory, an allergy, a policy number -- spend four exchanges elsewhere, and
> then ask about that fact. Keeping the last four turns scores **0 of 14**, which is worse than sending no
> history at all. Relevance selection scores **13 of 14** -- one more than sending every turn -- on 17% fewer
> tokens. And where the fact sits inside a long earlier answer, compressing that answer to its relevant
> sentences sends 30% fewer tokens again with no answers lost. `.\demo.ps1 followups` prints it.

**"These layers are basic. Anyone could build this."**
> Anyone can build the obvious version of each one, and we did — `parsimony.eval.naive` holds them as running
> code, not as a description. Then we ran them against ours on 45 held-out questions over six documents each,
> at the same token budget, on the real model. Keeping the last sentences until the budget runs out scores
> **14/45**. Ranking sentences by BM25 and sending the top ones — textbook retrieval — scores **32/45**.
> Deleting stopwords everywhere scores **29/45** while still sending three times as many tokens. Ours scores
> **36/45**, against **40/45** for sending the whole document, and the difference from sending everything is
> not statistically significant. The gap is not in the idea; it is in what the idea needs to survive contact
> with a real question: a sentence beginning "It employs 58 people" has to be found by a question about
> Tallinn, and the sentence that says what "It" is has to come with it. Truncation answers **none** of the
> nine questions like that. We answer all nine. `.\demo.ps1 longctx` prints that table in a second, from the
> recorded run.

**"What's novel here? These techniques all exist."**
> Individually, yes - our survey covers 54 papers on them. What does not exist is a measurement of what
> happens when you **compose** them. Every paper measures one technique against an uncompressed baseline.
> We measured four together and found the savings do not add: the additivity shortfall is 15.13 pp
> [11.15, 18.27]. We also found a direct conflict between two well-cited results: *Lost in the Middle* says
> put relevant content at the prompt edges; prefix caching needs a stable prompt head. Doing the first
> destroys the second - 212 milliseconds against 18,914 for the same content.

**"Isn't your compressor just EXIT? Or Provence?"** - *expect this, and raise it yourself rather than be
caught by it.*
> Architecturally our context tier is the same idea as EXIT, and the survey says so: sentence-level,
> query-aware, threshold, recombined in original order. EXIT published it independently at ACL 2025. We do
> not claim that design.
>
> What differs is the cost. EXIT fine-tunes Gemma-2B with LoRA - about 90 GPU-hours on an A100-80GB - and
> Provence trains a DeBERTa model. Ours is BM25 plus a 45 MB embedder with **no fitted parameters**, which
> is why it runs inside a laptop CPU budget at all. That is an engineering position, not a scientific
> contribution, and we report it as one.
>
> The scientific claim is compositional. EXIT, Provence and LLMLingua are each measured alone. None is
> measured beside a semantic cache, a history manager and an output budgeter paid out of the same tokens -
> and when we did that, the savings stopped adding. None carries a component that can *refuse* a saving
> either: the strand documents compression-induced hallucination as a finding to report, where we built a
> gate against it and measured what it costs us.

**"Your corpus is your own. Did you only do well because you set the exam?"** - *the sharpest question you
will get, and the answer includes a loss. Give the loss first.*
> Fair, and it is why `eval/longbench.py` exists: the shipped configuration, unchanged, on **LongBench
> 2wikimqa**, with LongBench's own prompt and F1 metric, the first 40 items in file order, never sampled.
>
> **It costs us accuracy there.** Full context scores 35.2, we score 29.6 - three items better, seven
> worse, thirty tied. On 40 items that is p = 0.34, so it is not a detectable difference, but the point
> estimate is against us and we report it that way. We are also level with plain truncation.
>
> What we keep is the cost: a fifth of the context, **6.7x less prefill** - 19 minutes against 130 for the
> same forty questions - and the same nine exact answers as full context.
>
> **Then the part that matters.** An accuracy tie does not say who failed, so we measured evidence recall:
> does the answer still appear in what each arm sent? No model, deterministic, and it runs in seconds.
>
> | | evidence sent | accuracy |
> |---|---|---|
> | our corpus, single-hop | Parsimony 98.3%, truncate 27.6% | 40/45 against 16/45 |
> | LongBench, multi-hop | Parsimony 79.5%, truncate 41.0% | F1 29.6 against 29.3 |
>
> The compressor sends the evidence **3.6x as often as truncation on our corpus and 1.9x as often on
> LongBench**. The retention advantage is consistent; what changes is whether it converts. Single-hop
> lookups turn retention into accuracy almost one for one. Multi-hop composition does not, because the
> model scores 34.9 with the answer in a fifth of the context and 36.1 with the whole document in front of
> it - it is failing at composition, not starving for evidence.
>
> So the claim we defend is narrower than "compression keeps accuracy" and stronger than "it ties with
> truncation": at a fifth of the context it retains the evidence nearly twice as reliably as the obvious
> baseline, and whether that becomes an answer depends on a model capable of composing one.

**"So your compressor does not help on the public benchmark?"** - *the follow-up. Do not retreat from it.*
> On the accuracy column, no - and the reason is measurable rather than a guess. The ceiling is 36.1 F1,
> which is what the model scores with the entire document. Nothing a compressor does can be observed above
> a ceiling that low. What is observable there is the cost, and the retention, and both are ours.
>
> The honest version of the limitation is on our side too: our own retention falls from 98.3% to 79.5% on
> multi-hop documents. One item in five loses its answer, because the evidence sits across two passages and
> a sentence-level selector keeping the top-scoring fifth will sometimes keep one hop and drop the other.
> We tried to fix that - second-hop bridging, ADR-048 - measured it on twenty items, found no gain, and
> removed it. The obstacle turned out to be coreference, not retrieval.


**"How do you know your measurements are real?"**
> Because three times they were not, and each was found by measuring rather than reasoning. `localhost`
> resolving to IPv6 first added 2 seconds to every call (ADR-041). The mock provider's 120 ms TTFT
> understated prefill by an order of magnitude, in the direction that flattered us (ADR-034). And the model
> server was silently truncating any prompt over ~2,048 tokens and reporting the truncated count as if it
> were the whole prompt - which the display was reading as *cache reuse*, a saving that never happened
> (ADR-045). We checked every recorded row: the largest prompt ever sent was 869 tokens, so nothing
> published was affected. The guard now refuses the measurement rather than recording it.

**"What's left to do?"** - *the honest answer is a list that has been pruned as well as added to: three
things it used to carry were done and the list had not noticed (findings §12, report §7.9).*
> Six worth saying out loud, and the findings carry a seventh. Each is open because a measurement made it
> open. The factorial sweep under a real provider - token counts do not depend on who answers, so that half
> is done, but the interaction between a shorter prompt and the answer's length is not. Energy, which is
> still arithmetic on a nameplate TDP. A second model in the calibration table: vocabulary transfer is
> measured, a second model has only been run as judge and escalation target. A usable judge, and we know
> why - shown the same answer twice it picks the same slot every time, 50 points of position bias. The
> adaptive floor on multi-hop data, which needs multi-hop data we have not already spent. And the
> near-off-topic case, which our corpus does not contain at all.
>
> Escalation is *not* on that list any more: it was measured and it is a negative result. `llama3.2:3b`
> scored 36/40 against `qwen2.5:1.5b`'s 36/40, item for item identical, for 16% more wall clock and twice
> the memory (ADR-037).

**"How much of this is your own work?"**
> The corpus is ours - 151 conversations, 45 adversarial pairs, 40 gold answers, authored and hashed. The
> architecture, all eight modules, the evaluation harness and over 1,200 tests are ours. The techniques come
> from the literature, and where a design of ours matches a published one we name it. The composition, the
> measurement instrument and every finding are ours.


---

## 4. If something goes wrong

| Symptom | Fix |
|---|---|
| First model query hangs ~10 s | Normal — it is loading. You should have warmed it in §0. |
| `nothing is listening at localhost:11434` | Ollama stopped. Run `ollama list` to restart it, then retry. |
| Tables wrap badly | Terminal too narrow. Maximise, or add `--turns 4` to shorten. |
| A number differs slightly from the report | Say so plainly: wall-clock and middleware-ms vary run to run; token counts do not. Point at the token column. |
| Something errors outright | Fall back to `.\demo.ps1 4`, which regenerates everything, and to `docs/09-findings.md`. |

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
> Everything regenerates from raw logs with one command, and there are over 1,130 tests."
