"""Build the BCSE497J Project-I report from the school's template.

The template is a specimen, not a shell: its body is placeholders and formatting
notes -- "<Title of Project>", "(Times New Roman 16, Bold, Upper Case)",
"/****** Sample*******/". Appending to it leaves page one reading
"<Title of Project>". So this opens the template for its PAGE SETUP and styles,
clears the body, and rewrites the report to the section tree the template
specifies, at the type sizes it specifies:

    section heading      Times New Roman 14, bold, upper case, spacing 1.5
    sub-heading          Times New Roman 13, bold, title case, spacing 1.5
    sub-sub-heading      Times New Roman 12, bold, spacing 1.5
    body                 Times New Roman 12, spacing 1.15

Every number in the report is read from figures/*.csv, so the document cannot
drift from the measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt

FONT = "Times New Roman"
ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "figures" / "diagrams"

TITLE = "TOKEN-EFFICIENT LLM INTERACTION ON CPU-ONLY HARDWARE"
SUBTITLE = ("A Stacked, Self-Improving Optimisation Layer "
            "for Small Language Models")
GUIDE = "Dr Sathya K"
DESIGNATION = "Associate Professor"
SCHOOL = "School of Computer Science and Engineering"
TEAM = [("Alok Singh", "23BCI0158"),        # sorted on register number
        ("Arrsh Tripathi", "23BCI0191"),
        ("Dev Singh", "23BCE0794")]

ABSTRACT_MAX_WORDS = 300          # the template: "not exceeding 300 words"

ABSTRACT = [
    "A language model charges for every token it reads and every token it "
    "writes, and on a CPU-only laptop it is slow: this project measures 91.7 "
    "to 98.8 percent of that time as the model reading the prompt rather than "
    "writing the answer. A literature reduces this cost through prompt "
    "compression, semantic caching, key-value cache reuse, routing and output "
    "budgeting, but every technique is published in isolation, on GPU, against "
    "an uncompressed baseline. Nothing establishes what happens when they are "
    "composed, or whether their published operating points survive a descent "
    "to a 1.5-billion-parameter model on consumer hardware.",

    "This project surveys forty papers, extracts the limitation of each for a "
    "CPU-only single-user deployment, and derives six research gaps. It then "
    "builds Parsimony, a middleware layer of seven optimisation modules and an "
    "always-on fidelity gate, in which every module proposes an edit and one "
    "orchestrator commits it only after an invariant check. Because each "
    "module is independently switchable, the system is a factorial experiment "
    "as well as a pipeline.",

    "Across 151 conversations and 263 requests the full stack removes 33.9 "
    "percent of tokens for 4.06 milliseconds of middleware overhead, with zero "
    "quality regressions on 40 gold items. Three findings are new. Savings are "
    "not additive: modules whose individual reductions sum to 29.0 percentage "
    "points deliver 27.4 together, the shortfall concentrated in one negative "
    "interaction. Token count is an unreliable proxy for cost: two prompts 0.5 "
    "percent apart in length differ roughly 89-fold in steady-state cost. And "
    "no similarity threshold makes cached-answer reuse safe, the false-hit "
    "rate staying flat at 51.1 percent across every usable threshold, whereas "
    "four non-neural set comparisons reduce it to zero.",

    "Keywords - Prompt Compression, Semantic Caching, KV-Cache Reuse, Small "
    "Language Models, CPU Inference, Factorial Ablation, Fidelity Verification",
]

# Each node is (kind, payload). Kinds: h1 h2 h3 p b img tbl
CONTENT = [
    ("h1", "1. INTRODUCTION"),
    ("h2", "1.1 Background"),
    ("p", "Large language models are increasingly run locally rather than "
          "called over a network. Quantised models of one to three billion "
          "parameters now fit in the memory of an ordinary laptop and answer "
          "usefully without an API key, a subscription or an internet "
          "connection. This shift changes what 'efficiency' means. On a "
          "metered cloud API the quantity that matters is the number of tokens "
          "billed. On a CPU with no graphics card the quantity that matters is "
          "time, and time is dominated by a different part of the request than "
          "the billing model would suggest."),
    ("p", "A body of research addresses token cost: prompt compression removes "
          "words the model does not need, semantic caching reuses an earlier "
          "answer to a similar question, key-value cache reuse avoids "
          "re-reading an unchanged prompt prefix, routing sends easy questions "
          "to a cheaper destination, and output budgeting stops the model "
          "rambling. Each of these has been published separately, evaluated "
          "against an unoptimised baseline, and tuned on GPT-class models "
          "running on datacentre hardware."),
    ("p", "This project asks what happens when those techniques are switched "
          "on together, on the hardware and at the model scale the literature "
          "has not evaluated, and whether the settings published as safe "
          "remain safe there."),

    ("h2", "1.2 Motivation"),
    ("p", "Three observations motivated the work. First, a practical one: a "
          "student laptop can run a small model, but a multi-turn "
          "conversation becomes painfully slow because the entire history is "
          "re-read on every turn. The cost grows with the conversation even "
          "though the new question is short."),
    ("p", "Second, a scientific one. Every paper in this area reports its own "
          "technique against a plain baseline. If four techniques each save "
          "ten percent, do they save forty percent together? Nobody has "
          "reported the answer, because nobody has run them in one pipeline. "
          "The question is not rhetorical: a compressor that rewrites a query "
          "changes what the cache sees, and a trimmer that drops turns changes "
          "what the compressor has left to remove."),
    ("p", "Third, a safety one. Semantic caches decide whether to reuse an old "
          "answer by measuring similarity. The pair 'Is it safe to mix bleach "
          "and vinegar?' and 'Is it not safe to mix bleach and vinegar?' "
          "differs by one word. If a cache confuses them it does not fail "
          "loudly; it confidently returns the opposite of the truth. Whether "
          "the thresholds published as safe actually prevent this at small "
          "model scale had not been tested."),

    ("h2", "1.3 Scope of the Project"),
    ("p", "In scope: a middleware layer that sits between an application and a "
          "locally hosted small language model, containing seven optimisation "
          "modules and one always-on fidelity gate; a token ledger recording "
          "every decision; an evaluation harness able to run the full "
          "factorial of module combinations; and a corpus of 151 "
          "conversations, 45 adversarial question pairs with 45 matched "
          "controls, and 40 gold-answer items."),
    ("p", "Also in scope: five experiments measuring token reduction, the "
          "prefill and decode split of CPU inference, threshold sensitivity of "
          "cache reuse, answer quality against the real model, and whether a "
          "calibration transfers to a second tokenizer vocabulary and to "
          "unseen conversations."),
    ("p", "Out of scope: training or fine-tuning any model; any technique "
          "requiring access to model internals such as attention weights; GPU "
          "execution; any paid API, dataset or service; and a graphical user "
          "interface. The system is exposed through a command-line surface "
          "only. The escalation router to a larger model was implemented and "
          "calibrated but is disabled by default, because measurement showed a "
          "three-billion-parameter model scoring item-for-item identically to "
          "the 1.5-billion-parameter model on the gold subset."),

    ("h1", "2. PROJECT DESCRIPTION AND GOALS"),
    ("h2", "2.1 Literature Review"),
    ("p", "Forty papers were reviewed across six strands. For each, the "
          "limitation recorded is a limitation with respect to this project's "
          "setting -- a CPU-only, single-user, multi-turn deployment with "
          "several techniques composed -- and not a criticism of the paper on "
          "its own terms."),
    ("b", "Prompt compression (15 papers). Selective Context [1] and LLMLingua "
          "[2] score tokens by informativeness and drop the least useful; "
          "LongLLMLingua [3] and LLMLingua-2 [4] refine this for long contexts "
          "and task-agnostic use. All require either a second model in the "
          "loop or fine-tuning, are measured on GPU, and are never composed "
          "with a cache. An empirical study [12] names two failure modes -- "
          "altered-semantic and information-loss hallucination -- and measures "
          "how often they occur, but proposes no run-time guard against them."),
    ("b", "Semantic caching (7 papers). GPTCache [16] established the design "
          "every subsequent cache follows: embed the query, compare to stored "
          "queries, serve above a similarity threshold. MeanCache [18] adds a "
          "context chain and on-device privacy; ContextCache [19] is closest "
          "to our multi-turn setting. None reports a false-hit rate on "
          "adversarially constructed near-duplicates. SAFE-CACHE [22] "
          "concludes vector similarity is insufficient but answers with a "
          "cluster-centroid redesign."),
    ("b", "KV cache and prefix reuse (5 papers). PagedAttention [23] and vLLM "
          "automatic prefix caching [24] establish that an identical prompt "
          "prefix can be skipped entirely. The requirement is stated; the cost "
          "to an application of violating it is never measured."),
    ("b", "Positional effects (4 papers). Lost in the Middle [28] shows "
          "accuracy peaks when relevant content sits at the beginning or end "
          "of a prompt, which motivates reordering by relevance. Reordering "
          "destroys prefix stability. Neither strand cites the other."),
    ("b", "Routing and cascades (6 papers). FrugalGPT [32] and RouteLLM [33] "
          "cut cost by sending easy queries to cheaper models, assuming a "
          "meaningfully better expensive tier exists. Output length control "
          "(3 papers) [38, 39] requires fine-tuning or control-token training."),
    ("p", "Five limitations recur and together define the space this project "
          "occupies:"),
    ("tbl", (["#", "Recurring limitation", "Papers"], [
        ["L1", "Evaluated in isolation against an uncompressed baseline; never composed with another technique",
         "[1]-[11], [16]-[20], [38]-[40]"],
        ["L2", "Assumes GPU or datacentre serving, where the cost structure is the inverse of a CPU laptop",
         "[2], [3], [23]-[27], [32]-[35]"],
        ["L3", "Failure modes diagnosed but not prevented at run time",
         "[12]-[14], [21], [22]"],
        ["L4", "Operating points quoted as universal, calibrated only on GPT-class models",
         "[2], [4], [16], [17], [19]"],
        ["L5", "Requires training, fine-tuning, or access to model internals",
         "[5], [7], [8], [38], [39]"],
    ], "Table 2.1  Limitations recurring across the surveyed literature")),

    ("h2", "2.2 Research Gap"),
    ("p", "Six gaps follow from the limitation column above."),
    ("b", "Gap 1 -- Compositional effects are unmeasured. Every technique is "
          "evaluated alone. Whether savings add when composed is unknown, "
          "because no study runs them in one pipeline."),
    ("b", "Gap 2 -- The prefill and decode split is uncharacterised for "
          "CPU-only single-user inference. The serving literature measures GPU "
          "datacentre throughput. Which half of a request a laptop actually "
          "pays for is not established, and no optimisation stack contains an "
          "output-side module at all."),
    ("b", "Gap 3 -- The compression and cache interaction is unmeasured. "
          "Compression strips the modifiers that distinguish similar "
          "questions, which should make false hits worse; it also normalises "
          "phrasing, which should make true hits better. Neither direction has "
          "been measured."),
    ("b", "Gap 4 -- Token savings and prefix-cache survival have never been "
          "measured together. Token reduction is treated as a proxy for cost, "
          "and on local hardware that proxy can invert."),
    ("b", "Gap 5 -- Published cache thresholds are not adversarially validated "
          "at small model scale. The 0.85 to 0.92 range is quoted as safe. "
          "Whether near-identical, opposite-meaning questions sit above or "
          "below it is not reported."),
    ("b", "Gap 6 -- Every cache in the literature starts empty. Results are "
          "reported on a warm cache. A single user's own conversation history "
          "is a large, free, perfectly on-distribution corpus that no "
          "optimisation system consumes."),

    ("h2", "2.3 Objectives"),
    ("p", "The objectives below are stated so that each is verifiable by a "
          "specific experiment rather than by inspection."),
    ("tbl", (["#", "Objective", "Verified by", "Result"], [
        ["O1", "Compose eight techniques in one pipeline with every module independently switchable",
         "2^4 factorial sweep, 17 configurations", "Achieved"],
        ["O2", "Establish the CPU prefill and decode cost structure and convert tokens to milliseconds",
         "Server-reported prefill and decode counters", "Achieved"],
        ["O3", "Drive the false-answer rate of cache reuse to zero without a second model",
         "45 adversarial pairs, 45 controls, 9 thresholds", "Achieved (0/45)"],
        ["O4", "Ensure no optimisation turns a correct answer into an incorrect one",
         "40 gold items, paired per item, real model", "Achieved (0 regressions)"],
        ["O5", "Keep total middleware overhead below 120 ms per request",
         "Ledger timings on all 17 configurations", "Achieved (4.06 ms)"],
        ["O6", "Make every result reproducible from raw logs by one command",
         "reproduce.py, approximately 100 seconds", "Achieved"],
    ], "Table 2.2  Project objectives and their verification")),

    ("h2", "2.4 Problem Statement"),
    ("p", "Build a middleware layer that measurably reduces the token and time "
          "cost of interacting with a small language model on CPU-only "
          "consumer hardware, without changing the meaning of any request, and "
          "instrument it so that the contribution of every individual module, "
          "and every interaction between modules, is separately measurable and "
          "independently reproducible."),
    ("p", "The emphasis on meaning is deliberate. A system that saves tokens by "
          "quietly damaging requests is not a contribution; the safety "
          "property is the constraint under which every optimisation must "
          "operate. This is why the fidelity gate is always on and is not one "
          "of the switchable modules."),
    ("p", "The deliverable is correspondingly not a single headline percentage "
          "but a calibrated operating curve: for a given model, quantisation "
          "and query class, which combination of modules should be enabled and "
          "at what setting. The measured spread across query classes -- from "
          "20.7 percent on summarisation to 69.3 percent on arithmetic -- is "
          "the empirical justification for stating the deliverable this way."),

    ("h2", "2.5 Project Plan"),
    ("p", "The schedule below organises the work into nine phases across the "
          "sixteen weeks of Project-I, with the dependencies that constrain "
          "their order. Phases 2 and 3 overlap because the contracts layer was "
          "stable before the modules that consume it were finished; phases 3 "
          "and 4 overlap because the corpus could be authored while the "
          "modules were being built. Phase 6 could not begin until both the "
          "modules and the harness existed, which is the critical path. The "
          "final phase is carried into Project-II."),
    ("img", (DIAG / "gantt.png", 6.5,
             "Figure 2.1  Gantt chart of the Project-I schedule.")),
    ("tbl", (["Phase", "Work", "Status"], [
        ["1", "Literature survey of 40 papers; extraction of limitations; derivation of six research gaps", "Complete"],
        ["2", "Contracts (L0) and infrastructure (L1): tokeniser, encoders, providers, ledger schema", "Complete"],
        ["3", "Modules M1 to M8; orchestrator with propose-and-commit discipline; stage-order validator", "Complete"],
        ["4", "Corpus authoring: 151 conversations, 45 adversarial pairs, 45 controls, 40 gold items", "Complete"],
        ["5", "Evaluation harness: factorial sweep, bootstrap statistics, effect sizes, calibration", "Complete"],
        ["6", "Experiments E1 to E5; analysis; threats to validity", "Complete"],
        ["7", "Attach the real model via Ollama; re-measure latency and quality against it", "Complete"],
        ["8", "Reporting: project report, research paper, reproducibility package", "Complete"],
        ["9", "Instrumented energy measurement; per-configuration quality on the real model", "Project-II"],
    ], "Table 2.3  Project plan and current status")),

    ("h1", "3. TECHNICAL SPECIFICATION"),
    ("h2", "3.1 Requirements"),
    ("h3", "3.1.1  Functional"),
    ("p", "The system must accept a user query together with the running "
          "conversation history, and return an answer together with a complete "
          "record of how it was produced."),
    ("tbl", (["ID", "Functional requirement", "Realised by"], [
        ["FR1", "Recognise queries answerable without the model and compute the answer exactly", "M6a deterministic router"],
        ["FR2", "Check whether a semantically equivalent question has been answered before, and defend that decision against near-duplicates", "M2 cache with three-zone policy and verifier"],
        ["FR3", "Select which historical turns remain relevant and decide where to place them", "M3a selector, M3b arranger"],
        ["FR4", "Compress the remaining prompt, measuring the result in real tokenizer tokens rather than words", "M1 tiers 1 to 3"],
        ["FR5", "Refuse any transformation that drops a number, named entity, unit, negation or operative modifier", "M8 fidelity gate"],
        ["FR6", "Assemble the final prompt so that its invariant prefix stays byte-stable across turns", "M4 assembler"],
        ["FR7", "Classify the query, set an output budget appropriate to it, and stop generation once the answer is complete", "M5 budgeter with streaming early stop"],
        ["FR8", "Log every decision with its token and time cost, including decisions to do nothing", "Ledger v1, one row per stage"],
        ["FR9", "Mine past conversations offline to warm-start the cache and redundancy lexicon", "M7 policy learner"],
        ["FR10", "Enable or disable any optimisation module independently at run time", "Orchestrator, stage order as configuration"],
    ], "Table 3.1  Functional requirements")),

    ("h3", "3.1.2  Non-Functional"),
    ("tbl", (["ID", "Non-functional requirement", "Target", "Measured"], [
        ["NFR1", "Middleware overhead per request, model time excluded", "Under 120 ms", "4.06 ms mean, 9.49 ms p95"],
        ["NFR2", "No degradation of answer correctness", "Zero regressions", "0 of 40 gold items"],
        ["NFR3", "False-answer rate when reusing a cached answer", "Below 2 percent", "0.0 percent (0/45)"],
        ["NFR4", "Operate with no GPU, no API key and no paid dataset", "Mandatory", "Satisfied"],
        ["NFR5", "Operate fully offline after model download", "Mandatory", "Satisfied"],
        ["NFR6", "Every reported result reproducible by one command", "Mandatory", "reproduce.py, ~100 s"],
        ["NFR7", "Auditability: every stage writes a ledger row whether or not it acted", "Mandatory", "Satisfied"],
        ["NFR8", "Architectural layering enforced automatically", "Mandatory", "test_architecture.py, 719 tests pass"],
    ], "Table 3.2  Non-functional requirements and measured outcomes")),

    ("h2", "3.2 Feasibility Study"),
    ("h3", "3.2.1  Technical Feasibility"),
    ("p", "The project is technically feasible and has been fully implemented. "
          "The heaviest dependency is the model runtime, and Ollama serves a "
          "4-bit quantised 1.5-billion-parameter model within the memory of a "
          "16 GB laptop. All remaining components are pure Python: the "
          "tokeniser is the model's own vocabulary loaded through the "
          "tokenizers library, the encoders and statistics use numpy, and the "
          "provider layer speaks to Ollama over the standard library alone. No "
          "deep-learning framework is installed and no model is trained. The "
          "principal technical risk identified at the outset -- that "
          "middleware overhead would consume the savings -- was measured and "
          "did not materialise: overhead is roughly 0.24 percent of the time "
          "the same request spends being read by the model."),
    ("h3", "3.2.2  Economic Feasibility"),
    ("p", "The total monetary cost of the project is zero. Every dependency is "
          "open-source and every model weight is openly licensed. There is no "
          "API subscription, no cloud compute, no paid dataset and no "
          "commercial tool in the pipeline. The hardware used is a personal "
          "laptop already owned by a team member. Because the system runs "
          "locally, the operating cost after deployment is also zero, which is "
          "precisely the deployment scenario the project targets and which the "
          "cost-focused cloud literature does not address."),
    ("h3", "3.2.3  Social Feasibility"),
    ("p", "The system is designed for users who cannot or prefer not to send "
          "their conversations to a third-party service: students without a "
          "subscription, users on unreliable connectivity, and anyone handling "
          "private material. Because everything runs on the user's machine, no "
          "conversation leaves the device, and the policy learner that mines "
          "past chats to warm-start the cache does so entirely locally. The "
          "fidelity gate addresses the principal social risk of this class of "
          "system, which is silent corruption: an optimisation that changes "
          "what the user asked without telling them. Every decision is written "
          "to a ledger the user can inspect."),

    ("h2", "3.3 System Specification"),
    ("h3", "3.3.1  Hardware Specification"),
    ("tbl", (["Component", "Specification used", "Minimum requirement"], [
        ["Processor", "AMD Ryzen 7 5800HS, 8 cores", "Any modern x86-64 CPU, 4 cores"],
        ["Memory", "16 GB RAM", "8 GB RAM"],
        ["Graphics", "None used", "None required"],
        ["Storage", "About 2 GB for model weights and corpus", "2 GB free"],
        ["Network", "Required once, to download model weights", "Offline thereafter"],
    ], "Table 3.3  Hardware specification")),
    ("h3", "3.3.2  Software Specification"),
    ("tbl", (["Component", "Choice", "Purpose"], [
        ["Operating system", "Windows 11 (platform independent)", "Host"],
        ["Language", "Python 3.11", "Implementation"],
        ["Model runtime", "Ollama", "Serves the model locally on CPU"],
        ["Model", "qwen2.5:1.5b-instruct, Q4_K_M quantisation", "The system under optimisation"],
        ["Tokenisation", "tokenizers (Qwen2.5 vocabulary, 151,665 entries)", "Exact token counts"],
        ["Numerics", "numpy", "Embeddings and statistics"],
        ["Interface", "typer, rich", "Command-line surface"],
        ["Testing", "pytest", "719 automated tests"],
        ["Deliberately absent", "PyTorch, any GPU dependency, any paid API", "Keeps the target hardware honest"],
    ], "Table 3.4  Software specification")),

    ("h1", "4. DESIGN APPROACH AND DETAILS"),
    ("h2", "4.1 System Architecture"),
    ("p", "The governing design decision is that every module proposes a "
          "change and a single orchestrator commits it. A proposal is a "
          "description of an edit, not the edit itself. Three properties "
          "follow that no individual module has to implement. Switching a "
          "module off genuinely removes its effect, so a factorial cell means "
          "what it claims. Every edit passes one safety gate, so fidelity is "
          "not re-implemented seven times. And reverting is free, because a "
          "refused proposal is simply never assigned."),
    ("p", "The system is built in six layers, L0 to L5. A layer may depend only "
          "on layers below it, and this is enforced by an automated test "
          "rather than by convention. Stage order is configuration rather than "
          "code, validated against a reads-and-writes dependency graph before "
          "the pipeline runs; research Gap 3 is answerable only because the "
          "cache lookup can be moved before or after the compressor without "
          "editing the pipeline."),
    ("img", (DIAG / "architecture.png", 6.4,
             "Figure 4.1  Layered system architecture. Modules propose, the "
             "orchestrator commits, and the fidelity gate inspects every "
             "proposal before it is applied.")),
    ("b", "Cross-cutting components. The token ledger writes one row per stage "
          "per request -- module, outcome, tokens before and after, duration, "
          "rationale, gate event and the provider's content digest. A stage "
          "that did nothing still writes a row, because an invisible stage is "
          "an unauditable one and the entire evaluation depends on the trace "
          "being complete. The provider layer refuses rather than falling back "
          "when the real model is requested and cannot be reached, since a run "
          "that believes it measured a real model and actually measured a "
          "stand-in is the worst failure available here."),

    ("h2", "4.2 Design"),
    ("h3", "4.2.1  Data Flow Diagram"),
    ("p", "The context diagram shows Parsimony as a single process between the "
          "user and the local model. The level-1 decomposition shows the eight "
          "processes it contains and the three data stores they use. Two "
          "properties of the design are visible in the diagram. Processes 1.0 "
          "and 2.0 may terminate the flow early -- a deterministic answer or a "
          "verified cache hit returns to the user without the model being "
          "called at all -- and process 8.0, the fidelity gate, receives a "
          "proposal from every other process and can refuse it."),
    ("img", (DIAG / "dfd.png", 6.4,
             "Figure 4.2  Data flow diagram, level 0 and level 1.")),
    ("h3", "4.2.2  Use Case Diagram"),
    ("p", "Three actors interact with the system. The end user asks questions "
          "and receives answers. The researcher runs sweeps, calibrates "
          "thresholds and inspects the ledger; this actor exists because the "
          "system is a measurement instrument as much as it is middleware. The "
          "local model is an external actor invoked only when the request "
          "survives the early-exit paths. UC1 and UC2 both include UC8, the "
          "fidelity check: no answer, generated or reused, reaches the user "
          "without passing it."),
    ("img", (DIAG / "usecase.png", 6.1,
             "Figure 4.3  Use case diagram.")),
    ("h3", "4.2.3  Class Diagram"),
    ("p", "The core type model is defined in layer L0 and is deliberately "
          "small. A Request carries the current query and the list of prior "
          "Turn objects. Every module implements a single protocol method that "
          "receives a Request and returns a Proposal, which is one of three "
          "kinds: a ContextPatch describing an edit, a ShortCircuit carrying "
          "an answer that ends the pipeline, or a NoOp. The orchestrator holds "
          "the list of stages, the gate and the ledger, and is the only object "
          "permitted to mutate a Request. Each committed or refused proposal "
          "produces one LedgerRow. This is what makes the ablation sound: "
          "because modules cannot mutate anything, disabling one removes its "
          "effect completely and no residue can leak into another cell."),
    ("tbl", (["Class", "Key attributes", "Responsibility"], [
        ["Request", "query, history, metadata", "Immutable input to every stage"],
        ["Turn", "role, content, token count", "One message in the conversation"],
        ["Proposal", "kind, payload, rationale", "What a module wants to change; abstract"],
        ["ContextPatch", "before, after, transform kind", "A proposed edit to the prompt"],
        ["ShortCircuit", "answer, source", "Ends the pipeline without calling the model"],
        ["NoOp", "reason", "Explicit statement that nothing was proposed"],
        ["Orchestrator", "stages, gate, ledger", "Applies stages in order; commits or refuses"],
        ["FidelityGate", "invariant extractors", "Accepts or refuses a proposal"],
        ["LedgerRow", "module, outcome, tokens, duration, digest", "One auditable record per stage"],
        ["PolicyBundle", "cache seeds, lexicon, templates", "Mined offline by M7 to warm-start modules"],
    ], "Table 4.1  Core classes and responsibilities")),
    ("h3", "4.2.4  Sequence Diagram"),
    ("p", "A request proceeds as follows. The surface constructs a Request from "
          "the query and history and hands it to the orchestrator. The "
          "orchestrator calls each stage in the configured order. Each stage "
          "returns a Proposal. For every proposal other than a NoOp, the "
          "orchestrator calls the fidelity gate, which extracts invariants "
          "from the text before and after the proposed edit and either accepts "
          "or refuses. An accepted ContextPatch is applied to the Request; a "
          "refused one is discarded and the Request is unchanged. A "
          "ShortCircuit from M6a or M2 ends the sequence immediately and the "
          "answer is returned without any call to the model. If the sequence "
          "reaches the end, the assembled prompt and the output budget are "
          "sent to the provider, the response is streamed back subject to the "
          "early-stop rule, and the answer is returned. At every step, "
          "including refusals and no-ops, a LedgerRow is written."),
    ("img", (DIAG / "sequence.png", 6.5,
             "Figure 4.4  Sequence diagram for a single request, showing the "
             "propose-check-commit cycle, both early-exit paths, and the "
             "ledger write that follows every stage.")),
]

REFERENCES = [
    "Y. Li, B. Dong, C. Lin and F. Guerin, \"Compressing context to enhance inference efficiency of large language models,\" in Proc. EMNLP, 2023.",
    "H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang and L. Qiu, \"LLMLingua: Compressing prompts for accelerated inference of large language models,\" in Proc. EMNLP, 2023.",
    "H. Jiang et al., \"LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression,\" arXiv:2310.06839, 2023.",
    "\"LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression,\" arXiv:2403.12968, 2024.",
    "\"Efficient prompt compression with evaluator heads for long-context transformer inference,\" arXiv:2501.12959, 2025.",
    "\"SCOPE: A generative approach for LLM prompt compression,\" arXiv:2508.15813, 2025.",
    "\"Long context in-context compression by getting to the gist of gisting,\" arXiv:2504.08934, 2025.",
    "\"Compressing lengthy context with UltraGist,\" arXiv:2405.16635, 2024.",
    "\"ATACompressor: Adaptive task-aware compression for efficient long-context processing in LLMs,\" arXiv:2602.03226, 2026.",
    "\"SARA: Selective and adaptive retrieval-augmented generation with context compression,\" arXiv:2507.05633, 2025.",
    "\"PCToolkit: A unified plug-and-play prompt compression toolkit of large language models,\" arXiv:2403.17411, 2024.",
    "\"An empirical study on prompt compression for large language models,\" in Proc. ICLR, 2025.",
    "\"Understanding and improving information preservation in prompt compression for LLMs,\" arXiv:2503.19114, 2025.",
    "\"When summaries distort decisions: Information fidelity in LLM-compressed financial analysis,\" arXiv:2606.29251, 2026.",
    "\"The compression paradox in LLM inference: Provider-dependent energy effects of prompt compression,\" arXiv:2603.23528, 2026.",
    "F. Bang, \"GPTCache: An open-source semantic cache for LLM applications,\" in Proc. NLP-OSS at EMNLP, 2023.",
    "\"GPT semantic cache: Reducing LLM costs and latency via semantic embedding caching,\" arXiv:2411.05276, 2024.",
    "\"MeanCache: User-centric semantic caching for LLM web services,\" arXiv:2403.02694, 2024.",
    "\"ContextCache: Context-aware semantic cache for multi-turn queries in large language models,\" arXiv:2506.22791, 2025.",
    "\"A generative caching system for large language models,\" arXiv:2503.17603, 2025.",
    "\"From similarity to vulnerability: Key collision attack on LLM semantic caching,\" arXiv:2601.23088, 2026.",
    "\"Enhancing adversarial resilience in semantic caching for secure retrieval-augmented generation systems,\" Scientific Reports, 2026.",
    "W. Kwon et al., \"Efficient memory management for large language model serving with PagedAttention,\" in Proc. SOSP, 2023.",
    "vLLM Project, \"Automatic prefix caching,\" technical documentation, 2025.",
    "\"ChunkAttention: Efficient self-attention with prefix-aware KV cache and two-phase partition,\" arXiv:2402.15220, 2024.",
    "\"Sparse prefix caching for hybrid and recurrent LLM serving,\" arXiv:2605.05219, 2026.",
    "\"Multi-segment attention: Efficient KV-cache management for faster LLM serving,\" arXiv:2606.02964, 2026.",
    "N. F. Liu et al., \"Lost in the middle: How language models use long contexts,\" Transactions of the ACL, vol. 12, pp. 157-173, 2024.",
    "\"Adaptive focus memory for language models,\" arXiv:2511.12712, 2025.",
    "\"A survey on multi-turn interaction capabilities of large language models,\" arXiv:2501.09959, 2025.",
    "\"From human memory to AI memory: A survey on memory mechanisms in the era of LLMs,\" arXiv:2504.15965, 2025.",
    "L. Chen, M. Zaharia and J. Zou, \"FrugalGPT: How to use large language models while reducing cost and improving performance,\" arXiv:2305.05176, 2023.",
    "I. Ong et al., \"RouteLLM: Learning to route LLMs with preference data,\" arXiv:2406.18665, 2024.",
    "\"When to reason: Semantic router for vLLM,\" arXiv:2510.08731, 2025.",
    "\"UCCI: Calibrated uncertainty for cost-optimal LLM cascade routing,\" arXiv:2605.18796, 2026.",
    "\"Confident or seek stronger: Uncertainty-based on-device LLM routing,\" arXiv:2502.04428, 2025.",
    "\"Cluster, route, escalate: Cascaded framework for cost-aware LLM serving,\" arXiv:2606.27457, 2026.",
    "\"Precise length control for large language models,\" Natural Language Processing Journal, Elsevier, 2025.",
    "\"BudgetThinker: Empowering budget-aware LLM reasoning with control tokens,\" arXiv:2508.17196, 2025.",
    "\"An empirical study of LLM reasoning ability under strict output length constraint,\" arXiv:2504.14350, 2025.",
]


# The template shows references grouped by source type, each in IEEE format.
# Grouping changes the order, so numbers are reassigned in the printed order
# and every citation in the body is remapped from the same table -- doing it by
# hand across fifteen call sites is how a report ends up citing the wrong paper.
REFERENCE_GROUPS = [
    ("Journals: <IEEE Format>", [28, 22, 38]),
    ("Conference: <IEEE Format>", [1, 2, 16, 12, 23]),
    ("Preprints and technical reports: <IEEE Format>",
     [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21,
      25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40]),
    ("Weblinks:", [24]),
]

RENUM = {old: new for new, old in enumerate(
    [r for _, members in REFERENCE_GROUPS for r in members], start=1)}


def remap_citations(text: str) -> str:
    """Rewrite [n] and [a]-[b] ranges in body text to the printed numbering."""
    import re                                              # noqa: PLC0415

    def one(m):
        n = int(m.group(1))
        return f"[{RENUM.get(n, n)}]"

    def rng(m):
        a, b = int(m.group(1)), int(m.group(2))
        lo = min(RENUM.get(i, i) for i in range(a, b + 1))
        hi = max(RENUM.get(i, i) for i in range(a, b + 1))
        return f"[{lo}]-[{hi}]"

    text = re.sub(r"\[(\d+)\]-\[(\d+)\]", rng, text)
    return re.sub(r"\[(\d+)\]", one, text)


# ---------------------------------------------------------------- writing --

def clear_body(doc) -> None:
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith(("}p", "}tbl")):
            body.remove(child)


def para(doc, text="", *, size=12, bold=False, italic=False, upper=False,
         align=None, spacing=1.15, before=0, after=6):
    p = doc.add_paragraph()
    run = p.add_run(text.upper() if upper else text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    return p


def heading(doc, text, level):
    """Heading levels exactly as the template specifies them.

        level 1   Times New Roman 14, Bold, Upper Case, spacing 1.5
        level 2   Times New Roman 13, Bold, Title Case, spacing 1.5
        level 3   Times New Roman 12, Bold WITH ITALIC, Title Case, 1.5
    """
    if level == 1:
        return para(doc, text, size=14, bold=True, upper=True, spacing=1.5,
                    before=18, after=8)
    if level == 2:
        return para(doc, text, size=13, bold=True, spacing=1.5,
                    before=14, after=6)
    return para(doc, text, size=12, bold=True, italic=True, spacing=1.5,
                before=10, after=5)


def borderless(table) -> None:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for b in tcPr.findall(f"{ns}tcBorders"):
                tcPr.remove(b)


def add_table(doc, header, rows, caption):
    table = doc.add_table(rows=len(rows) + 1, cols=len(header))
    table.style = "Table Grid"
    for j, text in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.font.name, run.font.size, run.bold = FONT, Pt(10.5), True
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.name, run.font.size = FONT, Pt(10)
    para(doc, caption, size=10.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         before=4, after=10)


def add_image(doc, path, width_in, caption):
    if not Path(path).exists():
        para(doc, f"[missing figure: {path}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))
    para(doc, caption, size=10.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         before=4, after=10)


def title_page(doc) -> None:
    C = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "BCSE497J - PROJECT-I", size=13, bold=True, align=C, spacing=1.5,
         before=30)
    para(doc, TITLE, size=16, bold=True, upper=True, align=C, spacing=1.5,
         before=26, after=8)
    para(doc, SUBTITLE, size=12, italic=True, align=C, spacing=1.5, after=22)
    para(doc, "Submitted in partial fulfilment of the requirements for the "
              "degree of", size=12, align=C, spacing=1.5)
    para(doc, "B.Tech.", size=13, bold=True, align=C, spacing=1.5)
    para(doc, "in", size=12, align=C, spacing=1.5)
    para(doc, "Computer Science and Engineering", size=13, bold=True, align=C,
         spacing=1.5, after=20)
    para(doc, "by", size=12, align=C, spacing=1.5, after=8)

    # The template gives the candidates as a 3x2 table, Reg. No. then NAME in
    # upper case and bold, sorted on register number.
    t = doc.add_table(rows=len(TEAM), cols=2)
    for i, (name, reg) in enumerate(TEAM):
        for j, text in enumerate((reg, name.upper())):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = C
            p.paragraph_format.line_spacing = 1.5
            run = p.add_run(text)
            run.font.name, run.font.size, run.bold = FONT, Pt(13), True
    borderless(t)

    para(doc, "Under the Supervision of", size=12, align=C, spacing=1.5,
         before=18, after=8)

    # ...and the guide as a 3x1 table: name (bold), designation, school.
    g = doc.add_table(rows=3, cols=1)
    for i, (text, bold) in enumerate(((GUIDE, True),
                                      (DESIGNATION, False),
                                      (SCHOOL, False))):
        cell = g.cell(i, 0)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = C
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(text)
        run.font.name, run.font.size, run.bold = FONT, Pt(13 if bold else 12), bold
    borderless(g)

    para(doc, "Vellore Institute of Technology, Vellore", size=12, align=C,
         spacing=1.5, before=14)
    para(doc, "September 2026", size=12, bold=True, align=C, spacing=1.5,
         before=16)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def contents_page(doc) -> None:
    """Three columns -- Sl.No | Contents | Page No. -- as the template's own
    table of contents is laid out, with borders removed as it instructs."""
    heading(doc, "TABLE OF CONTENTS", 1)

    rows = [("", "Abstract", "i")]
    n = 0
    for kind, payload in CONTENT:
        if kind == "h1":
            n += 1
            num, _, title = payload.partition(" ")
            rows.append((num, title.upper(), ""))
        elif kind == "h2":
            rows.append(("", payload, ""))
        elif kind == "h3":
            rows.append(("", "     " + payload.strip(), ""))
    rows.append((f"{n + 1}.", "REFERENCES", ""))

    table = doc.add_table(rows=len(rows) + 1, cols=3)
    header = ("Sl.No", "Contents", "Page No.")
    for j, text in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(text)
        run.font.name, run.font.size, run.bold = FONT, Pt(12), True

    for i, (sl, label, pg) in enumerate(rows, start=1):
        top_level = bool(sl)
        for j, text in enumerate((sl, label, pg)):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.5
            run = p.add_run(text)
            run.font.name = FONT
            run.font.size = Pt(12)
            run.bold = top_level or label == "Abstract"
            if j == 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    borderless(table)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build(template: Path, out: Path) -> None:
    doc = docx.Document(str(template))
    clear_body(doc)

    title_page(doc)

    heading(doc, "ABSTRACT", 1)
    for block in ABSTRACT:
        para(doc, block, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    contents_page(doc)

    J = WD_ALIGN_PARAGRAPH.JUSTIFY
    for kind, payload in CONTENT:
        if kind in ("p", "b"):
            payload = remap_citations(payload)
        elif kind == "tbl":
            head, body_rows, cap = payload
            body_rows = [[remap_citations(str(c)) for c in r]
                         for r in body_rows]
            payload = (head, body_rows, cap)
        if kind == "h1":
            heading(doc, payload, 1)
        elif kind == "h2":
            heading(doc, payload, 2)
        elif kind == "h3":
            heading(doc, payload, 3)
        elif kind == "p":
            para(doc, payload, align=J)
        elif kind == "b":                       # bolded lead-in sentence
            p = doc.add_paragraph()
            head, _, rest = payload.partition(". ")
            for text, bold in ((head + ". ", True), (rest, False)):
                run = p.add_run(text)
                run.font.name, run.font.size, run.bold = FONT, Pt(12), bold
            p.alignment = J
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(6)
        elif kind == "tbl":
            add_table(doc, *payload)
        elif kind == "img":
            add_image(doc, *payload)

    heading(doc, "5. REFERENCES", 1)
    for group, members in REFERENCE_GROUPS:
        para(doc, group, size=12, bold=True, italic=True, spacing=1.5,
             before=10, after=5)
        for old in members:
            p = para(doc, f"[{RENUM[old]}]  {REFERENCES[old - 1]}",
                     size=11.5, after=5)
            p.paragraph_format.left_indent = Pt(30)
            p.paragraph_format.first_line_indent = Pt(-30)

    doc.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.template, a.out)
