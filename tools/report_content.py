"""The report's text, in one place.

Kept apart from the builders so the deck and the report cannot drift: a number
corrected here is corrected in both. Every figure matches `figures/` and what
the live demo prints.
"""

TITLE = "Token-Efficient LLM Interaction on CPU-Only Hardware"
SPECIALISATION = ""   # no specialisation; the template prints one only if given
GUIDE = "Dr Sathya K"
#: Sorted on register number, as the template requires.
TEAM = [("Alok Singh", "23BCI0158"),
        ("Arrsh Tripathi", "23BCI0191"),
        ("Dev Singh", "23BCE0794")]

ABSTRACT = [
    "Language models charge by the token, and on a CPU-only laptop they spend almost all of "
    "their time reading the prompt rather than writing the answer. A substantial literature "
    "has grown around reducing that cost — prompt compression, semantic caching, KV-cache "
    "reuse, model routing and output budgeting — but it has grown in isolated strands. Each "
    "technique is proposed, measured against an uncompressed baseline, and published alone. "
    "Almost nothing measures what happens when they are composed, which is what any real "
    "deployment does.",

    "This project surveys forty papers across eight strands, extracts their limitations with "
    "respect to a CPU-only single-user deployment, and derives six research gaps. It then "
    "builds Parsimony: a middleware layer of seven optimisation modules plus an always-on "
    "fidelity gate, in which every module proposes a change and a single orchestrator commits "
    "it only after a safety check. Because every module is independently switchable, the "
    "system is a 2^4 factorial experiment rather than a fixed pipeline.",

    "Measured over 151 conversations and 263 requests against a locally hosted 1.5-billion "
    "parameter model, the full stack removes 33.9% of tokens with no loss of answer accuracy: "
    "gold accuracy rises from 92.5% to 97.5% with zero regressions. Three findings are new. "
    "Savings do not compound, with an additivity shortfall of 1.63 percentage points whose "
    "size is a property of the configuration rather than a constant. Prefill dominates CPU "
    "inference at 92 to 99 per cent of total time, approximately 8.5 milliseconds per input "
    "token. And the cache similarity thresholds published as safe are unsafe at this scale: "
    "an adversarial negation pair scores higher than every genuine paraphrase in the test set, "
    "so a verifier rather than a threshold is what makes reuse safe, taking the false-answer "
    "rate from 26.7% to 0.0%.",
]

# Each section: (title, [(sub-heading, [blocks…]), …]).  A block may be a
# string, a "• " bullet, or a (rows, header) tuple rendered as a table.
SECTIONS = [
    ("Introduction", [
        ("1.1 Background", [
            "Large language models are increasingly deployed on consumer hardware rather than in "
            "data centres. A model of one to three billion parameters, quantised to four bits, "
            "runs on an ordinary laptop with no graphics card. This removes the per-token "
            "monetary cost of a hosted API, but replaces it with a latency cost that is far more "
            "severe: measurements in this project show that reading the prompt accounts for 92 to "
            "99 per cent of the time spent on a request, at roughly 8.5 milliseconds per input "
            "token. Every token removed from the input is therefore repaid directly in seconds of "
            "waiting.",
        ]),
        ("1.2 Motivation", [
            "The efficiency literature offers many ways to reduce the number of tokens sent to a "
            "model. Each is published in isolation and evaluated against an uncompressed "
            "baseline. A practitioner assembling a real system must combine several of them, and "
            "at that point the published numbers stop being a guide: nothing in the literature "
            "states whether a technique that saves ten per cent alone still saves ten per cent "
            "beside another. This project was motivated by that missing measurement, and by the "
            "observation that the safety of these techniques is asserted far more often than it "
            "is tested.",
        ]),
        ("1.3 Scope of the Project", [
            "In scope: a middleware layer sitting between an application and a locally hosted "
            "small language model; seven optimisation modules and one fidelity gate; an "
            "evaluation harness capable of a full factorial ablation; a frozen, hashed corpus of "
            "151 conversations, 90 adversarial and control pairs and 40 gold answers; and "
            "reproducibility of every reported figure from raw logs.",
            "Out of scope: training or fine-tuning any model; a graphical or web interface; "
            "multi-user or multi-tenant serving; and any component requiring a graphics card, "
            "network access at inference time, or paid service.",
        ]),
    ]),

    ("Project Description and Goals", [
        ("2.1 Literature Review", [
            "Forty papers were reviewed across eight strands. For each, the method proposed, its "
            "limitation with respect to this project's setting, and the metrics it reports were "
            "recorded. The limitations below are stated relative to a CPU-only, single-user, "
            "multi-turn deployment in which several techniques are composed; they are not "
            "criticisms of the papers on their own terms, most being data-centre serving or "
            "single-technique works that are sound in their own context.",
            ([
                ("Prompt compression", "15",
                 "LLMLingua (EMNLP 2023), Selective Context (EMNLP 2023), LLMLingua-2, "
                 "evaluator-head and generative compression",
                 "Every method is measured alone against an uncompressed baseline; none is "
                 "measured in the presence of a cache"),
                ("Semantic caching", "7",
                 "GPTCache (NLP-OSS 2023), MeanCache, ContextCache, key-collision attacks",
                 "All rely on a single similarity threshold quoted as safe at 0.85 to 0.92, "
                 "never validated against opposite-meaning pairs"),
                ("KV cache and prefix reuse", "5",
                 "PagedAttention (SOSP 2023), vLLM automatic prefix caching, ChunkAttention",
                 "Written for GPU serving; the application-side cost of an unstable prompt "
                 "head is never priced"),
                ("Positional effects in context", "4",
                 "Lost in the Middle (TACL 2024), adaptive memory, multi-turn surveys",
                 "Motivates reordering by relevance, which destroys prefix reuse — a conflict "
                 "neither strand names"),
                ("Routing and cascades", "6",
                 "FrugalGPT, RouteLLM, uncertainty-based on-device routing",
                 "Assumes the larger tier is meaningfully better, which does not hold at 1.5B "
                 "against 3B in our measurements"),
                ("Output length control", "3",
                 "Length-difference positional encoding, BudgetThinker, constrained reasoning",
                 "Requires fine-tuning, which is out of reach on the target hardware"),
                ("Evaluation validity", "4",
                 "Position bias and self-preference bias in LLM-as-a-judge",
                 "Documents the biases; rarely prescribes a pre-registered check before "
                 "reporting a score"),
                ("Small models, quantisation, energy", "6",
                 "Small-model and on-device surveys, AWQ, energy benchmarking",
                 "All energy measurement is GPU data-centre; the CPU single-user regime is "
                 "not covered"),
            ], ["Strand", "Papers", "Representative work", "Limitation w.r.t. this project"]),
            "Four papers sit closest to this work. GPTCache established the embed-compare-"
            "threshold design that every semantic cache now follows, and Section 5 shows that "
            "the threshold is the wrong place to put the safety. Lost in the Middle and vLLM's "
            "automatic prefix caching are measured against each other, because the first "
            "recommends reordering content by relevance while the second requires the prompt "
            "head to remain byte-stable; doing the first destroys the second. An Empirical Study "
            "on Prompt Compression (ICLR 2025) names the two failure modes — altered-semantic "
            "and information-loss hallucination — that the fidelity gate exists to prevent, "
            "without proposing a run-time mechanism to prevent them.",
        ]),
        ("2.2 Research Gap", [
            "• Gap 1. Compositional effects are unmeasured. Every technique is evaluated alone, "
            "so whether the savings add when composed is unknown.",
            "• Gap 2. The prefill and decode split is uncharacterised for CPU-only single-user "
            "inference; the literature measures GPU throughput.",
            "• Gap 3. The interaction between compression and caching is unexamined. Normalising "
            "a query before the cache sees it should raise the hit rate, but whether it also "
            "raises the false-hit rate is not reported.",
            "• Gap 4. Published cache thresholds are not adversarially validated at small scale. "
            "Where near-identical, opposite-meaning questions actually sit is not reported.",
            "• Gap 5. No work tests whether its own calibration transfers to a second "
            "configuration without re-tuning.",
            "• Gap 6. Self-improving systems are evaluated on the data they were mined from, "
            "which measures memorisation rather than transfer.",
        ]),
        ("2.3 Objectives", [
            "RQ1. When several token-reduction techniques are composed in one pipeline, do their "
            "individual savings add, and if not, by how much do they fall short?",
            "RQ2. On CPU-only hardware, which half of a request dominates the cost, and what is "
            "one input token worth in milliseconds?",
            "RQ3. Can a semantic cache be made safe against near-identical, opposite-meaning "
            "questions without a second neural forward pass?",
            "RQ4. Does a calibration transfer to a different configuration without re-tuning?",
            "Corresponding objectives: build an instrumented pipeline in which every module is "
            "independently switchable; attach a real model and measure the prefill and decode "
            "split; construct an adversarial pair set and drive the false-answer rate below two "
            "per cent; and re-run the identical protocol on a second configuration.",
        ]),
        ("2.4 Problem Statement", [
            "Build a middleware layer that measurably reduces the token cost of interacting with "
            "a small language model on CPU-only consumer hardware, without changing the meaning "
            "of any request, and instrument it so that the contribution of every individual "
            "module, and every interaction between modules, is separately measurable and "
            "independently reproducible.",
            "The emphasis on meaning is deliberate. A system that saves tokens by quietly "
            "damaging requests is not a contribution; the safety property is the constraint under "
            "which every optimisation must operate, and it is enforced by a single gate through "
            "which all proposed edits pass.",
        ]),
        ("2.5 Project Plan", [
            ([
                ("Sprint 0-1", "Architecture, L0 contracts, ledger schema, deterministic "
                               "provider, frozen corpus", "Complete"),
                ("Sprint 2", "All eight modules; factorial sweep runner; threshold calibration",
                 "Complete"),
                ("Sprint 3", "Real model attached via Ollama; latency and gold accuracy measured",
                 "Complete"),
                ("Sprint 4", "Calibration transfer, policy transfer, adversarial hardening",
                 "Complete"),
                ("Remaining", "Neural encoder swap; full sweep on the real model; a usable judge; "
                              "a second model for escalation", "Planned"),
            ], ["Phase", "Work", "Status"]),
            "Current status: all eight modules built, 719 automated tests passing, 39 "
            "architecture decision records, and every reported figure regenerating from raw logs "
            "in approximately 100 seconds.",
        ]),
    ]),

    ("Technical Specification", [
        ("3.1 Requirements", [
            "3.1.1 Functional",
            "• F1. Every module is independently switchable, so that a factorial cell means what "
            "it claims.",
            "• F2. Stage order is configuration rather than code, validated against a "
            "reads-and-writes dependency graph.",
            "• F3. Every stage writes a ledger row, including stages that did nothing; an "
            "invisible stage is an unauditable one.",
            "• F4. No transformation is applied until the fidelity gate has approved it.",
            "• F5. A run that requests the real model and cannot reach it refuses, rather than "
            "silently falling back to the deterministic stand-in.",
            "• F6. Every reported figure regenerates from raw logs with a single command.",
            "3.1.2 Non-Functional",
            "• CPU-only operation on 8 GB of RAM, with no graphics card and no network access at "
            "inference time.",
            "• Middleware overhead below 120 milliseconds per request, measured and enforced.",
            "• Dependencies limited to numpy, tokenizers, typer, rich and pytest.",
            "• Personally identifying information redacted at the cache write boundary.",
        ]),
        ("3.2 Feasibility Study", [
            "Technical feasibility. The complete pipeline runs today on the target hardware "
            "against qwen2.5:1.5b-instruct at four-bit quantisation, entirely offline. The "
            "measured middleware overhead is 6 to 41 milliseconds against a 120 millisecond "
            "budget.",
            "Economic feasibility. There is no paid component of any kind. The runtime, the model "
            "weights and every dependency are openly licensed, and inference happens on hardware "
            "the team already owns.",
            "Social feasibility. Because the system runs entirely offline, no user text leaves "
            "the machine. The cache, being the one component with memory, redacts personally "
            "identifying information before writing.",
        ]),
        ("3.3 System Specification", [
            "Hardware, as measured: AMD Ryzen 7 5800HS, 8 physical cores and 16 logical "
            "processors, 15.4 GB RAM, no discrete graphics.",
            "Software: Python 3.11; Ollama 0.33.2; qwen2.5:1.5b-instruct quantised to Q4_K_M "
            "(0.92 GB) as the primary model, with llama3.2:3b (2.0 GB) used for escalation and "
            "judging experiments.",
        ]),
    ]),

    ("Design Approach and Details", [
        ("4.1 System Architecture", [
            "The governing design decision is that modules propose and a single orchestrator "
            "commits. A proposal describes an edit rather than performing it. Three properties "
            "follow that no individual module has to implement: ablation becomes exact, because "
            "switching a module off genuinely removes its effect; safety has one choke point, so "
            "fidelity is not re-implemented seven times; and reverting is free, because a refused "
            "proposal is simply never assigned.",
            "Stage order is configuration validated against a dependency graph. This is what "
            "makes Gap 3 answerable at all: the cache lookup can be moved before or after the "
            "compressor without editing the pipeline, and the effect measured.",
            ([
                ("M6a", "Deterministic router",
                 "Recognises arithmetic and date questions and computes the answer exactly, "
                 "using an AST evaluator rather than eval",
                 "23 tokens to 0"),
                ("M2", "Semantic cache",
                 "Embeds the question and applies a three-zone accept / verify / reject policy; "
                 "borderline matches are settled by four set comparisons",
                 "Prompt never sent"),
                ("M3a", "History selector",
                 "Scores prior turns for relevance and keeps those that fit the budget",
                 "8 turns to 6"),
                ("M3b", "History arranger", "Orders the kept turns",
                 "Order changes cost"),
                ("M1a", "Compressor, tier 1",
                 "Removes greetings, politeness and markdown, losslessly and sentence-aware",
                 "28 to 17 tokens"),
                ("M1b", "Compressor, tier 2",
                 "Deletes a sentence restating a fact already present",
                 "36 to 26 tokens"),
                ("M1c", "Compressor, tier 3",
                 "Shortens phrasing, with a guard rejecting any edit that does not reduce tokens",
                 "Rejects zero-yield edits"),
                ("M4", "Prefix-stable assembler",
                 "Assembles the prompt with invariant content first, so the model reuses its "
                 "key-value cache",
                 "212 ms against 18,914 ms"),
                ("M5", "Output budgeter",
                 "Classifies the question and sets the generation budget, with a streaming "
                 "early stop on trigram novelty",
                 "48 to 640 tokens"),
                ("M6b", "Escalation router",
                 "Scores complexity and decides whether a larger model is warranted",
                 "Calibrated; off by default"),
                ("M8", "Fidelity gate",
                 "Runs on every proposal from every stage and refuses any edit that would drop "
                 "a number, entity, negation or operative modifier",
                 "26.7% to 0.0% false answers"),
                ("M7", "Policy learner",
                 "Offline: mines recurring questions and standing facts into a bundle that "
                 "warm-starts the cache",
                 "+17.83 pp at 57% recurrence"),
            ], ["ID", "Module", "What it does", "Measured effect"]),
        ]),
        ("4.2 Design", [
            "Every stage writes one ledger row recording the module, the outcome — applied, "
            "reverted, short-circuited or no-op — the token count before and after, the duration, "
            "the rationale, and any gate event. A stage that did nothing still writes a row. The "
            "entire evaluation depends on the trace being complete, and a stage that could stay "
            "silent would be unauditable.",
            "Data flow. A request enters as a query plus conversation history. Each stage may "
            "answer it outright and short-circuit, propose a context patch, or decline. Patches "
            "are checked by the gate and either committed or reverted. What survives is assembled "
            "into a prompt, sent to the provider, and the response is written back to the cache "
            "with redaction applied at the boundary.",
        ]),
    ]),

    ("Results and Discussion", [
        ("5.1 The headline result", [
            "A full 2^4 factorial over compressor, cache, history manager and output budgeter, "
            "across 151 conversations and 263 requests in 17 cells, with bootstrap confidence "
            "intervals and partial eta-squared effect sizes.",
            ([
                ("M5 output budgeter", "+13.44", "0.556"),
                ("M3 history manager", "+11.82", "0.430"),
                ("M2 semantic cache", "+1.94", "0.012"),
                ("M1 compressor", "+0.23", "0.000"),
                ("M3 x M5 interaction", "-0.70", "0.002"),
            ], ["Effect", "Estimate (pp)", "Partial eta-squared"]),
            "The full stack reaches 33.9% total token reduction. Both material interaction terms "
            "are negative and both involve M5, because trimming history and shortening output "
            "reduce the same conversation. The additivity shortfall is 1.63 percentage points "
            "with a 95% confidence interval of -0.02 to +3.23.",
            "The sharper result is that the shortfall is configuration-dependent. Under a weaker "
            "lexical encoder it measures 2.53 points with an interval excluding zero; improving "
            "the encoder makes the cache hit more often, so it overlaps its neighbours less, and "
            "the shortfall falls to 1.63 points with an interval touching zero. The weaker "
            "encoder was not reinstated to protect the interval.",
        ]),
        ("5.2 Where the time goes", [
            ([
                ("146", "1,360", "122", "91.7%"),
                ("474", "3,902", "165", "95.9%"),
                ("906", "7,691", "131", "98.3%"),
                ("1,338", "11,609", "136", "98.8%"),
            ], ["Input tokens", "Prefill (ms)", "Decode (ms)", "Prefill share"]),
            "Prefill is linear in input length at approximately 8.5 milliseconds per token and "
            "accounts for 92 to 99 per cent of total time. Applying this rate to the ablation, "
            "the full stack's 11,836 saved input tokens amount to 100.6 seconds of prefill across "
            "the corpus, or 383 milliseconds per request.",
            "Prompt order carries a separate and larger cost. With the same content and a token "
            "count differing by half a per cent, a stable prompt prefix reaches 98.5% key-value "
            "cache reuse and a steady-state prefill of 212 milliseconds, while a volatile head "
            "reaches 0.5% reuse and 18,914 milliseconds — approximately eighty times more "
            "expensive. Every metric in the compression literature scores these two "
            "configurations identically.",
        ]),
        ("5.3 Safety of the cache", [
            "Measured on 45 adversarial pairs differing by one operative token, plus 45 controls "
            "that mean the same thing and should match. The adversarial negation pair sits at "
            "cosine 0.924, higher than every genuine paraphrase in the set, so the 0.85 to 0.92 "
            "range published as safe would accept it and return the opposite answer.",
            ([
                ("Modifier", "78%", "0%"),
                ("Negation", "31%", "0%"),
                ("Entity", "11%", "0%"),
                ("Number", "0%", "0%"),
                ("Overall", "26.7%", "0.0%"),
            ], ["Operative token", "False-hit rate before", "After the verifier"]),
            "No threshold separates these pairs, because the adversarial cases sit above the "
            "genuine ones. What separates them is a verifier of four set comparisons costing "
            "microseconds, three of which are not performed by any surveyed cache: operative "
            "modifiers such as minimum against maximum, morphological negation such as possible "
            "against impossible, and alphanumeric identifiers such as pandas against Panda3D.",
        ]),
        ("5.4 Quality, transfer, and threats to validity", [
            "Answer quality does not pay for the savings. On 40 gold items with the real model, "
            "baseline accuracy is 92.5% and full-stack accuracy 97.5%, and paired per item there "
            "are zero regressions: not one answer the baseline answered correctly was lost.",
            "Calibration transfers as a ratio but not as a mechanism. Re-running the protocol "
            "against a second real vocabulary leaves reduction ratios within 0.1 percentage "
            "points and module ranking identical, while one of the two mechanisms this project "
            "had proposed for a tokenizer effect proved specific to a single tokenizer.",
            "Mined policy transfer is a function of traffic repetition, measured at +0.00 points "
            "at zero per cent recurrence and +17.83 points at 57 per cent, with no fidelity "
            "violations at any level. The exact zero at zero recurrence is what makes the "
            "remainder credible.",
            "Threats to validity. All latency figures come from a single hardware configuration "
            "and are not claimed to generalise. The gold set of 40 items contains no question in "
            "the difficulty band where a 1.5-billion parameter model fails and a 3-billion one "
            "succeeds, so the escalation result measures the gold set as much as the models. The "
            "embedder is lexical rather than neural, which shifts operating points though not "
            "rankings. And the corpus recurrence of 1.9% is why policy transfer is reported as a "
            "curve over synthetic traffic rather than as a single number.",
        ]),
    ]),
]

REFERENCES = [
    '[1] Y. Li, B. Dong, C. Lin and F. Guerin, "Compressing Context to Enhance Inference '
    'Efficiency of Large Language Models," in Proc. EMNLP, 2023.',
    '[2] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang and L. Qiu, "LLMLingua: Compressing Prompts for '
    'Accelerated Inference of Large Language Models," in Proc. EMNLP, 2023.',
    '[3] H. Jiang et al., "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context '
    'Scenarios via Prompt Compression," arXiv:2310.06839, 2023.',
    '[4] "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt '
    'Compression," arXiv:2403.12968, 2024.',
    '[5] "An Empirical Study on Prompt Compression for Large Language Models," in Proc. ICLR, '
    '2025.',
    '[6] "Understanding and Improving Information Preservation in Prompt Compression for LLMs," '
    'arXiv:2503.19114, 2025.',
    '[7] F. Bang, "GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster '
    'Answers and Cost Savings," in Proc. NLP-OSS at EMNLP, 2023.',
    '[8] "MeanCache: User-Centric Semantic Caching for LLM Web Services," arXiv:2403.02694, 2024.',
    '[9] "ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries in Large Language '
    'Models," arXiv:2506.22791, 2025.',
    '[10] "From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching," '
    'arXiv:2601.23088, 2026.',
    '[11] "Enhancing adversarial resilience in semantic caching for secure retrieval augmented '
    'generation systems," Scientific Reports, 2026.',
    '[12] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with '
    'PagedAttention," in Proc. SOSP, 2023.',
    '[13] "ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase '
    'Partition," arXiv:2402.15220, 2024.',
    '[14] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," '
    'Transactions of the ACL, vol. 12, pp. 157-173, 2024.',
    '[15] L. Chen, M. Zaharia and J. Zou, "FrugalGPT: How to Use Large Language Models While '
    'Reducing Cost and Improving Performance," arXiv:2305.05176, 2023.',
    '[16] I. Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," '
    'arXiv:2406.18665, 2024.',
    '[17] "Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing," '
    'arXiv:2502.04428, 2025.',
    '[18] "BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens," '
    'arXiv:2508.17196, 2025.',
    '[19] "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge," '
    'arXiv:2406.07791, 2024.',
    '[20] "Self-Preference Bias in LLM-as-a-Judge," arXiv:2410.21819, 2024.',
    '[21] "A Survey of Small Language Models," arXiv:2410.20011, 2024.',
    '[22] "On-Device Language Models: A Comprehensive Review," arXiv:2409.00088, 2024.',
    '[23] "Taming the Titans: A Survey of Efficient LLM Inference Serving," arXiv:2504.19720, '
    '2025.',
    '[24] J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and '
    'Acceleration," arXiv:2306.00978, 2023.',
    '[25] "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference," '
    'arXiv:2505.09598, 2025.',
    "",
    "The complete forty-paper survey, with the method, limitation and reported metrics for each, "
    "accompanies this report as a separate dossier.",
]
