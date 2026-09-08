"""Fill the Review-2 PPT and the Project-I report templates.

Both are filled in place from the templates the school supplies, rather than
rebuilt from scratch: the templates carry the school's masters, headers and
numbering, and a deck that does not look like the others in the room is a
distraction from the content.

Every number here comes from `figures/` or from the decision log, and is the
same number the demo prints live.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

import docx
from docx.shared import Pt as DocxPt

# --------------------------------------------------------------------- data --

TITLE = "Token-Efficient LLM Interaction on CPU-Only Hardware"
SUBTITLE = "Parsimony — a stacked, self-improving optimisation layer for small language models"
TEAM = [("Arrsh Tripathi", "23BCI0191"),
        ("Alok Singh", "23BCI0158"),
        ("Dev Singh", "23BCE0794")]
GUIDE = "Dr Sathya K"

AIM = [
    "Cut the token cost of talking to a small language model on a CPU-only laptop —",
    "without changing the meaning of any request.",
    "",
    "Seven optimisation modules plus an always-on fidelity gate. Every module PROPOSES a",
    "change; one orchestrator COMMITS it, and only after the gate has checked it.",
    "",
    "Because every module is independently switchable, the system is a 2^4 factorial",
    "experiment rather than a fixed pipeline — which is what makes the headline result",
    "measurable at all.",
]

ABSTRACT = [
    "Prompt compression, semantic caching, KV-cache reuse, routing and output budgeting each",
    "reduce token cost. They are published ALONE, each against an uncompressed baseline.",
    "",
    "We survey 40 papers, derive six research gaps, and build all of these techniques into one",
    "instrumented pipeline so their INTERACTION can be measured.",
    "",
    "Result: 33.9% fewer tokens with zero loss of answer accuracy (92.5% → 97.5% on 40 gold",
    "items, no regressions). Savings do NOT compound — an additivity shortfall of 1.63 pp.",
    "Prefill is 92–99% of CPU time at ~8.5 ms per input token. And the cache thresholds",
    "published as safe would serve the opposite answer on our adversarial set.",
]

LITERATURE = [
    ("Prompt compression — 15 papers",
     "LLMLingua (EMNLP'23), Selective Context (EMNLP'23), LLMLingua-2. Up to 20x compression."
     "  Gap: every one measured in isolation, never against a cache."),
    ("Semantic caching — 7 papers",
     "GPTCache (NLP-OSS'23), MeanCache, ContextCache. All use a single similarity threshold,"
     " quoted safe at 0.85–0.92.  Gap: never validated against opposite-meaning pairs."),
    ("KV cache / prefix reuse — 5 papers",
     "PagedAttention (SOSP'23), vLLM prefix caching, ChunkAttention."
     "  Gap: GPU serving; the application cost of an unstable prompt head is never priced."),
    ("Positional effects — 4 papers",
     "Lost in the Middle (TACL'24): accuracy peaks at the prompt edges."
     "  Gap: reordering by relevance DESTROYS prefix reuse — neither literature names it."),
    ("Routing and cascades — 6 papers",
     "FrugalGPT, RouteLLM. Up to 98% cost reduction against commercial APIs."
     "  Gap: assumes the larger tier is meaningfully better — untrue at 1.5B vs 3B here."),
    ("Output length, evaluation, energy — 3 + surveys",
     "Length control needs fine-tuning; judge-bias work documents position and self-preference"
     " bias; all energy measurement is GPU datacentre."),
]

GAPS = [
    ("Gap 1  Composition is unmeasured",
     "Every technique is evaluated alone. Whether savings ADD when composed is unknown."),
    ("Gap 2  CPU prefill/decode split uncharacterised",
     "The literature measures GPU throughput. Which half a laptop pays for is unestablished."),
    ("Gap 3  Compression x cache interaction",
     "Normalising before the cache should raise hit rate. Does it raise FALSE hit rate?"),
    ("Gap 4  Published cache thresholds unvalidated",
     "0.85–0.92 is quoted safe; nobody reports where opposite-meaning pairs actually sit."),
    ("Gap 5  Does a calibration transfer?",
     "Thresholds are published as universal. Nobody re-runs the protocol without re-tuning."),
    ("Gap 6  Does mined policy transfer?",
     "Self-improving systems are evaluated on the data they were mined from."),
]

OBJECTIVES = [
    "RQ1  When techniques are composed, do their savings add — and if not, by how much short?",
    "RQ2  On CPU, which half of a request dominates, and what is one input token worth in ms?",
    "RQ3  Can a cache be made safe against opposite-meaning pairs without a second model pass?",
    "RQ4  Does a calibration transfer to a different configuration without re-tuning?",
    "",
    "Contribution 1  The additivity shortfall, measured: 1.63 pp, 95% CI [-0.02, +3.23] —",
    "                and shown to be a property of the CONFIGURATION, not a constant.",
    "Contribution 2  CPU cost structure: prefill 92–99%, ~8.5 ms/input token; and the price of",
    "                prompt order — 212 ms vs 18,914 ms for the same content.",
    "Contribution 3  A verifier that makes reuse safe without a second model: 26.7% → 0.0%.",
    "Contribution 4  Transfer tested both ways: ratios transfer, mechanisms do not; and mined",
    "                policy pays only above ~5–10% traffic repetition.",
]

ARCHITECTURE = [
    ("M6a  Deterministic router", "Answers maths and dates exactly, with no model", "23 tokens -> 0"),
    ("M2   Semantic cache", "Reuses an earlier answer; three-zone accept/verify/reject", "prompt never sent"),
    ("M3a  History selector", "Keeps only the prior turns that matter (MMR)", "8 turns -> 6"),
    ("M3b  History arranger", "Orders the kept turns", "order changes cost"),
    ("M1a  Tier 1 boilerplate", "Strips greetings, politeness, markdown", "28 -> 17 tokens"),
    ("M1b  Tier 2 redundancy", "Deletes a sentence repeating a known fact", "36 -> 26 tokens"),
    ("M1c  Tier 3 word level", "Shortens wording, with a negative-yield guard", "rejects zero-yield"),
    ("M4   Prefix-stable assembler", "Invariant text first, so the KV cache is reusable", "212 ms vs 18,914 ms"),
    ("M5   Output budgeter", "Caps answer length by question class", "48 ... 640 tokens"),
    ("M6b  Escalation router", "Decides whether a bigger model is warranted", "calibrated, off"),
    ("M8   FIDELITY GATE", "Checks EVERY proposal above; refuses any that drops a number,"
                           " name, negation or qualifier", "26.7% -> 0.0% false answers"),
]

FUNCTIONAL = [
    "F1  Every module independently switchable, so a factorial cell means what it claims.",
    "F2  Stage order is configuration, validated against a reads/writes dependency graph.",
    "F3  Every stage writes a ledger row — including stages that did nothing.",
    "F4  No transformation is applied until the fidelity gate has approved it.",
    "F5  A run that asks for the real model and cannot reach it REFUSES rather than",
    "    silently falling back to the deterministic stand-in.",
    "F6  Every reported figure regenerates from raw logs with one command (~100 s).",
    "",
    "Non-functional: CPU-only, 8 GB RAM, fully offline; middleware overhead under 120 ms;",
    "dependencies limited to numpy, tokenizers, typer, rich, pytest.",
]

RESULTS = [
    ("Headline — savings do not compound", [
        "M5 output budgeter  +13.44 pp   (eta^2 0.556)",
        "M3 history manager  +11.82 pp   (eta^2 0.430)",
        "M2 semantic cache    +1.94 pp",
        "M1 compressor        +0.23 pp",
        "Full stack: +33.9% total token reduction",
        "Additivity shortfall: 1.63 pp, 95% CI [-0.02, +3.23]",
    ]),
    ("Prefill dominates on CPU", [
        "146 tokens ->  1,360 ms prefill   (91.7% of total)",
        "1,338 tokens -> 11,609 ms prefill  (98.8% of total)",
        "~8.5 ms per input token  =>  100.6 s saved across the corpus, 383 ms/request",
    ]),
    ("Safety: the verifier, not the threshold", [
        "Adversarial negation pair sits at cosine 0.924 — ABOVE every genuine paraphrase",
        "False-answer rate 26.7% -> 0.0% at tau >= 0.92, on 45 adversarial + 45 controls",
    ]),
    ("Quality is not the price", [
        "Gold accuracy, real model: baseline 92.5%  ->  full stack 97.5%",
        "Zero regressions: not one answer the baseline got right was lost",
    ]),
]

CONCLUSION = [
    "The literature has a rich toolbox and a thin account of what happens when the tools",
    "are used together. The contribution here is a MEASUREMENT INSTRUMENT: eight techniques",
    "in one harness, each independently switchable, every decision written to a ledger.",
    "",
    "What it found, repeatedly, is that published operating points are configuration-specific.",
    "A threshold quoted as safe is unsafe here. A placement strategy justified by a well-cited",
    "attention result is 80x more expensive here. A calibration transfers as a ratio but not as",
    "a mechanism. An escalation premise that holds at 70B does not hold at 3B.",
    "",
    "For this class of system the deliverable is a CALIBRATION PROCEDURE, not a number.",
    "",
    "Status: 719 automated tests passing; 39 architecture decision records; all figures",
    "regenerate from raw logs in about 100 seconds.",
]

REFERENCES = [
    '[1] Y. Li, B. Dong, C. Lin, F. Guerin, "Compressing Context to Enhance Inference Efficiency '
    'of Large Language Models," EMNLP, 2023.',
    '[2] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, L. Qiu, "LLMLingua: Compressing Prompts for '
    'Accelerated Inference of Large Language Models," EMNLP, 2023.',
    '[3] F. Bang, "GPTCache: An Open-Source Semantic Cache for LLM Applications," NLP-OSS at '
    'EMNLP, 2023.',
    '[4] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with '
    'PagedAttention," SOSP, 2023.',
    '[5] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," TACL, '
    'vol. 12, pp. 157–173, 2024.',
    '[6] L. Chen, M. Zaharia, J. Zou, "FrugalGPT: How to Use Large Language Models While '
    'Reducing Cost and Improving Performance," arXiv:2305.05176, 2023.',
    '[7] I. Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," '
    'arXiv:2406.18665, 2024.',
    '[8] "An Empirical Study on Prompt Compression for Large Language Models," ICLR, 2025.',
    '[9] "From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching," '
    'arXiv:2601.23088, 2026.',
    '[10] "Enhancing adversarial resilience in semantic caching for secure retrieval augmented '
    'generation systems," Scientific Reports, 2026.',
    "",
    "Full 40-paper survey with limitations and metrics: docs/13-review2-dossier.md",
]

# ---------------------------------------------------------------------- ppt --


def body_placeholder(slide):
    """The content placeholder, whichever index the template gave it."""
    best = None
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        if sh == slide.shapes.title:
            continue
        # The largest text frame is the content area; the small ones are the
        # date, footer and slide-number placeholders the master supplies.
        area = (sh.width or 0) * (sh.height or 0)
        if best is None or area > (best.width or 0) * (best.height or 0):
            best = sh
    return best


def fill(slide, lines, size=14, bold_first=False):
    tf = body_placeholder(slide).text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = line
        for run in para.runs:
            run.font.size = Pt(size)
            if bold_first and i == 0:
                run.font.bold = True
        para.space_after = Pt(3)


def fill_pairs(slide, pairs, size=13):
    tf = body_placeholder(slide).text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for head, detail in pairs:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.text = head
        for run in para.runs:
            run.font.size = Pt(size)
            run.font.bold = True
        para.space_before = Pt(6)
        sub = tf.add_paragraph()
        sub.text = f"    {detail}"
        for run in sub.runs:
            run.font.size = Pt(size - 2)
        sub.space_after = Pt(2)


def build_ppt(template: Path, out: Path) -> None:
    prs = Presentation(str(template))
    s = prs.slides

    # 1 — title
    t = s[0].shapes.title.text_frame
    t.text = "B.Tech · BCSE497J Project-I"
    p = t.add_paragraph(); p.text = TITLE
    for r in p.runs:
        r.font.size = Pt(28); r.font.bold = True
    members = body_placeholder(s[0])
    if members is not None:
        fill(s[0],
             ["Team members:"]
             + [f"    {n}    ({r})" for n, r in TEAM]
             + ["", f"Faculty guide : {GUIDE}", "School of Computer Science and Engineering"],
             size=15)

    fill(s[2], AIM, size=15)
    fill(s[3], ABSTRACT, size=13)
    fill_pairs(s[4], LITERATURE, size=13)
    fill_pairs(s[5], GAPS, size=13)
    fill(s[6], OBJECTIVES, size=12)

    fill(s[7],
         ["Modules PROPOSE · one orchestrator COMMITS · the gate checks every proposal", ""]
         + [f"{code:<28}{does:<52}{effect}" for code, does, effect in ARCHITECTURE],
         size=11)

    fill(s[8], FUNCTIONAL, size=13)

    fill(s[9],
         ["Each module is switchable independently, which is what makes the ablation valid.", ""]
         + [f"{code.split()[0]:<6}{code.split(maxsplit=1)[1]:<24}{does}"
            for code, does, _ in ARCHITECTURE],
         size=11)

    res = []
    for head, rows in RESULTS:
        res.append(head)
        res += [f"    {r}" for r in rows]
        res.append("")
    fill(s[10], res, size=11)

    fill(s[11], CONCLUSION, size=12)
    fill(s[12], REFERENCES, size=10)

    prs.save(str(out))
    print(f"wrote {out}")


# --------------------------------------------------------------------- docx --

DOCX_SECTIONS = [
    ("ABSTRACT", ABSTRACT),
    ("1. INTRODUCTION", [
        "1.1 Background",
        "Language models charge by the token. On a CPU-only consumer laptop with no GPU, they also "
        "spend almost all of their time reading the prompt rather than writing the answer — measured "
        "here at 92–99% of total request time. Every token removed from the input is therefore paid "
        "back directly in seconds of waiting.",
        "",
        "1.2 Motivation",
        "A literature has grown around reducing this cost — prompt compression, semantic caching, "
        "KV-cache reuse, model routing, output budgeting. It has grown in isolated strands: each "
        "technique is proposed, measured against an uncompressed baseline, and published alone. "
        "Almost nothing measures what happens when they are composed, which is exactly what a real "
        "deployment does.",
        "",
        "1.3 Scope of the Project",
        "A middleware layer between an application and a locally hosted small language model. Seven "
        "optimisation modules plus an always-on fidelity gate, running on 8 GB of RAM with no GPU "
        "and no internet. Out of scope: training or fine-tuning any model, a web interface, and "
        "multi-user serving.",
    ]),
    ("2. PROJECT DESCRIPTION AND GOALS", [
        "2.1 Literature Review",
        "Forty papers across eight strands: prompt compression (15), semantic caching (7), KV cache "
        "and prefix reuse (5), positional effects (4), routing and cascades (6), output length "
        "control (3), and evaluation-validity and efficiency surveys. For each, the method, its "
        "limitation with respect to a CPU-only composed deployment, and the metrics it reports are "
        "tabulated in the accompanying dossier.",
        "",
        "The four closest antecedents are GPTCache (NLP-OSS 2023), which established the "
        "embed-compare-threshold design every semantic cache now follows; Lost in the Middle "
        "(TACL 2024) and vLLM automatic prefix caching, which this work measures against each other "
        "because their recommendations are in direct conflict; and An Empirical Study on Prompt "
        "Compression (ICLR 2025), which names the two failure modes the fidelity gate exists to "
        "prevent.",
        "",
        "2.2 Research Gap",
        "Six gaps follow from the survey: (1) compositional effects are unmeasured; (2) the "
        "prefill/decode split is uncharacterised for CPU-only single-user inference; (3) the "
        "compression-cache interaction is unexamined; (4) published cache thresholds are not "
        "adversarially validated at small scale; (5) no work tests whether its own calibration "
        "transfers; (6) self-improving systems are evaluated on the data they were mined from.",
        "",
        "2.3 Objectives",
        "RQ1: when techniques are composed, do their savings add, and if not by how much do they "
        "fall short? RQ2: on CPU, which half of a request dominates the cost? RQ3: can a semantic "
        "cache be made safe against opposite-meaning questions without a second neural pass? "
        "RQ4: does a calibration transfer without re-tuning?",
        "",
        "2.4 Problem Statement",
        "Build a middleware layer that measurably reduces the token cost of interacting with a small "
        "language model on CPU-only consumer hardware, without changing the meaning of any request, "
        "and instrument it so that the contribution of every individual module and every interaction "
        "between modules is separately measurable and independently reproducible.",
        "",
        "2.5 Project Plan",
        "Sprint 0-1: architecture, contracts, ledger, mock provider, corpus. Sprint 2: all eight "
        "modules and the factorial harness. Sprint 3: real model attached, latency and quality "
        "measured. Sprint 4: calibration transfer, policy transfer, adversarial hardening. "
        "Current status: all eight modules built, 719 tests passing, 39 decision records.",
    ]),
    ("3. TECHNICAL SPECIFICATION", [
        "3.1.1 Functional Requirements",
        "\n".join(FUNCTIONAL[:6]),
        "",
        "3.1.2 Non-Functional Requirements",
        "CPU-only operation on 8 GB of RAM with no GPU and no network. Middleware overhead under "
        "120 ms per request. Dependencies limited to numpy, tokenizers, typer, rich and pytest — a "
        "constraint that is itself checkable, and which the Ollama connector honours by using only "
        "the standard library.",
        "",
        "3.2 Feasibility Study",
        "Technical: the full pipeline runs on the target hardware today, against "
        "qwen2.5:1.5b-instruct at 4-bit quantisation. Economic: no paid component of any kind — the "
        "runtime, the model weights and every dependency are openly licensed. Social: the system "
        "runs entirely offline, so no user text leaves the machine; the cache redacts personally "
        "identifying information at the write boundary.",
        "",
        "3.3 System Specification",
        "Hardware as measured: AMD Ryzen 7 5800HS, 8 physical cores, 15.4 GB RAM, no discrete GPU. "
        "Software: Python 3.11, Ollama 0.33.2, qwen2.5:1.5b-instruct (Q4_K_M, 0.92 GB).",
    ]),
    ("4. DESIGN APPROACH AND DETAILS", [
        "4.1 System Architecture",
        "Modules propose; a single orchestrator commits. A proposal describes an edit rather than "
        "performing it, and three properties follow that no module has to implement: independent "
        "ablation, a single safety choke point, and free reverting. Stage order is configuration "
        "validated against a reads/writes dependency graph, which is what makes the "
        "compression-cache interaction answerable at all.",
        "",
        "The ten pipeline stages, in order:",
        "\n".join(f"{code} — {does} ({effect})" for code, does, effect in ARCHITECTURE),
        "",
        "4.2 Design",
        "Every stage writes one ledger row: module, outcome, tokens before and after, duration, "
        "rationale and any gate event. A stage that did nothing still writes a row — an invisible "
        "stage is an unauditable one, and the entire evaluation depends on the trace being complete.",
    ]),
    ("5. RESULTS AND DISCUSSION", [
        "\n".join([head] + [f"  {r}" for r in rows] + [""])
        for head, rows in RESULTS
    ]),
    ("6. REFERENCES", REFERENCES),
]


def build_docx(template: Path, out: Path) -> None:
    d = docx.Document(str(template))

    # The template is a specimen: every section is followed by a formatting note
    # and a worked sample. Appending our content after it keeps the school's
    # styles and page setup while making the submitted document ours.
    d.add_page_break()
    h = d.add_paragraph("PROJECT CONTENT")
    h.runs[0].font.bold = True
    h.runs[0].font.size = DocxPt(14)

    for heading, blocks in DOCX_SECTIONS:
        p = d.add_paragraph(heading)
        p.runs[0].font.bold = True
        p.runs[0].font.size = DocxPt(14)
        p.paragraph_format.space_before = DocxPt(14)
        for block in blocks:
            if not block:
                continue
            for line in str(block).split("\n"):
                para = d.add_paragraph(line)
                for run in para.runs:
                    run.font.size = DocxPt(12)
                para.paragraph_format.space_after = DocxPt(4)

    d.save(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppt-template", type=Path, required=True)
    ap.add_argument("--docx-template", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    build_ppt(a.ppt_template, a.outdir / "Parsimony-Review2.pptx")
    build_docx(a.docx_template, a.outdir / "Parsimony-Project-I-Report.docx")


if __name__ == "__main__":
    main()
