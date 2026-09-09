"""Fill the Review-2 PPT template with the project's content.

The template ships 13 slides of placeholders (<<...>>, "<To ...>") plus a
footer date that PowerPoint renders from an auto-updating field. The field
carries a cached string -- 22-08-2026 -- which is what a reader sees until
PowerPoint refreshes it, so the cached text is rewritten too rather than only
the runs, which was the bug in the first attempt: a datetime field has zero
runs, so run-level replacement silently did nothing.

Slide order and layouts come from the template and are not changed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "figures" / "diagrams" / "architecture.png"
REVIEW_DATE = "09-09-2026"

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

# (title, [(text, indent_level), ...]) for each of the 13 slides.
SLIDES: list[tuple[str, list[tuple[str, int]]]] = [
    # 1 -- title slide is handled separately
    ("", []),

    # 2 -- approval
    ("Approval Mail From Guide", [
        ("Approval obtained from the faculty guide for:", 0),
        ("Project title, problem statement and the six research gaps", 1),
        ("Proposed architecture: seven optimisation modules and a fidelity gate", 1),
        ("Evaluation plan: 2^4 factorial ablation over 151 conversations", 1),
        ("Outcome target: conference paper (Scopus-indexed)", 1),
        ("[ Paste the screenshot of the guide's approval mail here ]", 0),
    ]),

    # 3 -- aim
    ("Aim", [
        ("To build a fully local, CPU-only middleware layer that measurably "
         "reduces the token and time cost of using a small language model, "
         "without changing the meaning of any request.", 0),
        ("And to quantify, through a systematic factorial ablation, how the "
         "optimisation techniques interact when stacked - at a model scale and "
         "on hardware the existing literature has not evaluated.", 0),
        ("The deliverable is a calibrated operating curve, not a single "
         "headline percentage.", 0),
    ]),

    # 4 -- abstract
    ("Abstract", [
        ("Problem: prompt compression, semantic caching, KV-cache reuse, "
         "routing and output budgeting are each published in isolation, on GPU, "
         "against an uncompressed baseline. Nothing establishes what happens "
         "when they are composed on CPU.", 0),
        ("Method: 40 papers surveyed, 6 research gaps derived. Parsimony built "
         "as 7 optimisation modules + an always-on fidelity gate, where every "
         "module proposes and one orchestrator commits. Every module is "
         "independently switchable, so the system is also an experiment.", 0),
        ("Evaluation: 151 conversations, 263 requests, 17 configurations, "
         "45 adversarial pairs, 40 gold items, real model on CPU.", 0),
        ("Result: 33.9% fewer tokens, 4.06 ms overhead, zero quality "
         "regressions - and three findings the literature had not reported.", 0),
    ]),

    # 5 -- literature review (a real table replaces the template's picture)
    ("Literature Review", []),

    # 6 -- research gap
    ("Research Gap", [
        ("G1  Compositional effects are unmeasured - no study runs the "
         "techniques in one pipeline, so it is unknown whether savings add", 0),
        ("G2  The prefill/decode split is uncharacterised for CPU-only "
         "inference, and no stack contains an output-side module at all", 0),
        ("G3  The compression x cache interaction is unmeasured - compression "
         "strips exactly the words that tell similar questions apart", 0),
        ("G4  Token savings and prefix-cache survival have never been measured "
         "together; on local hardware the token proxy can invert", 0),
        ("G5  Published cache thresholds (0.85-0.92) are not adversarially "
         "validated at small model scale", 0),
        ("G6  Every cache in the literature starts empty; the user's own chat "
         "history is a free, on-distribution corpus nobody mines", 0),
    ]),

    # 7 -- objectives
    ("Objectives", [
        ("O1  Compose eight techniques in one pipeline, every module "
         "independently switchable", 0),
        ("O2  Establish the CPU prefill/decode cost structure and convert "
         "tokens into milliseconds", 0),
        ("O3  Drive the false-answer rate of cache reuse to zero without a "
         "second model", 0),
        ("O4  Guarantee no optimisation turns a correct answer into an "
         "incorrect one", 0),
        ("O5  Keep total middleware overhead under 120 ms per request", 0),
        ("O6  Make every result reproducible from raw logs by a single command", 0),
    ]),

    # 8 -- architecture (image slide)
    ("Framework / Architecture", []),

    # 9 -- functional requirements
    ("Functional Requirements", [
        ("FR1  Answer trivially answerable queries without calling the model "
         "(arithmetic, dates)", 0),
        ("FR2  Reuse a previous answer when the question is equivalent - and "
         "defend that decision against near-duplicates", 0),
        ("FR3  Select which prior turns stay relevant, and decide where to "
         "place them", 0),
        ("FR4  Compress the prompt, measured in real tokenizer tokens rather "
         "than words", 0),
        ("FR5  Refuse any edit that drops a number, name, unit, negation or "
         "qualifier", 0),
        ("FR6  Keep the prompt's invariant prefix byte-stable across turns", 0),
        ("FR7  Set an output budget by query class and stop once the answer is "
         "complete", 0),
        ("FR8  Log every decision - including decisions to do nothing - with "
         "its token and time cost", 0),
        ("Non-functional: under 120 ms overhead, no GPU, no API key, fully "
         "offline, one-command reproducibility.", 0),
    ]),

    # 10 -- modules
    ("Modules", [
        ("M1  Compressor - three tiers, sentence aware, with a guard that "
         "rejects any edit not reducing the token count", 0),
        ("M2  Semantic cache - exact + similarity tiers, three-zone policy "
         "(accept / verify / reject)", 0),
        ("M3  History manager - relevance selector and a separate arranger, so "
         "what is kept and where it goes ablate independently", 0),
        ("M4  Prefix-stable assembler - invariant content first, with a "
         "token-level prefix-survival instrument", 0),
        ("M5  Output budgeter - per-class token budget plus a streaming early "
         "stop on trigram novelty", 0),
        ("M6  Router - a deterministic tier costing zero model tokens, and a "
         "calibrated escalation tier", 0),
        ("M7  Policy learner - offline counterfactual replay of past chats into "
         "a PolicyBundle that warm-starts the cache", 0),
        ("M8  Fidelity gate - ALWAYS ON; checks every proposal from every "
         "module before it is committed", 0),
    ]),

    # 11 -- experiments and results
    ("Experiments and Results", [
        ("E1  Factorial ablation, 17 configurations, 263 requests:  33.9% total "
         "token reduction (full stack)", 0),
        ("Savings are NOT additive - 29.0 pp predicted, 27.4 pp measured. The "
         "whole shortfall is one interaction, M3xM5 = -0.70 pp", 1),
        ("E2  Latency structure on the real model:  prefill is 91.7-98.8% of "
         "inference time, ~8.5 ms per input token", 0),
        ("Two prompts 0.5% apart in tokens cost 212 ms vs 18,914 ms - about "
         "89x - depending only on prefix stability", 1),
        ("E3  Threshold sweep, 45 adversarial pairs:  no threshold is safe - "
         "false-hit rate flat at 51.1% from 0.75 to 0.99", 0),
        ("The verifier reaches 0/45 (0.0%) at 0.92, giving up 8.9 pp of genuine "
         "reuse to eliminate 51.1 pp of wrong answers", 1),
        ("E4  Quality and overhead:  gold accuracy 92.5% -> 97.5%, ZERO paired "
         "regressions; overhead 4.06 ms (budget was 120 ms)", 0),
        ("E5  Transfer:  reduction ratios replicate within 0.09 pp on a second "
         "vocabulary; warm-start gain +0.00 pp at 0% recurrence, +17.83 pp at "
         "57.5%", 0),
    ]),

    # 12 -- conclusion
    ("Conclusion", [
        ("Eight techniques were composed in one instrumented pipeline and "
         "measured, rather than assumed. 719 automated tests; every table "
         "regenerates from raw logs in ~100 seconds.", 0),
        ("Three results the literature had not reported:", 0),
        ("Savings do not compound, and the shortfall is one interaction, not a "
         "diffuse loss", 1),
        ("Token count is a good target but a poor proxy for cost on CPU - "
         "prefix stability matters roughly 89x more", 1),
        ("Similarity thresholds cannot make answer reuse safe; a non-neural "
         "verifier can, at microsecond cost", 1),
        ("The recurring pattern: published operating points are properties of "
         "the configuration they were tuned on. For this class of system the "
         "deliverable is a calibration procedure, not a number.", 0),
        ("Next (Project-II): instrumented energy measurement, per-configuration "
         "quality on the real model, replication on a second CPU.", 0),
    ]),

    # 13 -- references
    ("References", [
        ("[1] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang and L. Qiu, \"LLMLingua: "
         "Compressing prompts for accelerated inference of large language "
         "models,\" in Proc. EMNLP, 2023.", 0),
        ("[2] F. Bang, \"GPTCache: An open-source semantic cache for LLM "
         "applications,\" in Proc. NLP-OSS at EMNLP, 2023.", 0),
        ("[3] N. F. Liu et al., \"Lost in the middle: How language models use "
         "long contexts,\" Trans. ACL, vol. 12, pp. 157-173, 2024.", 0),
        ("[4] W. Kwon et al., \"Efficient memory management for large language "
         "model serving with PagedAttention,\" in Proc. SOSP, 2023.", 0),
        ("[5] L. Chen, M. Zaharia and J. Zou, \"FrugalGPT: How to use large "
         "language models while reducing cost and improving performance,\" "
         "arXiv:2305.05176, 2023.", 0),
        ("[6] \"An empirical study on prompt compression for large language "
         "models,\" in Proc. ICLR, 2025.", 0),
        ("[7] \"MeanCache: User-centric semantic caching for LLM web "
         "services,\" arXiv:2403.02694, 2024.", 0),
        ("[8] \"Enhancing adversarial resilience in semantic caching for secure "
         "retrieval-augmented generation systems,\" Scientific Reports, 2026.", 0),
        ("Full 40-reference list in the project report, Section 5.", 0),
    ]),
]

TITLE_SLIDE_HEAD = [
    ("Programme: B.Tech   |   Course Code: BCSE497J - Project I", 16, False),
    ("Token-Efficient LLM Interaction on CPU-Only Hardware", 28, True),
]
TITLE_SLIDE_BODY = [
    ("Team members:", 16, True),
    ("Alok Singh   -   23BCI0158", 16, False),
    ("Arrsh Tripathi   -   23BCI0191", 16, False),
    ("Dev Singh   -   23BCE0794", 16, False),
    ("", 10, False),
    ("Faculty guide:   Dr Sathya K", 16, True),
]

# Slide 5 ships a PICTURE of an empty five-column table inside a placeholder,
# so it cannot be filled. It is removed and a real table is drawn instead.
LIT_TABLE_HEAD = ["Paper", "Objective", "Methodology", "Pros / Cons",
                  "Our finding"]
LIT_TABLE_ROWS = [
    ["LLMLingua\n(EMNLP 2023)",
     "Cut prompt tokens",
     "Budget controller plus token-level iterative compression, scored by perplexity",
     "+ Up to 20x compression\n- GPU only; tested alone, never with a cache",
     "Weakest module in our stack: +0.23 pp of 33.9%"],
    ["GPTCache\n(NLP-OSS 2023)",
     "Reuse a past answer",
     "Embed the query, compare, serve above a similarity threshold",
     "+ Large saving on repeats\n- One threshold, no verification, never attacked",
     "E3: no threshold is safe - false hits flat at 51.1%"],
    ["Lost in the Middle\n(TACL 2024)",
     "Improve long-context accuracy",
     "Measure accuracy against the position of relevant content",
     "+ Well evidenced\n- Implies reordering, which destroys prefix reuse",
     "E2: that reordering costs about 89x in prefill"],
    ["vLLM Prefix Caching\n(2025)",
     "Skip re-reading a stable prefix",
     "Hash KV blocks and share identical prompt prefixes",
     "+ Largest latency win on CPU\n- States the rule, never prices breaking it",
     "E2: 98.5% vs 0.5% reuse; 212 ms vs 18,914 ms"],
    ["Empirical Study on Prompt Compression (ICLR 2025)",
     "Characterise compression failures",
     "Measure hallucination rates across three models",
     "+ Names two failure modes\n- Offers no run-time guard against them",
     "Our M8 gate is that guard: 51.1% to 0.0%"],
]
LIT_COL_W = [1.95, 1.75, 2.85, 3.05, 2.70]        # inches, total 12.30


def fix_dates(prs) -> None:
    """Rewrite the cached text of auto-updating DATE fields only.

    A field carries its kind in @type: "datetime*" for the footer date,
    "slidenum" for the page number. The first version rewrote every field it
    found, which stamped the date into the slide-number placeholder on all 13
    slides. PowerPoint recomputes slidenum on render so it looked right, but
    the stored value was wrong.
    """
    dates = nums = 0
    for slide in prs.slides:
        for fld in slide._element.iter(f"{A}fld"):
            kind = (fld.get("type") or "").lower()
            if kind.startswith("datetime"):
                for t in fld.iter(f"{A}t"):
                    t.text = REVIEW_DATE
                    dates += 1
            elif kind == "slidenum":
                nums += 1
    print(f"  date fields rewritten: {dates}  (slide-number fields "
          f"left alone: {nums})")


def body_placeholder(slide, title_shape):
    """The largest text frame that is not the title, date, footer or number."""
    best, best_area = None, 0
    for sh in slide.shapes:
        if sh is title_shape or not sh.has_text_frame:
            continue
        if sh.element.find(f".//{A}fld") is not None:
            continue                       # date / slide-number placeholder
        try:
            area = sh.width * sh.height
        except TypeError:
            continue
        if area > best_area:
            best, best_area = sh, area
    return best


def no_bullet(paragraph) -> None:
    """Force <a:buNone/> so the paragraph carries no bullet glyph.

    Needed where entries are already numbered. The template's first body
    paragraph is styled differently from the rest, so reusing it left
    reference [1] unbulleted and indented differently from [2] onward;
    clearing every paragraph is the consistent fix.
    """
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("buChar", "buAutoNum", "buNone"):
        for el in pPr.findall(f"{A}{tag}"):
            pPr.remove(el)
    pPr.append(pPr.makeelement(f"{A}buNone", {}))
    pPr.set("indent", "0")
    pPr.set("marL", "0")


def fill_body(shape, bullets, size=15, bulleted=True):
    tf = shape.text_frame
    tf.word_wrap = True
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    tf.paragraphs[0].clear()

    for i, (text, level) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.bold = level == 0 and len(bullets) > 4 and text.endswith(":")
        p.space_after = Pt(6)
        if not bulleted:
            no_bullet(p)


def drop(shape) -> None:
    shape._element.getparent().remove(shape._element)


def set_lines(shape, lines):
    """Write (text, pt, bold) triples into a text frame, one per paragraph."""
    tf = shape.text_frame
    tf.word_wrap = True
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    tf.paragraphs[0].clear()
    for i, (text, size, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        p.space_after = Pt(3)


def lit_table(slide) -> None:
    """Draw the five-column survey table the template only mocked up."""
    top, left = Inches(1.55), Inches(0.45)
    rows, cols = len(LIT_TABLE_ROWS) + 1, len(LIT_TABLE_HEAD)
    width = Inches(sum(LIT_COL_W))
    graphic = slide.shapes.add_table(rows, cols, left, top, width,
                                     Inches(4.9))
    table = graphic.table
    for j, w in enumerate(LIT_COL_W):
        table.columns[j].width = Inches(w)

    for j, text in enumerate(LIT_TABLE_HEAD):
        cell = table.cell(0, j)
        cell.text = text
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold = Pt(11), True

    for i, row in enumerate(LIT_TABLE_ROWS, start=1):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = text
            cell.margin_top = cell.margin_bottom = Pt(2)
            cell.margin_left = cell.margin_right = Pt(4)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
                    r.font.bold = j == 0


def fit_picture(slide, image: Path, prs, top_in: float, bottom_in: float,
                max_w_frac: float = 0.92) -> None:
    """Place an image so it fits the free band, constrained by BOTH axes.

    The first version constrained width only. The architecture figure is
    1180x760, so a 0.86-width placement came out 7.4 inches tall on a 7.5-inch
    slide and its bottom two layers fell off the page.
    """
    from PIL import Image                                   # noqa: PLC0415
    try:
        iw, ih = Image.open(image).size
    except Exception:                                        # no Pillow
        iw, ih = 1180, 760
    avail_h = Inches(bottom_in - top_in)
    avail_w = Emu(int(prs.slide_width * max_w_frac))
    scale = min(avail_w / iw, avail_h / ih)
    w, h = Emu(int(iw * scale)), Emu(int(ih * scale))
    left = Emu(int((prs.slide_width - w) / 2))
    top = Emu(int(Inches(top_in) + (avail_h - h) / 2))
    slide.shapes.add_picture(str(image), left, top, width=w, height=h)


def build(template: Path, out: Path) -> None:
    prs = Presentation(str(template))
    print(f"slides in template: {len(prs.slides)}")

    for idx, slide in enumerate(prs.slides):
        title = slide.shapes.title

        # ---- slide 1: title slide -------------------------------------
        if idx == 0:
            if title is not None:
                set_lines(title, TITLE_SLIDE_HEAD)
            body = body_placeholder(slide, title)
            if body is not None:
                set_lines(body, TITLE_SLIDE_BODY)
            print("  slide 1: title")
            continue

        heading, bullets = SLIDES[idx]
        if title is not None and heading:
            title.text_frame.text = heading
            for p in title.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(30)
                    r.font.bold = True

        body = body_placeholder(slide, title)

        # ---- slide 2: drop the template's specimen approval mail ------
        if idx == 1:
            for sh in list(slide.shapes):
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    drop(sh)
                    print("  slide 2: removed template's sample mail image")
            if body is not None:
                fill_body(body, bullets, 16)
            continue

        # ---- slide 5: replace the mocked-up table with a real one -----
        if idx == 4:
            for sh in list(slide.shapes):
                if sh is title:
                    continue
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE or sh is body:
                    drop(sh)                 # table picture + "Note:" image
            lit_table(slide)
            print("  slide 5: real 5-column table drawn")
            continue

        # ---- slide 7: stop the body above the SDG / TRL block ---------
        if idx == 6 and body is not None:
            # All four must be set: a placeholder inherits its geometry from
            # the layout, and writing only two collapses the other two to zero.
            body.left, body.width = Inches(0.92), Inches(11.5)
            body.top, body.height = Inches(1.55), Inches(3.15)
            fill_body(body, bullets, 15)
            print("  slide 7: body clipped above the SDG block")
            continue

        # ---- slide 8: architecture image, fitted on both axes ---------
        if idx == 7:
            if body is not None:
                drop(body)
            if ARCH.exists():
                fit_picture(slide, ARCH, prs, 1.45, 6.80)
                print("  slide 8: architecture image fitted")
            continue

        if body is None:
            print(f"  slide {idx + 1}: no body placeholder found")
            continue

        size = 15
        if idx == 10:             # results -- denser
            size = 13
        if idx == 12:             # references
            # The template's box is 2.12in tall and holds one sample entry;
            # nine references overflow it upward into the title. Give it the
            # full band, setting all four values (see slide 7).
            size = 12
            body.left, body.width = Inches(0.6), Inches(11.9)
            body.top, body.height = Inches(1.58), Inches(5.15)
        fill_body(body, bullets, size, bulleted=idx != 12)
        print(f"  slide {idx + 1}: {heading} ({len(bullets)} lines)")

    fix_dates(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.template, a.out)
