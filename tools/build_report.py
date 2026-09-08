"""Build a SUBMITTABLE Project-I report.

The first attempt opened the school's template and appended content after it.
That was wrong. The template is a specimen: its body is placeholders and
formatting notes — "<Title of Project>", "(Times New Roman 16, Bold, Upper
Case, Line spacing 1.5)", "/****** Sample*******/", "One page and not
exceeding 300 words". Appending left 278 paragraphs of that in front of the
real report, so page one of the submission read "<Title of Project>".

This opens the template for its PAGE SETUP and styles, clears the body, and
writes the report to the formatting the template itself specifies:

    section headings   Times New Roman 14, bold, upper case, 1.5 spacing
    sub-headings       Times New Roman 13, bold, title case, 1.5 spacing
    body               Times New Roman 12, 1.15 spacing

so the submitted document matches the school's requirements without carrying
the instructions that describe them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt

from report_content import (
    ABSTRACT, GUIDE, REFERENCES, SECTIONS, SPECIALISATION, TEAM, TITLE,
)

FONT = "Times New Roman"


def clear_body(doc: docx.Document) -> None:
    """Remove every paragraph and table, keeping sections and styles."""
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


def heading(doc, text, level=1):
    """Section and sub-section headings, to the template's own specification."""
    if level == 1:
        return para(doc, text, size=14, bold=True, upper=True, spacing=1.5,
                    before=18, after=8)
    return para(doc, text, size=13, bold=True, spacing=1.5, before=12, after=6)


def title_page(doc) -> None:
    C = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "BCSE497J - Project-I", size=13, bold=True, align=C, spacing=1.5, before=36)
    para(doc, TITLE, size=16, bold=True, upper=True, align=C, spacing=1.5, before=24, after=24)

    para(doc, "Submitted in partial fulfilment of the requirements for the degree of",
         size=12, align=C, spacing=1.5)
    para(doc, "B.Tech.", size=13, bold=True, align=C, spacing=1.5)
    para(doc, "in", size=12, align=C, spacing=1.5)
    para(doc, "Computer Science and Engineering", size=13, bold=True, align=C, spacing=1.5)
    if SPECIALISATION:
        para(doc, f"({SPECIALISATION})", size=12, align=C, spacing=1.5, after=24)

    para(doc, "by", size=12, align=C, spacing=1.5)
    for name, reg in TEAM:                      # sorted on register number
        para(doc, f"{name}   ({reg})", size=13, bold=True, align=C, spacing=1.5, after=2)

    para(doc, "Under the Supervision of", size=12, align=C, spacing=1.5, before=18)
    para(doc, GUIDE, size=13, bold=True, align=C, spacing=1.5)
    para(doc, "School of Computer Science and Engineering", size=12, align=C, spacing=1.5,
         before=18)
    para(doc, "Vellore Institute of Technology, Vellore", size=12, align=C, spacing=1.5)
    para(doc, "September 2026", size=12, bold=True, align=C, spacing=1.5, before=18)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def contents_page(doc) -> None:
    heading(doc, "Table of Contents")
    rows = [("Abstract", "i")]
    n = 1
    for sec, subs in SECTIONS:
        rows.append((f"{n}. {sec}", str(n)))
        for sub, _ in subs:
            rows.append((f"    {sub}", ""))
        n += 1
    rows.append((f"{n}. References", str(n)))

    table = doc.add_table(rows=len(rows), cols=2)
    for i, (label, page) in enumerate(rows):
        for j, text in enumerate((label, page)):
            cell = table.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.name = FONT
            run.font.size = Pt(12)
            run.bold = label[0].isdigit() and j == 0
            cell.paragraphs[0].paragraph_format.line_spacing = 1.5
            if j == 1:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    table.style = "Table Grid"
    for row in table.rows:                       # the template says: remove borders
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for border in tcPr.findall(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tcBorders"):
                tcPr.remove(border)
    table.style = None
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def build(template: Path, out: Path) -> None:
    doc = docx.Document(str(template))
    clear_body(doc)

    title_page(doc)

    heading(doc, "Abstract")
    for block in ABSTRACT:
        para(doc, block, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    contents_page(doc)

    for n, (section, subs) in enumerate(SECTIONS, start=1):
        heading(doc, f"{n}. {section}")
        for sub, blocks in subs:
            if sub:
                heading(doc, sub, level=2)
            for block in blocks:
                if isinstance(block, tuple):     # (rows, header) -> a table
                    add_table(doc, block[0], block[1])
                elif block.startswith("• "):
                    p = para(doc, block[2:], align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=3)
                    p.paragraph_format.left_indent = Pt(18)
                else:
                    para(doc, block, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    heading(doc, f"{len(SECTIONS) + 1}. References")
    for ref in REFERENCES:
        p = para(doc, ref, size=12, after=4)
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-24)

    doc.save(str(out))
    print(f"wrote {out}")


def add_table(doc, rows, header):
    table = doc.add_table(rows=len(rows) + 1, cols=len(header))
    table.style = "Table Grid"
    for j, text in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.font.name, run.font.size, run.bold = FONT, Pt(11), True
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.name, run.font.size = FONT, Pt(10.5)
    doc.add_paragraph()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    build(ap.parse_args().template, ap.parse_args().out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.template, a.out)
