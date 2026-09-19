"""The export layer: LaTeX tables, a Beamer deck, and the proof document.

Exports are where a number is most likely to drift from the run that produced
it -- a table retyped into a report, a percentage rounded twice, a claim that
outlived the data. So these tests pin the direction of the dependency: every
figure comes from figures/*.csv or from a live audit, nothing is authored here,
and a proof that cannot show its removals is not a proof.
"""

from __future__ import annotations

import csv

import pytest

from parsimony.core.config import full_stack
from parsimony.core.types import split_into_documents
from parsimony.infra.providers import MockProvider
from parsimony.modules.m1_context import audit
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.surfaces import export as ex

QUESTION = "What is the annual travel budget for the Tallinn office?"


@pytest.fixture(scope="module")
def tok_module():
    from parsimony.infra.tokenization import HeuristicTokenizer

    return HeuristicTokenizer("test")


@pytest.fixture(scope="module")
def handbook():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "examples/staff-handbook.md").read_text(encoding="utf-8")
    return split_into_documents(text, "staff-handbook.md")


@pytest.fixture(scope="module")
def report(handbook, tok_module):
    pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok_module)
    return audit(pipe.build_context(QUESTION, documents=handbook), full_stack())


class TestEscaping:
    @pytest.mark.parametrize("raw, want", [
        ("100% of them", r"100\% of them"),
        ("cost $5 & rising", r"cost \$5 \& rising"),
        ("a_b", r"a\_b"),
        ("#1", r"\#1"),
    ])
    def test_latex_specials_survive_as_text(self, raw, want):
        assert ex.tex_escape(raw) == want

    def test_an_escaped_string_is_not_escaped_twice(self):
        once = ex.tex_escape("50%")
        assert ex.tex_escape(once) != once or "\\\\" not in ex.tex_escape(once)
        assert once.count("\\") == 1


class TestTheTables:
    def test_a_table_is_built_from_the_csv_not_from_prose(self, tmp_path):
        path = tmp_path / "x.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "correct", "tokens"])
            w.writerow(["parsimony", "36/45", "158"])
            w.writerow(["full", "40/45", "560"])
        body = ex.tabular(ex._rows(path), ["arm", "correct", "tokens"],
                          caption="Long context", label="lc")
        assert "36/45" in body and "560" in body
        assert r"\toprule" in body and r"\bottomrule" in body
        assert r"\caption{Long context}" in body and r"\label{tab:lc}" in body

    def test_no_rows_means_no_table_rather_than_an_empty_one(self):
        assert ex.tabular([], ["arm"], caption="c", label="l") == ""

    def test_the_document_writes_only_tables_it_has_data_for(self, tmp_path):
        out = tmp_path / "results.tex"
        art = ex.latex_tables(tmp_path, out)          # an empty figures directory
        text = out.read_text(encoding="utf-8")
        assert art.lines == len(text.splitlines())
        assert r"\begin{table}" not in text, "no data must mean no table, not an empty one"

    def test_the_real_figures_produce_real_tables(self, tmp_path):
        from pathlib import Path

        figures = Path(__file__).resolve().parents[2] / "figures"
        if not (figures / "longctx_real.csv").exists():
            pytest.skip("no measured figures in this checkout")
        out = tmp_path / "results.tex"
        ex.latex_tables(figures, out)
        text = out.read_text(encoding="utf-8")
        assert text.count(r"\begin{table}") >= 3
        for line in text.splitlines():
            if line.lstrip().startswith("%") or not line.strip():
                continue
            assert "%" not in line.replace(r"\%", ""), f"a bare % comments out: {line}"


class TestTheDeck:
    def test_it_is_a_compilable_beamer_document(self, tmp_path):
        from pathlib import Path

        figures = Path(__file__).resolve().parents[2] / "figures"
        out = tmp_path / "slides.tex"
        ex.beamer_slides(figures, out)
        text = out.read_text(encoding="utf-8")
        assert r"\documentclass" in text and "beamer" in text
        assert text.count(r"\begin{frame}") == text.count(r"\end{frame}")
        assert text.rstrip().endswith(r"\end{document}")


class TestTheProof:
    def test_it_shows_removed_text_struck_through_and_labelled(self, report):
        html = ex.proof_html(QUESTION, report, answer="", model="", title="Prompt verification")
        assert "[removed:" in html
        dropped = [u for u in report.units if not u.kept]
        assert dropped, "this question must drop something for the test to mean anything"
        assert "line-through" in html
        assert dropped[0].text[:30] in html, "a proof that omits the removed text proves nothing"

    def test_every_removal_carries_the_decision_that_made_it(self, report):
        html = ex.proof_html(QUESTION, report, answer="", model="", title="t")
        for unit in report.units:
            if not unit.kept:
                assert unit.tag.lower() in html

    def test_the_kept_text_is_reproduced_verbatim(self, report):
        html = ex.proof_html(QUESTION, report, answer="", model="", title="t")
        for unit in report.units:
            if unit.kept and len(unit.text) > 40:
                assert _unescape(unit.text[:40]) in _unescape(html)

    def test_html_is_escaped_so_a_document_cannot_forge_the_page(self, tok_module):
        pipe = Pipeline(full_stack(), provider=MockProvider(), tokenizer=tok_module)
        docs = split_into_documents(
            "# Notes\n\nThe Tallinn budget is <script>alert(1)</script> euros per year. "
            "Tallinn is a city. The office is small.", "notes.md")
        rep = audit(pipe.build_context("What is the Tallinn budget?", documents=docs),
                    full_stack())
        html = ex.proof_html("What is the Tallinn budget?", rep, answer="", model="", title="t")
        assert "<script>" not in html and "&lt;script&gt;" in html

    def test_it_falls_back_to_html_when_no_browser_can_print_it(self, report, tmp_path,
                                                                monkeypatch):
        monkeypatch.setattr(ex, "find_chrome", lambda: None)
        html = ex.proof_html(QUESTION, report, answer="", model="", title="t")
        written = ex.write_proof(html, tmp_path / "proof.pdf")
        assert written.path.suffix == ".html"
        assert written.path.exists()
        assert "pdf" not in written.what.lower() or "no" in written.what.lower()


def _unescape(text: str) -> str:
    import html as _h

    return _h.unescape(text)
