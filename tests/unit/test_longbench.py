"""The public-benchmark harness: its metric, its arms, and what it refuses.

The whole point of this file is that we did not write the exam, so the tests
that matter are the ones stopping us from quietly rewriting it anyway: the
metric must be LongBench's, the items must be taken in file order rather than
chosen, and the instruction around the context must not drift.
"""

from __future__ import annotations

import json

import pytest

from parsimony.eval import longbench as lb


class TestTheMetricIsTheirs:
    """Token F1 after LongBench's normalisation, maximised over the answers."""

    @pytest.mark.parametrize("prediction, answers, expected", [
        ("Paris", ["Paris"], 1.0),
        ("the Eiffel Tower", ["Eiffel Tower"], 1.0),        # articles are stripped
        ("Paris.", ["Paris"], 1.0),                          # punctuation is stripped
        ("PARIS", ["Paris"], 1.0),                           # case is folded
        ("London", ["Paris"], 0.0),
        ("Paris France", ["Paris"], 2 / 3),                  # precision 1/2, recall 1/1
        ("Paris", ["London", "Paris"], 1.0),                 # best of the references
        ("", ["Paris"], 0.0),
    ])
    def test_it_scores_as_longbench_does(self, prediction, answers, expected):
        assert lb.qa_f1(prediction, answers) == pytest.approx(expected, abs=1e-6)

    def test_a_wordy_answer_is_penalised_not_rewarded(self):
        """A model that pads its answer must not score higher for it."""
        tight = lb.qa_f1("Ozalj", ["Ozalj"])
        padded = lb.qa_f1("The wife of Francis I Rakoczi was born in Ozalj", ["Ozalj"])
        assert tight == 1.0
        assert padded < tight

    def test_no_shared_token_is_zero_not_an_error(self):
        assert lb.qa_f1("nothing at all", ["something else"]) == 0.0


class TestTheItems:
    def test_a_multi_passage_context_becomes_documents(self):
        context = "Passage 1: Alpha\nAlpha is a town.\n\nPassage 2: Beta\nBeta is a river."
        docs = lb._as_documents(context, "x")
        assert len(docs) == 2
        assert "Alpha is a town." in docs[0].content
        assert docs[1].title.startswith("Passage 2")

    def test_a_single_block_context_stays_one_document(self):
        """A single-document task must not be split into invented boundaries."""
        docs = lb._as_documents("One continuous passage with no blank lines.", "x")
        assert len(docs) == 1
        assert docs[0].content.startswith("One continuous")

    def test_items_are_taken_in_file_order(self, tmp_path):
        """Never sampled, never sorted by length: choosing items is setting the exam."""
        rows = [{"_id": f"id{i}", "input": f"q{i}?", "answers": [f"a{i}"],
                 "context": f"Passage: {i}\nbody {'word ' * (100 - i)}"} for i in range(5)]
        path = tmp_path / "demo.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        got = lb.load("demo", tmp_path, limit=3)
        assert [i.item_id for i in got] == ["id0", "id1", "id2"]

    def test_the_digest_ties_a_result_to_the_bytes_it_came_from(self, tmp_path):
        (tmp_path / "demo.jsonl").write_text('{"_id":"a","input":"q","answers":["x"],'
                                             '"context":"c"}', encoding="utf-8")
        first = lb.data_digest("demo", tmp_path)
        (tmp_path / "demo.jsonl").write_text('{"_id":"a","input":"q","answers":["y"],'
                                             '"context":"c"}', encoding="utf-8")
        assert lb.data_digest("demo", tmp_path) != first


class TestThePrompt:
    def test_the_instruction_is_longbenchs_and_surrounds_the_context(self):
        """Only the context may change. Rewriting the instruction would make
        this a different experiment from the one being cited."""
        assert "Answer the question based on the given passages" in lb.PROMPT
        assert lb.PROMPT.count("{context}") == 1 and lb.PROMPT.count("{question}") == 1

    def test_compression_changes_the_context_and_nothing_else(self):
        from parsimony.core.types import Document

        question = "Where was she born?"
        full = lb._render((Document("d0", "Alpha is a town. Beta is a river.", ""),), question)
        cut = lb._render((Document("d0", "Alpha is a town.", ""),), question)
        assert full.replace("Beta is a river.", "").split() == cut.split()


class TestWhatItRefuses:
    def test_a_prompt_that_does_not_fit_is_refused_rather_than_recorded(self, tmp_path):
        """A truncated full-context arm hands compression a win it did not earn."""
        from parsimony.core.types import GenParams
        from parsimony.infra.providers import ProviderError

        class TinyWindow:
            model_name = "tiny"
            model_digest = "tiny-digest"

            def complete(self, prompt, params: GenParams):
                return "x", {"prompt_eval_count": 40, "num_ctx": 64}

        rows = [{"_id": "a", "input": "q?", "answers": ["x"],
                 "context": "Passage: 1\n" + "word " * 400}]
        (tmp_path / "demo.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
        with pytest.raises(ProviderError, match="exceed"):
            lb.run("demo", tmp_path, TinyWindow(), tmp_path / "out.jsonl", limit=1,
                   arms=("full",))

    def test_summarise_reports_what_each_arm_sent(self):
        rows = [
            {"arm": "full", "item_id": "a", "f1": 1.0, "context_tokens": 100,
             "prefill_ms": 1000},
            {"arm": "parsimony", "item_id": "a", "f1": 1.0, "context_tokens": 25,
             "prefill_ms": 250},
        ]
        got = {r["arm"]: r for r in lb.summarise(rows)}
        assert got["full"]["context_kept_pct"] == 100.0
        assert got["parsimony"]["context_kept_pct"] == 25.0
        assert got["parsimony"]["f1"] == 100.0


class TestTheAnswerPositionSplit:
    """Separating "truncation worked" from "truncation got lucky".

    Truncation's entire strategy is "keep the front". On documents in their
    natural order that is a coin toss it wins about half the time, which is why
    it can score level with a compressor overall while doing nothing a
    compressor does. The split is made on where the answer sits in the DATA,
    computed without reference to any result.
    """

    @staticmethod
    def _item(context: str, answers: list[str]):
        from parsimony.eval.longbench import BenchItem, _as_documents

        return BenchItem(item_id="x", task="t", question="q?", answers=tuple(answers),
                         documents=_as_documents(context, "x"),
                         context_words=len(context.split()))

    def test_position_is_where_the_answer_first_appears(self):
        item = self._item("alpha " * 10 + "Ozalj " + "beta " * 80, ["Ozalj"])
        position = lb.answer_position(item)
        assert position is not None
        assert 0.05 < position < 0.15

    def test_an_answer_at_the_very_front_is_zero(self):
        assert lb.answer_position(self._item("Ozalj is a town. " + "x " * 90,
                                             ["Ozalj"])) == 0.0

    def test_an_answer_that_is_not_in_the_context_has_no_position(self):
        """Answers are not guaranteed to be extractive, and inventing a
        position for one would put it in whichever group flattered us."""
        assert lb.answer_position(self._item("nothing relevant here", ["Ozalj"])) is None

    def test_the_split_uses_the_budget_as_its_cutoff(self):
        """Below the budget truncation keeps the answer by construction, which
        is exactly the line that makes the two groups mean different things."""
        items = {
            "early": self._item("Ozalj " + "x " * 99, ["Ozalj"]),
            "late": self._item("x " * 90 + "Ozalj " + "x " * 9, ["Ozalj"]),
        }
        rows = [
            {"item_id": "early", "arm": "parsimony", "f1": 1.0, "context_tokens": 20,
             "prefill_ms": 10},
            {"item_id": "early", "arm": "full", "f1": 1.0, "context_tokens": 100,
             "prefill_ms": 50},
            {"item_id": "late", "arm": "parsimony", "f1": 1.0, "context_tokens": 20,
             "prefill_ms": 10},
            {"item_id": "late", "arm": "full", "f1": 0.0, "context_tokens": 100,
             "prefill_ms": 50},
        ]
        split = lb.by_position(rows, items, cutoff=0.20)
        assert {c["group"]: c["n"] for c in split["counts"]} == {"early": 1, "late": 1}
        late_parsimony = next(r for r in split["late"] if r["arm"] == "parsimony")
        assert late_parsimony["f1"] == 100.0

    def test_an_item_with_no_locatable_answer_goes_to_the_harder_group(self):
        """Unlocatable is not early. Putting it in the group truncation wins
        would be choosing the split to suit the result."""
        items = {"a": self._item("no answer text at all here", ["Ozalj"])}
        rows = [{"item_id": "a", "arm": "full", "f1": 0.0, "context_tokens": 10,
                 "prefill_ms": 1}]
        split = lb.by_position(rows, items)
        assert {c["group"]: c["n"] for c in split["counts"]} == {"early": 0, "late": 1}

    def test_the_split_works_from_the_rows_alone(self):
        """A results file must be re-analysable without the 110 MB it came from."""
        rows = [
            {"item_id": "a", "arm": "parsimony", "f1": 1.0, "context_tokens": 20,
             "prefill_ms": 10, "answer_position": 0.05},
            {"item_id": "b", "arm": "parsimony", "f1": 0.0, "context_tokens": 20,
             "prefill_ms": 10, "answer_position": 0.70},
        ]
        split = lb.by_position(rows)
        assert {c["group"]: c["n"] for c in split["counts"]} == {"early": 1, "late": 1}
        assert next(r for r in split["early"] if r["arm"] == "parsimony")["f1"] == 100.0
        assert next(r for r in split["late"] if r["arm"] == "parsimony")["f1"] == 0.0

    def test_a_row_without_a_recorded_position_falls_back_to_the_items(self):
        """Rows written before the field existed still analyse, given the data."""
        items = {"a": self._item("x " * 90 + "Ozalj " + "y " * 9, ["Ozalj"])}
        rows = [{"item_id": "a", "arm": "full", "f1": 0.0, "context_tokens": 10,
                 "prefill_ms": 1}]
        split = lb.by_position(rows, items)
        assert {c["group"]: c["n"] for c in split["counts"]} == {"early": 0, "late": 1}

    def test_a_recorded_position_is_preferred_over_recomputing_it(self):
        """The row is the record. Recomputing from data that may have moved on
        would let a result change without anything being re-run."""
        items = {"a": self._item("Ozalj " + "x " * 99, ["Ozalj"])}       # early
        rows = [{"item_id": "a", "arm": "full", "f1": 1.0, "context_tokens": 10,
                 "prefill_ms": 1, "answer_position": 0.9}]               # recorded late
        split = lb.by_position(rows, items)
        assert {c["group"]: c["n"] for c in split["counts"]} == {"early": 0, "late": 1}
