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
