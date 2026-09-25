"""M1's context tier and the EXTRACT gate (ADR-040).

The tier deletes most of a long context, so every guarantee it relies on is
tested from the side that would be hurt if it broke: the answer sentence must
survive, a sentence must never be reworded, what the question names must stay
findable, and a question about one document must never be answered from the
cache entry of another.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from parsimony.core.config import baseline, full_stack
from parsimony.core.proposals import ContextPatch, NoOp, TransformKind
from parsimony.core.types import Document, Turn
from parsimony.eval.longctx import (
    Methods,
    evidence_kept,
    load_longctx,
    run_offline,
)
from parsimony.eval.stats import mcnemar_exact, wilson_interval
from parsimony.infra.providers import MockProvider
from parsimony.modules.m1_context import ContextCompressor, ranking_terms
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.modules.m8_fidelity import is_sentence_extract

ITEMS = {i.item_id: i for i in load_longctx()}
HARLOW = ITEMS["harlow_q3"].documents        # six documents, ~780 tokens


def propose(pipeline, question, documents=(), history=(), cfg=None):
    ctx = pipeline.build_context(question, history, documents=documents)
    return ctx, ContextCompressor().propose(ctx, cfg or full_stack())


def commit(pipeline, ctx, proposal):
    candidate = replace(ctx, **dict(proposal.fields))
    return candidate, pipeline.gate.check(ctx, candidate, proposal.kind, "M1")


def context_text(ctx) -> str:
    return "\n".join(d.content for d in ctx.documents)


# ------------------------------------------------------------------ gate --
class TestSentenceExtractCheck:
    SOURCE = "The office opened in 2019. It employs 58 people. Staff work four days."

    def test_deleting_whole_sentences_passes(self):
        assert is_sentence_extract(self.SOURCE, "It employs 58 people.")
        assert is_sentence_extract(self.SOURCE, "The office opened in 2019. Staff work four days.")

    def test_whitespace_between_kept_sentences_is_free(self):
        assert is_sentence_extract(self.SOURCE, "The office opened in 2019.\nIt employs 58 people.")

    def test_rewording_a_sentence_fails(self):
        assert not is_sentence_extract(self.SOURCE, "It employs 59 people.")
        assert not is_sentence_extract(self.SOURCE, "It employs 58.")

    def test_reordering_sentences_fails(self):
        assert not is_sentence_extract(self.SOURCE, "It employs 58 people. The office opened in 2019.")

    def test_a_partial_sentence_fails(self):
        assert not is_sentence_extract(self.SOURCE, "The office opened")


class TestExtractGate:
    DOCS = (Document("d1", "The Tallinn office opened in 2019. It employs 58 people.", "Tallinn"),
            Document("d2", "The Porto office employs 41 people. It has a terrace.", "Porto"))

    def ctx(self, pipeline, question="How many people work in Tallinn?"):
        return pipeline.build_context(question, documents=self.DOCS)

    def check(self, pipeline, before, **fields):
        return pipeline.gate.check(before, replace(before, **fields), TransformKind.EXTRACT, "M1")

    def test_dropping_an_irrelevant_document_passes(self, pipeline):
        before = self.ctx(pipeline)
        assert self.check(pipeline, before, documents=(self.DOCS[0],)).passed

    def test_altering_the_question_fails(self, pipeline):
        before = self.ctx(pipeline)
        assert not self.check(pipeline, before, query="How many people?").passed

    def test_introducing_a_document_fails(self, pipeline):
        before = self.ctx(pipeline)
        extra = Document("d9", "The Tallinn office employs 900 people.", "Tallinn")
        assert not self.check(pipeline, before, documents=(*self.DOCS, extra)).passed

    def test_rewording_inside_a_document_fails(self, pipeline):
        before = self.ctx(pipeline)
        bent = replace(self.DOCS[0], content="The Tallinn office opened in 2019. It employs 85 people.")
        verdict = self.check(pipeline, before, documents=(bent, self.DOCS[1]))
        assert not verdict.passed and "altered a sentence" in verdict.detail

    def test_changing_a_title_fails(self, pipeline):
        before = self.ctx(pipeline)
        renamed = replace(self.DOCS[0], title="Porto")
        assert not self.check(pipeline, before, documents=(renamed, self.DOCS[1])).passed

    def test_losing_every_mention_of_what_the_question_names_fails(self, pipeline):
        before = self.ctx(pipeline)
        verdict = self.check(pipeline, before, documents=(self.DOCS[1],))
        assert not verdict.passed
        assert verdict.events[0].invariant_class == "entity"
        assert "Tallinn" in verdict.events[0].lost_values

    def test_a_name_the_context_never_had_is_not_required(self, pipeline):
        before = self.ctx(pipeline, "How many people work in Tallinn and in Oslo?")
        assert self.check(pipeline, before, documents=(self.DOCS[0],)).passed

    def test_removing_all_context_fails(self, pipeline):
        before = self.ctx(pipeline, "Summarise these.")
        assert not self.check(pipeline, before, documents=()).passed

    def test_turns_may_lose_sentences_but_not_be_dropped_or_reordered(self, pipeline):
        history = (Turn("t1", "assistant", "Leeds has 112 staff. Porto has 41 staff."),
                   Turn("t2", "user", "Thanks."))
        before = pipeline.build_context("What about Leeds?", history)
        shorter = (replace(history[0], content="Leeds has 112 staff."), history[1])
        assert self.check(pipeline, before, history=shorter).passed
        assert not self.check(pipeline, before, history=(history[1],)).passed
        assert not self.check(pipeline, before, history=(history[1], history[0])).passed

    def test_select_may_not_touch_documents(self, pipeline):
        before = self.ctx(pipeline)
        after = replace(before, documents=(self.DOCS[0],))
        assert not pipeline.gate.check(before, after, TransformKind.SELECT, "M3").passed


# ---------------------------------------------------------------- module --
class TestContextCompressor:
    def test_short_context_is_left_alone(self, pipeline):
        _, proposal = propose(pipeline, "How many people work there?",
                              (Document("d", "The office employs 58 people."),))
        assert isinstance(proposal, NoOp) and proposal.reason == "not_applicable"

    def test_disabled_by_config_or_with_m1_off(self, pipeline):
        ctx = pipeline.build_context("How many?", documents=HARLOW)
        off = replace(full_stack(), compression=replace(full_stack().compression,
                                                        context_enabled=False))
        assert not ContextCompressor().applies_to(ctx, off)
        assert not ContextCompressor().applies_to(ctx, baseline())

    def test_long_context_is_cut_and_the_gate_accepts_it(self, pipeline):
        ctx, proposal = propose(pipeline, "How many people does the Tallinn office employ?", HARLOW)
        assert isinstance(proposal, ContextPatch) and proposal.kind is TransformKind.EXTRACT
        after, verdict = commit(pipeline, ctx, proposal)
        assert verdict.passed
        assert after.query == ctx.query
        before_n, after_n = (pipeline.tokenizer.count(context_text(c)) for c in (ctx, after))
        assert after_n < 0.5 * before_n

    def test_the_answer_sentence_survives_even_though_it_never_names_the_office(self, pipeline):
        """'It employs 58 people.' shares no name with the question. It is found
        because it inherits 'Tallinn' from the sentence before it, and that
        sentence is kept so the model can tell what 'It' is."""
        ctx, proposal = propose(pipeline, "How many people does the Tallinn office employ?", HARLOW)
        after, _ = commit(pipeline, ctx, proposal)
        text = context_text(after)
        assert "It employs 58 people" in text
        assert "The Tallinn office opened under the Harlow name in March 2019." in text

    def test_distractors_for_other_entities_do_not_crowd_out_the_answer(self, pipeline):
        ctx, proposal = propose(pipeline, "What is the annual travel budget for the Tallinn office?",
                                HARLOW)
        after, _ = commit(pipeline, ctx, proposal)
        assert "The office has a travel budget of €39,000 per year." in context_text(after)

    def test_kept_sentences_stay_in_document_order(self, pipeline):
        ctx, proposal = propose(pipeline, "Which office has more employees, Porto or Tallinn?", HARLOW)
        after, _ = commit(pipeline, ctx, proposal)
        for doc in after.documents:
            source = next(d for d in HARLOW if d.doc_id == doc.doc_id)
            assert is_sentence_extract(source.content, doc.content)

    def test_evidence_records_what_was_decided(self, pipeline):
        _, proposal = propose(pipeline, "How many people does the Tallinn office employ?", HARLOW)
        e = proposal.evidence
        assert e["sentences_kept"] < e["sentences"]
        assert e["context_tokens_after"] < e["context_tokens_before"]
        assert "Tallinn" in e["anchors"]
        assert e["stopped_by"] in {"budget", "relevance floor", "exhausted"}

    def test_the_most_recent_exchange_is_never_compressed(self, pipeline):
        long_answer = " ".join(f"Point {i} concerns the Leeds calibration lab schedule." for i in range(40))
        history = (Turn("u1", "user", "Tell me about Leeds."),
                   Turn("a1", "assistant", long_answer),
                   Turn("u2", "user", "And Porto?"),
                   Turn("a2", "assistant", long_answer))
        ctx = pipeline.build_context("Which point mentions 7?", history)
        from parsimony.modules.m1_context import eligible_turns
        assert eligible_turns(ctx, full_stack()) == [1]

    def test_a_compressed_turn_is_never_emptied(self, pipeline):
        filler = " ".join(f"Item {i} is about gardening tools and soil." for i in range(60))
        history = (Turn("a0", "assistant", filler), Turn("u1", "user", "ok"),
                   Turn("a1", "assistant", "Fine."))
        ctx, proposal = propose(pipeline, "What is the capital of Peru?", history=history)
        if isinstance(proposal, ContextPatch):
            after, verdict = commit(pipeline, ctx, proposal)
            assert verdict.passed and after.history[0].content.strip()

    def test_inflections_meet_in_ranking(self):
        assert ranking_terms("employees") == ranking_terms("employs") == ranking_terms("employ")
        assert ranking_terms("station") != ranking_terms("statistics")


# -------------------------------------------------------------- pipeline --
class TestDocumentsThroughThePipeline:
    def test_documents_sit_between_history_and_question(self, make_pipeline):
        docs = (Document("d1", "The fee is 20 pounds.", "Fees"),)
        history = (Turn("u", "user", "What do you cover?"),
                   Turn("a", "assistant", "Membership and fees."))
        out = make_pipeline(full_stack()).run("What is the fee?", history, documents=docs)
        prompt = out.ctx.assembled.full_text
        assert prompt.index("Membership and fees.") < prompt.index("Context:") \
            < prompt.index("User: What is the fee?")
        assert "[1] Fees" in prompt

    def test_the_same_question_about_different_documents_is_not_a_cache_hit(self, make_pipeline):
        pipe = make_pipeline(full_stack())
        a = (Document("c1", "The notice period is one month.", "Contract A"),)
        b = (Document("c2", "The notice period is three months.", "Contract B"),)
        pipe.run("What is the notice period?", documents=a)
        second = pipe.run("What is the notice period?", documents=b)
        assert second.generated
        again = pipe.run("What is the notice period?", documents=a)
        assert not again.generated

    def test_long_documents_reach_the_model_shorter(self, make_pipeline):
        item = ITEMS["kestrel_q1"]
        on = make_pipeline(full_stack()).run(item.question, documents=item.documents)
        off = make_pipeline(baseline()).run(item.question, documents=item.documents)
        assert on.row.tokens_in_final < 0.6 * off.row.tokens_in_final
        assert "failed for 22 minutes" in "\n".join(d.content for d in on.ctx.documents)


# ------------------------------------------------------------- benchmark --
class TestLongContextBenchmark:
    def test_corpus_shape(self):
        items = list(ITEMS.values())
        assert len(items) == 124
        assert sum(i.split == "test" for i in items) == 45
        # A second held-out split, authored before any change made in response
        # to the first real-model run, so a later improvement can be confirmed
        # on questions nothing was tuned against.
        assert sum(i.split == "test2" for i in items) == 30
        # And questions the documents cannot answer at all (ADR-042).
        assert sum(i.split == "offtopic" for i in items) == 12
        assert sum(i.split == "offtopic_dev" for i in items) == 8
        # Questions whose answer sentence never names the entity its heading
        # names, and the same facts asked in paraphrase (ADR-044). Both are
        # probes: reported on their own, never added to a headline.
        assert sum(i.split == "sections" for i in items) == 13
        assert sum(i.split == "sections2" for i in items) == 6
        assert {i.kind for i in items} == {"lookup", "distractor", "anaphora", "negation",
                                           "two_hop", "off_topic", "section", "paraphrase"}
        for i in items:
            assert len(i.documents) == 6
            assert evidence_kept(i, i.documents) == (len(i.evidence), len(i.evidence))

    def test_no_off_topic_answer_is_hiding_in_its_own_documents(self):
        """Word answers only: a numeric answer like 0 or 12 appears somewhere in
        any six documents, and that says nothing about whether they answer the
        question."""
        import re

        for item in ITEMS.values():
            if not item.split.startswith("offtopic"):
                continue
            answer = item.gold.gold_answer
            if not answer.isalpha() or len(answer) < 3:
                continue
            body = " ".join(d.content for d in item.documents)
            assert not re.search(rf"{re.escape(answer)}", body, re.I), item.item_id

    def test_the_answerable_splits_share_no_collection(self):
        """dev/test/test2 are disjoint so tuning cannot leak into a result. The
        off-topic splits deliberately reuse those collections: the point is a
        question those very documents cannot answer."""
        by_split: dict[str, set[str]] = {}
        for item in ITEMS.values():
            by_split.setdefault(item.split, set()).add(item.collection)
        assert set(by_split) == {"dev", "test", "test2", "offtopic", "offtopic_dev",
                                 "sections", "sections2"}
        answerable = [c for split in ("dev", "test", "test2") for c in by_split[split]]
        assert len(answerable) == len(set(answerable))
        # The phenomenon splits deliberately reuse those collections. What each
        # measures is a property of the QUESTION -- how much of it the context
        # contains (offtopic), or whether it uses the documents' own vocabulary
        # (sections, sections2) -- so the documents carry no signal to leak.
        # They are reported separately and never folded into a headline, because
        # a split authored to contain a phenomenon can only demonstrate it.
        probes = [i for i in ITEMS.values()
                  if i.split.startswith(("offtopic", "sections"))]
        questions = [i.question for i in probes]
        assert len(questions) == len(set(questions))
        # No probe may restate a question the reported splits already ask.
        reported = {i.question for i in ITEMS.values()
                    if i.split in ("dev", "test", "test2")}
        assert not (set(questions) & reported)

    def test_baselines_respect_the_matched_budget(self, tok):
        m = Methods(tokenizer=tok)
        item = ITEMS["castellan_q1"]
        budget = m.context_tokens(m.parsimony(item).documents)
        for arm in ("truncate", "random", "bm25_topk"):
            got = getattr(m, arm)(item, budget)
            assert m.context_tokens(got.documents) <= budget + max(
                tok.count(s) for d in item.documents for s in d.content.split(". "))

    def test_closed_book_and_full_bracket_every_method(self, tok):
        items = [i for i in ITEMS.values() if i.collection == "vallin"]
        rows = run_offline(Methods(tokenizer=tok), items, arms=("closed_book", "full", "parsimony"))
        by = {(r.item_id, r.arm): r for r in rows}
        for i in items:
            assert by[(i.item_id, "closed_book")].context_tokens == 0
            assert by[(i.item_id, "full")].evidence_complete
            assert by[(i.item_id, "parsimony")].context_tokens < by[(i.item_id, "full")].context_tokens


class TestPairedStatistics:
    def test_wilson_stays_inside_the_unit_interval(self):
        for k in (0, 1, 44, 45):
            iv = wilson_interval(k, 45)
            assert 0.0 <= iv.low <= iv.point <= iv.high <= 100.0

    @pytest.mark.parametrize("b,c,p", [(0, 0, 1.0), (0, 2, 0.5), (2, 0, 0.5),
                                       (0, 6, 0.03125), (3, 3, 1.0)])
    def test_exact_mcnemar(self, b, c, p):
        assert mcnemar_exact(b, c) == pytest.approx(p)


# ------------------------------------------------------------ attachments --
class TestAttachingFiles:
    def test_headings_become_sections(self):
        from parsimony.core.types import split_into_documents

        text = "# Handbook\n\nIntro line.\n\n## Leave\n\nTwenty days.\n\n## Travel\n\nBook early."
        docs = split_into_documents(text, "h.md")
        assert [d.title for d in docs] == ["Handbook", "Leave", "Travel"]
        assert docs[1].content == "Twenty days."

    def test_plain_text_is_chunked_by_paragraph(self):
        from parsimony.core.types import split_into_documents

        para = " ".join(["word"] * 100)
        docs = split_into_documents("\n\n".join([para] * 4), "notes.txt", max_words=160)
        assert len(docs) == 4
        assert all(d.content == para for d in docs)

    def test_a_short_file_is_one_document_named_after_it(self):
        from parsimony.core.types import split_into_documents

        docs = split_into_documents("Just one line.", "tiny.txt")
        assert len(docs) == 1 and docs[0].title == "tiny.txt"

    def test_chat_answers_about_an_attached_file(self, tmp_path):
        from typer.testing import CliRunner

        from parsimony.surfaces.cli.main import app

        handbook = Path(__file__).resolve().parents[2] / "examples" / "staff-handbook.md"
        result = CliRunner().invoke(app, ["chat", "How many people does the Tallinn office employ?",
                                          "--file", str(handbook)])
        assert result.exit_code == 0, result.output
        # The live view: the prompt bar, the layer that did the work, and the
        # text that actually reached the model.
        assert "Prompt" in result.output and "removed" in result.output
        assert "Context selector" in result.output
        assert "What the AI actually received" in result.output
        assert "It employs 58 people" in result.output

    def test_a_missing_file_is_refused(self, tmp_path):
        from typer.testing import CliRunner

        from parsimony.surfaces.cli.main import app

        result = CliRunner().invoke(app, ["chat", "Anything?", "--file",
                                          str(tmp_path / "nope.txt")])
        assert result.exit_code != 0
        assert "cannot find" in result.output

    def test_longctx_show_marks_kept_and_removed_sentences(self):
        from typer.testing import CliRunner

        from parsimony.surfaces.cli.main import app

        result = CliRunner().invoke(app, ["longctx", "--show", "harlow_q3"])
        assert result.exit_code == 0, result.output
        # Marginalia: every sentence carries the decision taken on it, and the
        # tags name real branches of the selector rather than prose.
        assert "[KEEP: " in result.output and "[DROP: " in result.output
        assert "ANCHOR" in result.output and "FLOOR" in result.output
        assert "context tokens" in result.output
        assert "relevance 0." in result.output, "a decision without its number is an assertion"


class TestNothingRelevantIsNotKeptAnyway:
    """Relative relevance cannot tell "the best of six relevant sentences" from
    "the least irrelevant of sixty", so ~30% of an attached handbook survived a
    question it says nothing about. An absolute check now runs first (ADR-042).
    """

    OFF_TOPIC = "What is the capital of Peru?"

    def test_an_off_topic_question_keeps_one_sentence(self, pipeline):
        item = ITEMS["harlow_q1"]
        ctx, proposal = propose(pipeline, self.OFF_TOPIC, item.documents)
        after, verdict = commit(pipeline, ctx, proposal)
        assert verdict.passed
        kept = [s for d in after.documents for s in d.content.split(". ")]
        assert len(after.documents) == 1 and len(kept) == 1
        assert pipeline.tokenizer.count(context_text(after)) < 0.1 * pipeline.tokenizer.count(
            context_text(ctx))

    def test_it_says_so_in_its_evidence(self, pipeline):
        item = ITEMS["harlow_q1"]
        _, proposal = propose(pipeline, self.OFF_TOPIC, item.documents)
        assert proposal.evidence["off_topic"] is True
        assert proposal.evidence["topical_coverage"] < 0.5
        assert "bears on the question" in proposal.evidence["stopped_by"]

    def test_an_on_topic_question_is_untouched_by_the_check(self, pipeline):
        item = ITEMS["harlow_q3"]
        _, proposal = propose(pipeline, item.question, item.documents)
        assert proposal.evidence["off_topic"] is False
        assert proposal.evidence["topical_coverage"] >= 0.5
        assert proposal.evidence["sentences_kept"] > 1

    def test_the_whole_off_topic_split_collapses(self, tok):
        methods = Methods(tokenizer=tok)
        items = [i for i in ITEMS.values() if i.split == "offtopic"]
        kept = sum(methods.context_tokens(methods.parsimony(i).documents) for i in items)
        full = sum(methods.context_tokens(i.documents) for i in items)
        assert kept / full < 0.10, "an unanswerable question should not cost a fifth of the prompt"

    def test_and_the_answerable_splits_do_not(self, tok):
        methods = Methods(tokenizer=tok)
        for split in ("test", "test2"):
            items = [i for i in ITEMS.values() if i.split == split]
            complete = sum(evidence_kept(i, methods.parsimony(i).documents)[0]
                           == len(i.evidence) for i in items)
            assert complete / len(items) > 0.85


class TestSectionAnchors:
    """A sentence under "Porto office" is about Porto whether or not it says so.

    The failure this pins was a confident wrong answer, not a refusal: asked who
    manages the Porto office, the model named the LEEDS manager, because the
    sentence introducing her is the one that spells "Porto" (ADR-044).
    """

    HANDBOOK = "examples/staff-handbook.md"

    def _audit(self, question, *, sections, tok):
        from dataclasses import replace
        from pathlib import Path

        from parsimony.core.types import split_into_documents
        from parsimony.modules.m1_context import audit

        text = (Path(__file__).resolve().parents[2] / self.HANDBOOK).read_text(encoding="utf-8")
        docs = split_into_documents(text, "staff-handbook.md")
        cfg = full_stack()
        cfg = replace(cfg, compression=replace(cfg.compression,
                                               context_section_anchors=sections))
        pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
        return audit(pipe.build_context(question, documents=docs), cfg)

    def test_a_paraphrased_question_still_reaches_the_answer(self, tok):
        report = self._audit("Who manages the Porto office?", sections=True, tok=tok)
        answer = next(u for u in report.units if "Aguiar" in u.text)
        assert answer.kept, "the sentence naming the Porto manager must survive"

    def test_without_it_the_distractor_outranks_the_answer(self, tok):
        """The bug, pinned. If this ever passes, the flag has stopped mattering."""
        report = self._audit("Who manages the Porto office?", sections=False, tok=tok)
        answer = next(u for u in report.units if "Aguiar" in u.text)
        leeds = next(u for u in report.units if "Priya" in u.text)
        assert not answer.kept and leeds.kept
        assert leeds.score > answer.score

    def test_an_anchor_is_inherited_only_inside_its_own_section(self, tok):
        report = self._audit("Who manages the Porto office?", sections=True, tok=tok)
        porto = [u for u in report.units if u.source_label == "Porto office"]
        assert porto, "the handbook must still be split into titled sections"
        # Sections the question does not name must not all become anchored.
        tallinn = [u for u in report.units if u.source_label == "Tallinn office"]
        assert not all(u.tag == "ANCHOR" for u in tallinn)

    def test_it_changes_nothing_when_no_heading_names_an_anchor(self, tok):
        from parsimony.core.types import split_into_documents

        text = ("# Notes\n\nThe deadline is 15 March. Tallinn is cold in winter. "
                "The team is small. Coffee is provided.\n")
        docs = split_into_documents(text, "notes.md")
        from parsimony.modules.m1_context import audit

        seen = []
        for on in (False, True):
            cfg = full_stack()
            cfg = replace(cfg, compression=replace(cfg.compression,
                                                   context_section_anchors=on))
            pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
            report = audit(pipe.build_context("When is the deadline?", documents=docs), cfg)
            seen.append([(u.text, u.kept) for u in report.units])
        assert seen[0] == seen[1]


class TestTheFloorReadOffTheDistribution:
    """ADR-051. The constant was measured to be wrong in a way no constant can
    fix: 0.15 suits a peaked score profile and refuses the flat band a
    multi-hop question produces. These pin the rule's shape, not its results --
    the results live in the evaluation, where they can be re-measured.
    """

    @staticmethod
    def _units(scores, tokens=10):
        from parsimony.modules.m1_context import Unit

        return [Unit(("doc", 0), i, i, f"sentence {i}", tokens, ("w",))
                for i, _ in enumerate(scores)]

    def _read(self, scores, **over):
        from parsimony.modules.m1_context import elbow_floor

        c = full_stack().compression
        if over:
            c = replace(c, **over)
        return elbow_floor(list(scores), self._units(scores), 10 * len(scores), c)

    def test_a_collapse_into_noise_is_a_cliff_however_small_the_difference(self):
        """The rule that was tried first measured the gap as a subtraction, and
        this is the profile that showed it was the wrong reading: the drop is
        only 0.19, and it is obviously where the evidence stops."""
        reading = self._read([1.0, 0.29, 0.22, 0.03, 0.03, 0.02, 0.02, 0.01])
        assert reading.rank == 2 and reading.fall > 7
        assert 0.03 < reading.value < 0.22

    def test_a_flat_band_of_strong_sentences_is_not_a_cliff(self):
        """...and its largest SUBTRACTIVE gap, 0.25, is bigger than the one
        above. Under the old reading this profile was cut and that one was not,
        which is exactly backwards."""
        reading = self._read([1.0, 0.94, 0.89, 0.73, 0.72, 0.69, 0.68, 0.65])
        assert reading.rank == -1
        assert reading.value == full_stack().compression.context_relevance_floor

    def test_no_cliff_falls_back_to_the_constant_rather_than_to_permissiveness(self):
        """The first version dropped to the minimum when it could not read a
        shape, which made "I do not know" the most expensive branch and the
        commonest one."""
        c = full_stack().compression
        reading = self._read([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        assert reading.value == c.context_relevance_floor
        assert reading.value > c.context_elbow_floor_min

    def test_one_dominant_sentence_cannot_collapse_the_context_to_itself(self):
        """The steepest fall here is at rank 1. Cutting there would send a
        single sentence and call it a reading."""
        c = full_stack().compression
        reading = self._read([1.0, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
        kept = sum(s >= reading.value for s in [1.0, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
        assert reading.rank in (-1, c.context_elbow_min_keep - 1)
        assert kept >= 1

    def test_the_reading_is_clamped_to_the_range_the_constant_was_measured_over(self):
        c = full_stack().compression
        reading = self._read([1.0, 0.99, 0.98, 0.97, 0.10, 0.09, 0.08, 0.07])
        assert c.context_elbow_floor_min <= reading.value <= c.context_elbow_floor_max

    def test_too_few_candidates_to_read_a_shape_from_uses_the_constant(self):
        c = full_stack().compression
        reading = self._read([1.0, 0.2, 0.02])
        assert reading.value == c.context_relevance_floor
        assert "too few" in reading.why

    def test_the_floor_sits_in_the_middle_of_the_cliff(self):
        """No score lies strictly between two adjacent ranks, so the midpoint
        keeps exactly what the lower lip would and sits furthest from either.
        A floor AT the lower score would keep the sentence it means to refuse."""
        scores = [1.0, 0.8, 0.6, 0.5, 0.1, 0.09, 0.08, 0.07]
        reading = self._read(scores)
        assert reading.rank == 3
        assert 0.1 < reading.value < 0.5
        assert sum(s >= reading.value for s in scores) == 4

    def test_it_is_off_by_default_so_the_shipped_numbers_still_describe_it(self):
        assert full_stack().compression.context_adaptive_floor is False

    def test_the_selection_carries_the_reading_that_produced_it(self, tok):
        """Nothing outside select() can recompute an adaptive floor, so a
        surface that draws the line has to be given it."""
        from parsimony.core.types import split_into_documents
        from parsimony.modules.m1_context import audit

        text = (Path(__file__).resolve().parents[2]
                / "examples/staff-handbook.md").read_text(encoding="utf-8")
        docs = split_into_documents(text, "staff-handbook.md")
        cfg = full_stack()
        cfg = replace(cfg, compression=replace(cfg.compression, context_adaptive_floor=True))
        pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
        report = audit(pipe.build_context("When does the office close for Christmas?",
                                          documents=docs), cfg)
        assert report.floor.why and report.floor.value > 0
        if report.floor.rank >= 0:
            assert report.floor.fall >= cfg.compression.context_elbow_min_fall


class TestAQuestionSharingNothingWithTheContext:
    """ADR-052. The topical check is an AND: coverage below its floor AND best
    cosine below its own. A neural encoder scores any two pieces of workplace
    prose alike, so on a document in the question's own domain its arm almost
    never fires -- and the AND then discards the coverage signal exactly where
    coverage is right.

    Measured on the shipped handbook, which covers three offices, travel,
    equipment loans and onboarding: "What is the notice period?" shares not one
    content word with it, and 280 tokens were sent anyway. The protection got
    weaker when the encoder got better.
    """

    HANDBOOK = "examples/staff-handbook.md"

    def _audit(self, question, cfg=None, tok=None):
        from pathlib import Path

        from parsimony.core.types import split_into_documents
        from parsimony.modules.m1_context import audit

        text = (Path(__file__).resolve().parents[2] / self.HANDBOOK).read_text(encoding="utf-8")
        docs = split_into_documents(text, "staff-handbook.md")
        cfg = cfg or full_stack()
        pipe = Pipeline(cfg, provider=MockProvider(), tokenizer=tok)
        return audit(pipe.build_context(question, documents=docs), cfg)

    #: In the handbook's own domain, and absent from it. The class the frozen
    #: corpus does not contain -- its off-topic questions are all FAR off.
    ABSENT = "What is the notice period?"

    def test_nothing_in_common_is_refused_however_similar_the_encoder_finds_it(self, tok):
        report = self._audit(self.ABSENT, tok=tok)
        assert report.coverage == 0.0, (
            "this question must share no content word with the handbook, or it "
            "is not testing what it claims to")
        assert report.off_topic
        assert report.tokens_after < 40, (
            f"an unanswerable question sent {report.tokens_after} tokens")

    def test_the_rule_can_be_switched_off_and_the_old_behaviour_returns(self, tok):
        """Which is what makes it measurable rather than asserted."""
        from dataclasses import replace

        cfg = full_stack()
        off = replace(cfg, compression=replace(cfg.compression,
                                               context_topic_zero_coverage=False))
        assert self._audit(self.ABSENT, off, tok).coverage == 0.0

    def test_it_does_not_touch_a_question_the_handbook_answers(self, tok):
        for question in ("What is the travel budget for Porto?",
                         "Who manages the Porto office?"):
            report = self._audit(question, tok=tok)
            assert not report.off_topic, question
            assert report.tokens_after > 40, question

    def test_a_question_with_no_content_words_is_not_called_off_topic(self, tok):
        """`coverage` is 1.0 by definition when there is nothing to cover, and
        the rule must not read that as zero."""
        report = self._audit("What is it?", tok=tok)
        assert report.coverage == 1.0 or not report.off_topic

    def test_partial_overlap_is_still_the_encoder_and_the_floor_to_decide(self, tok):
        """The rule claims only the zero case. "When does the office close for
        Christmas?" shares "office" and is not covered by it -- saying so is
        the difference between a rule and a hope."""
        report = self._audit("When does the office close for Christmas?", tok=tok)
        assert 0.0 < report.coverage < 0.5
