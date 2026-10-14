"""Whether the cache clubs a question with the same question asked differently.

It did not, and the reason was not the encoder. Rewordings cleared the cosine
floor, reached the verifier, and were refused there by a single check: content
word Jaccard below 0.55. "manages" and "manager", "budget" and "allowance",
"borrow" and "loaning" are different strings, so a reworded question scored
0.20 to 0.40 and a floor of 0.55 refused it for being reworded.

Two suites here. The first pins the mechanism and runs anywhere: soft overlap
is Jaccard with a different notion of equal, and it must stay a Jaccard --
bounded, symmetric, and unable to let one word pay for three. The second is the
behaviour that motivated it, and it needs a real encoder, so it skips when one
is not reachable rather than asserting something the lexical encoder cannot
show (ADR-041, ADR-046).

The adversarial pairs are here too, and they matter more than the gains. Four
of the verifier's five checks are about meaning and none of them changed; this
suite exists to prove that widening what counts as the same word did not widen
what counts as the same question.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from parsimony.core.config import full_stack
from parsimony.infra.embedding import best_config, get_embedder
from parsimony.modules.m2_cache import cache_key_text, embed_text, soft_overlap


class _Stub:
    """An encoder with an opinion we control, so the measure can be pinned.

    Words in the same group are the same direction; everything else is
    orthogonal to everything. No real encoder is this tidy, which is the point:
    the properties asserted below must hold for the measure itself, not for one
    encoder's geometry.
    """

    GROUPS = (("manage", "manager", "runs"), ("budget", "allowance", "spend"))

    def __call__(self, words):
        out = []
        for w in words:
            v = np.zeros(8, dtype=np.float32)
            for i, group in enumerate(self.GROUPS):
                if w in group:
                    v[i] = 1.0
                    break
            else:
                v[2 + (hash(w) % 6)] = 1.0
            out.append(v)
        return np.vstack(out)


def fs(*words):
    return frozenset(words)


class TestSoftOverlapIsStillAJaccard:
    def test_identical_sets_are_one(self):
        assert soft_overlap(fs("a", "b"), fs("a", "b"), _Stub(), 0.6) == 1.0

    def test_disjoint_unrelated_sets_are_zero(self):
        assert soft_overlap(fs("alpha"), fs("omega"), _Stub(), 0.6) == 0.0

    def test_it_is_symmetric(self):
        a, b = fs("manage", "porto"), fs("manager", "porto")
        s = _Stub()
        assert soft_overlap(a, b, s, 0.6) == soft_overlap(b, a, s, 0.6)

    def test_a_synonym_counts_as_the_word_it_stands_for(self):
        """{manage, porto} against {manager, porto} is 1.0, not 0.33."""
        assert soft_overlap(fs("manage", "porto"), fs("manager", "porto"),
                            _Stub(), 0.6) == 1.0

    def test_one_word_cannot_pay_for_several(self):
        """Three synonyms on one side, one on the other: one match, not three.

        Without the greedy alignment spending each word once, a set containing
        a word and two of its synonyms would match everything and the floor
        would stop meaning anything.
        """
        got = soft_overlap(fs("manage"), fs("manager", "runs"), _Stub(), 0.6)
        assert got == pytest.approx(1 / 2)          # 1 matched, union 2

    def test_a_high_word_threshold_falls_back_to_exact_counting(self):
        exact = soft_overlap(fs("manage", "porto"), fs("manager", "porto"),
                             _Stub(), 1.01)
        assert exact == pytest.approx(1 / 3)

    def test_no_encoder_means_exact_counting_not_everything(self):
        """The safe direction when the encoder is gone is a missed hit."""
        assert soft_overlap(fs("manage", "porto"), fs("manager", "porto"),
                            None, 0.6) == pytest.approx(1 / 3)

    def test_an_encoder_that_raises_does_not_take_the_request_with_it(self):
        def angry(_words):
            raise RuntimeError("encoder gone")

        assert soft_overlap(fs("manage"), fs("manager"), angry, 0.6) == 0.0


class TestWhatTheEncoderIsGiven:
    """`embed_text` is the text the encoder sees; `cache_key_text` is not."""

    def test_a_contraction_keeps_its_verb(self):
        """It used to shed "what" and keep a bare "s" at the head."""
        assert embed_text("What's the travel budget?").startswith("what is")
        assert cache_key_text("What's the travel budget?").startswith("s ")

    def test_the_question_word_survives_for_the_encoder(self):
        assert embed_text("Who manages Porto?").startswith("who")

    def test_the_question_word_is_still_stripped_for_the_lexical_compare(self):
        assert not cache_key_text("Who manages Porto?").startswith("who")

    def test_courtesy_comes_out_of_both(self):
        assert "please" not in embed_text("Please tell me the budget, thanks!")

    def test_multiplication_signs_read_as_words(self):
        assert "time" in embed_text("What is 17 x 4?")

    def test_a_query_of_pure_courtesy_still_embeds_something(self):
        assert embed_text("please?").strip()


def _neural():
    cfg = best_config(full_stack())
    try:
        emb = get_embedder(cfg.embedder_id)
        emb.embed(["probe"])
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"neural encoder unreachable ({exc}); claim is encoder-specific")
    return cfg, emb


#: The same question asked differently. Each pair agrees on question type, so
#: each one reaches the overlap floor and nothing else stands in its way.
REWORDINGS = [
    ("What is the annual travel budget for the Tallinn office?",
     "What's the yearly travel allowance in Tallinn?"),
    ("Who manages the Porto office?", "Who is the site manager in Porto?"),
    ("Explain what a hash table is.", "What is a hash table?"),
]

#: The subset the old measure actually refused. The hash-table pair is not in
#: it: "Explain what a hash table is" and "What is a hash table" share every
#: content word once the framing is stripped, so strict Jaccard is already 1.0
#: and it matched before any of this. Asserting otherwise would be claiming
#: credit for a pair that was never broken.
NEEDED_SOFT_MATCHING = REWORDINGS[:2]

#: Different questions that share most of their words. Each is refused by a
#: check that is about meaning, and none of those checks changed.
NOT_THE_SAME = [
    ("Is the calibration lab in Leeds?", "Is the calibration lab not in Leeds?"),
    ("What is 17 times 4?", "What is 17 times 5?"),
    ("What is the Leeds travel budget?", "What is the Porto travel budget?"),
    ("What is the maximum hotel rate?", "What is the minimum hotel rate?"),
    ("Who manages Porto?", "Who manages Leeds?"),
]


class TestAgainstTheRealEncoder:
    @pytest.mark.parametrize("qa,qb", REWORDINGS)
    def test_a_rewording_clears_the_floor(self, qa, qb):
        cfg, emb = _neural()
        ca, cb = (frozenset(cache_key_text(q).split()) for q in (qa, qb))
        soft = soft_overlap(ca, cb, lambda ws: emb.embed(list(ws)),
                            cfg.cache.word_tau)
        assert soft >= cfg.cache.soft_jaccard_min, (
            f"{soft:.2f} < {cfg.cache.soft_jaccard_min}: a reworded question "
            "is being refused for being reworded")

    @pytest.mark.parametrize("qa,qb", NEEDED_SOFT_MATCHING)
    def test_and_the_old_measure_refused_it(self, qa, qb):
        """The half of the finding that says the floor was the binding one."""
        cfg, _emb = _neural()
        ca, cb = (frozenset(cache_key_text(q).split()) for q in (qa, qb))
        strict = len(ca & cb) / len(ca | cb) if (ca or cb) else 0.0
        assert strict < cfg.cache.jaccard_min

    @pytest.mark.parametrize("qa,qb", NOT_THE_SAME)
    def test_a_different_question_is_still_refused(self, qa, qb):
        """End to end, through the pipeline, with the real encoder."""
        from parsimony.infra.providers import MockProvider
        from parsimony.infra.tokenization import HeuristicTokenizer
        from parsimony.modules.m2_cache import SemanticCache
        from parsimony.pipeline.orchestrator import Pipeline
        from parsimony.pipeline.registry import default_registry

        cfg, _emb = _neural()
        cache = SemanticCache(cfg.cache.ttl_seconds)
        pipe = Pipeline(cfg, provider=MockProvider(),
                        tokenizer=HeuristicTokenizer("test"), cache=cache,
                        registry=default_registry(cache))
        pipe.run(qa, (), conversation_id="c", turn_index=0)
        out = pipe.run(qb, (), conversation_id="c", turn_index=1)
        assert not out.row.cache_hit, (
            f"served the answer to {qa!r} for {qb!r}")


class TestWhatKindOfAnswerIsWanted:
    """ADR-054. "What" is the generic interrogative and states no kind of its
    own; its subject does. Comparing resolved kinds instead of surface openers
    is what lets "what is the annual travel budget" match "how much can they
    spend" without letting "what is the deadline" match "when is the deadline".

    The controls matter more than the gains and are checked first.
    """

    #: Same subject, same words, a different opener. The opener IS the question
    #: here, so none of these may ever be served.
    OPENER_IS_THE_QUESTION = [
        ("What is the office budget?", "Where is the office budget?"),
        ("What is the loan period?", "Who sets the loan period?"),
        ("What is the travel policy?", "Why is the travel policy?"),
        ("When does the office close?", "Why does the office close?"),
        ("How many people work in Porto?", "Where do people work in Porto?"),
    ]

    #: Different questions about one subject, differently worded. The case the
    #: resolver could plausibly let through.
    SAME_SUBJECT_DIFFERENT_QUESTION = [
        ("What is the travel budget for Leeds?", "How do I claim back travel expenses?"),
        ("How long is a standard equipment loan?", "Who has to approve an equipment loan?"),
        ("Who manages the Porto office?", "How many people does the Porto office employ?"),
        ("How much is the hotel allowance?", "When must expense receipts be submitted?"),
    ]

    #: Reworded, and differing in opener. These are the gain.
    SAME_QUESTION_DIFFERENT_OPENER = [
        ("What is the annual travel budget for the Tallinn office?",
         "How much can Tallinn staff spend on travel in a year?"),
        ("How do I borrow test equipment?",
         "What is the process for loaning equipment?"),
        ("Who manages the Porto office?",
         "What is the name of the Porto site manager?"),
    ]

    @pytest.mark.parametrize("qa,qb", OPENER_IS_THE_QUESTION)
    def test_a_changed_opener_on_the_same_words_still_disagrees(self, qa, qb):
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import question_types_agree
        assert not question_types_agree(qa, qb, emb.embed)

    @pytest.mark.parametrize("qa,qb", SAME_SUBJECT_DIFFERENT_QUESTION)
    def test_a_different_question_about_one_subject_still_disagrees(self, qa, qb):
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import question_types_agree
        assert not question_types_agree(qa, qb, emb.embed)

    @pytest.mark.parametrize("qa,qb", SAME_QUESTION_DIFFERENT_OPENER)
    def test_the_same_question_agrees_despite_a_different_opener(self, qa, qb):
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import question_types_agree
        assert question_types_agree(qa, qb, emb.embed)
        assert not question_types_agree(qa, qb), (
            "this pair is only interesting because the surface openers differ")

    def test_a_specific_opener_is_never_overruled_by_its_subject(self):
        """Only "what" is resolved. "How much is the deadline" is nonsense, and
        the resolver must not invent a kind for a question that stated one."""
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import resolved_question_kind
        for q, kind in [("When is the deadline?", "when"),
                        ("Who manages Porto?", "who"),
                        ("Where is the lab?", "where")]:
            assert resolved_question_kind(q, emb.embed) == kind

    def test_without_an_encoder_it_falls_back_to_the_surface_opener(self):
        from parsimony.modules.m2_cache import question_type, resolved_question_kind
        q = "What is the annual travel budget?"
        assert resolved_question_kind(q, None) == question_type(q)

    #: Pairs whose SURFACE openers already agree. The resolver must not be
    #: consulted for these at all. Shipped without that guard, it was, and its
    #: own noise invented disagreements: "What is machine learning?" and
    #: "Define machine learning" resolved to "why" and "who", which cost three
    #: pairs of corpus/interactive_pairs.jsonl.
    ALREADY_AGREE = [
        ("What is machine learning?", "Define machine learning"),
        ("What is photosynthesis?", "Describe photosynthesis"),
        ("Which city is the capital of Japan?", "What is the capital of Japan?"),
        ("What is the travel budget?", "What is the travel budget for the team?"),
    ]

    @pytest.mark.parametrize("qa,qb", ALREADY_AGREE)
    def test_the_resolver_cannot_overturn_an_agreement(self, qa, qb):
        """It may only ever RESCUE a pair the openers refuse.

        That makes the whole mechanism monotone: turning it on can add matches
        and can never remove one. Any other shape makes it a second, noisier
        opinion about pairs that were already settled.
        """
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import question_types_agree
        assert question_types_agree(qa, qb), "fixture is wrong: these must agree on the surface"
        assert question_types_agree(qa, qb, emb.embed), (
            "the resolver overturned an agreement the openers had already reached")


class TestAModifierHasToOppose:
    """A word one question uses and the other does not is wording, not an
    inversion. Set equality treated it as one, and refused seven rewordings in
    forty for words like "long", "close" and "allowed" (ADR-055)."""

    def test_the_same_opposition_both_ways_round_is_caught(self):
        from parsimony.infra.nlp import modifiers_oppose
        assert modifiers_oppose("What is the maximum rate?", "What is the minimum rate?")
        assert modifiers_oppose("flights under six hours", "flights over six hours")
        assert modifiers_oppose("Is encryption required?", "Is encryption optional?")

    def test_a_word_only_one_side_uses_is_not_an_opposition(self):
        from parsimony.infra.nlp import modifiers_oppose
        assert not modifiers_oppose("How long is probation?",
                                    "What is the probation period?")
        assert not modifiers_oppose("When does the office close?",
                                    "What are the closing dates?")
        assert not modifiers_oppose("Can I take a laptop abroad?",
                                    "Am I allowed to take a laptop abroad?")

    def test_a_text_using_both_sides_still_differs_from_one_using_neither(self):
        """"between the minimum and the maximum" holds both sides of a group."""
        from parsimony.infra.nlp import modifiers_oppose
        assert modifiers_oppose("What is between the minimum and maximum rate?",
                                "What is the maximum rate?")


class TestNumbersWrittenAsWords:
    """"the first week" and "the second week" differ by a number and nothing
    else. With digits only, the extractor saw neither, so they agreed on
    numbers, agreed on everything else, and one was served for the other."""

    def test_ordinals_are_numbers(self):
        from parsimony.infra.nlp import RegexInvariantExtractor
        ex = RegexInvariantExtractor()
        assert ex.extract("What does the first week cover?").numbers != \
            ex.extract("What does the second week cover?").numbers

    def test_an_ordinal_folds_onto_its_cardinal(self):
        """"the second week" and "week two" are the same week."""
        from parsimony.infra.nlp import RegexInvariantExtractor
        ex = RegexInvariantExtractor()
        assert ex.extract("the second week").numbers == ex.extract("week two").numbers

    def test_a_number_word_inside_another_word_is_not_a_number(self):
        from parsimony.infra.nlp import RegexInvariantExtractor
        ex = RegexInvariantExtractor()
        assert not ex.extract("money for someone, nonetheless").numbers

    def test_digits_still_work(self):
        from parsimony.infra.nlp import RegexInvariantExtractor
        ex = RegexInvariantExtractor()
        assert ex.extract("17 times 4").numbers != ex.extract("17 times 5").numbers


class TestTheResolverStaysOutOfOpenerOnlyPairs:
    """"What is the deadline?" and "When is the deadline?" have one content word
    between them and it is the same word, so the opener carries the whole
    question. Without this guard the resolver reads "deadline" as a
    when-question and serves one for the other."""

    PAIRS = [
        ("What is the deadline?", "When is the deadline?"),
        ("What is the office budget?", "Where is the office budget?"),
        ("What is the travel policy?", "Why is the travel policy?"),
    ]

    @pytest.mark.parametrize("qa,qb", PAIRS)
    def test_identical_content_words_mean_the_opener_decides(self, qa, qb):
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import (
            cache_key_text, content_words, question_types_agree)
        assert content_words(cache_key_text(qa)) == content_words(cache_key_text(qb)), \
            "fixture is wrong: these must share their content words"
        assert not question_types_agree(qa, qb, emb.embed)


class TestTheHeadStatesTheKind:
    """ADR-057. "the headcount of the Porto office" is a headcount; "the booking
    process for the roof terrace" is a process. Resolving on all of it let the
    office and the terrace outvote the word carrying the kind, and both came
    back `where`."""

    def test_the_head_is_what_comes_before_what_it_is_about(self):
        from parsimony.modules.m2_cache import cache_key_text, complement_head
        for q, want in [
            ("What is the headcount of the Porto office?", "headcount"),
            ("What is the booking process for the roof terrace?", "booking process"),
            ("What is the method for allocating parking?", "method"),
            ("What is the deadline?", "deadline"),
        ]:
            assert complement_head(cache_key_text(q)) == want

    def test_a_tokenised_hyphen_does_not_truncate_the_head(self):
        """"sign-in process" becomes "sign in process"; cutting at "in" leaves
        "sign", which resolves to `why`. Only of/for/from/about end a head."""
        from parsimony.modules.m2_cache import cache_key_text, complement_head
        assert complement_head(cache_key_text("What is the sign-in process for visitors?")) \
            == "sign in process"

    def test_a_head_is_never_empty(self):
        from parsimony.modules.m2_cache import complement_head
        for subject in ("of", "the", "", "   ", "for"):
            assert complement_head(subject) == subject or complement_head(subject)

    def test_the_kinds_it_now_gets_right(self):
        _cfg, emb = _neural()
        from parsimony.modules.m2_cache import resolved_question_kind
        for q, kind in [
            ("What is the headcount of the Porto office?", "quantity"),
            ("What is the method for allocating parking?", "how"),
            ("What is the booking process for the roof terrace?", "how"),
            ("What is the name of the Porto site manager?", "who"),
            ("What is the address of the Leeds office?", "where"),
        ]:
            assert resolved_question_kind(q, emb.embed) == kind, q
