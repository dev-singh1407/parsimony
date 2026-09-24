"""The sentence splitter: where a sentence ends, and where it only looks like it."""

from __future__ import annotations

import pytest

from parsimony.infra.nlp import split_sentences


class TestTheSplitterKnowsWhereASentenceContinues:
    """Found by reading a compressed prompt, not by a failing test.

    A LongBench passage came back as "1312 - 16 September 1360) was an English
    nobleman", a sentence with no subject and an unmatched bracket. The
    abbreviation merge was testing the WRONG fragment: it asked whether the
    fragment being appended ended in an abbreviation, when the question is
    whether the one before it did. So "Contact Dr. Raman" split in two, and two
    genuinely separate sentences would merge whenever the second happened to
    end in "etc." Neither behaviour had a test.
    """

    @pytest.mark.parametrize("text", [
        "Contact Dr. Raman about the lab.",
        "The report (see Fig. 4) explains the method.",
        "Bring a laptop, e.g. a ThinkPad, to the session.",
        "Humphrey de Bohun (c. 1312 - 16 September 1360) was an English nobleman.",
        "Approx. 40 people work there.",
        "The team (led by Prof. Iyer) published in 2024.",
    ])
    def test_a_sentence_containing_an_abbreviation_stays_whole(self, text):
        assert split_sentences(text) == (text,)

    @pytest.mark.parametrize("text, count", [
        ("Leeds has a lab. Porto does not.", 2),
        ("The lab is in Leeds. Contact Dr. Raman.", 2),
        ("One. Two. Three.", 3),
    ])
    def test_real_boundaries_are_still_boundaries(self, text, count):
        """The fix must not turn a paragraph into one sentence."""
        assert len(split_sentences(text)) == count

    def test_a_full_stop_inside_a_parenthesis_is_not_a_boundary(self):
        """This is what catches the abbreviations nobody listed -- a date range,
        an initial, a citation."""
        text = "The site (opened 2014. rebuilt 2019) holds the calibration lab."
        assert split_sentences(text) == (text,)

    def test_nesting_is_counted_not_searched(self):
        text = "See the note (in the annex [section 2]. it is short) for detail."
        assert split_sentences(text) == (text,)

    def test_a_stray_closing_bracket_does_not_swallow_the_paragraph(self):
        """Prose contains unmatched brackets. Treating one as an open span
        would merge everything after it into a single sentence."""
        got = split_sentences("The budget is 84,000). The rest follows.")
        assert len(got) == 2

    def test_the_previous_fragment_decides_not_the_next(self):
        """The exact inversion that was wrong: a second sentence ending in an
        abbreviation must not glue itself to the first."""
        got = split_sentences("The lab is in Leeds. It runs daily, weekly, etc.")
        assert len(got) == 2
        assert got[0] == "The lab is in Leeds."

    def test_a_decimal_is_not_a_boundary_and_never_was(self):
        assert split_sentences("It cost 4.5 million in total.") == ("It cost 4.5 million in total.",)

    def test_code_fences_are_still_left_alone(self):
        text = "Run this:\n```\nx = 1. y = 2.\n```\nThen continue."
        got = split_sentences(text)
        assert any(g.startswith("```") and "x = 1. y = 2." in g for g in got)


class TestTheTokenizerLoadsWithoutTheNetwork:
    """The project's claim is that it runs on a laptop with the network off.

    Tokenizer.from_pretrained contacts the hub on every construction, even
    with the vocabulary already on disk, and announces it:

        Warning: You are sending unauthenticated requests to the HF Hub.

    Printing that at the start of a demo says the opposite of the thing being
    demonstrated. The cache is tried first now, and the hub is a fallback.
    """

    def test_it_reads_the_cache_rather_than_the_hub(self):
        import os

        from parsimony.infra.tokenization import HFTokenizer

        # If this needed the network it would fail here rather than warn.
        os.environ['HF_HUB_OFFLINE'] = '1'
        try:
            tok = HFTokenizer('Qwen/Qwen2.5-1.5B-Instruct')
            assert tok.count('The Tallinn office employs 58 people.') > 0
        finally:
            os.environ.pop('HF_HUB_OFFLINE', None)

    def test_it_leaves_the_environment_as_it_found_it(self):
        """Setting a process-global for the duration of a load is acceptable;
        leaving it set is not, since it would silently break anything that
        legitimately needs the hub later."""
        import os

        from parsimony.infra.tokenization import HFTokenizer

        before = os.environ.get('HF_HUB_OFFLINE')
        HFTokenizer('Qwen/Qwen2.5-1.5B-Instruct')
        assert os.environ.get('HF_HUB_OFFLINE') == before

    def test_a_preexisting_setting_is_preserved_not_clobbered(self):
        import os

        from parsimony.infra.tokenization import HFTokenizer

        os.environ['HF_HUB_OFFLINE'] = '0'
        try:
            HFTokenizer('Qwen/Qwen2.5-1.5B-Instruct')
            assert os.environ['HF_HUB_OFFLINE'] == '0'
        finally:
            os.environ.pop('HF_HUB_OFFLINE', None)
