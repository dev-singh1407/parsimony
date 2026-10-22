"""What the fidelity gate lets through.

M8 is the claim the architecture rests on: modules only propose, and the gate
refuses anything that changes what the request means. It was covered by about
eight hand-written examples, all passing, which is how M2 came to report 4 of 5
when the real number was 45%.

`_check_rewrite` asked one question -- was any extractable value LOST -- and a
value that was never there cannot be missing. So the gate was blind to anything
ADDED, and to every class the extractor does not model. Measured on forty
meaning-changing rewrites, twenty-two passed.

Each class below is one of those twenty-two. The second suite matters at least
as much: a gate that refuses everything scores perfectly on the first and makes
the compressor useless, and politeness removal is M1 tier 1's commonest edit.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import full_stack
from parsimony.core.proposals import TransformKind
from parsimony.infra.providers import MockProvider
from parsimony.infra.tokenization import HeuristicTokenizer
from parsimony.modules.m2_cache import SemanticCache
from parsimony.pipeline.orchestrator import Pipeline
from parsimony.pipeline.registry import default_registry


@pytest.fixture(scope="module")
def gate_allows():
    cfg = full_stack()
    cache = SemanticCache(cfg.cache.ttl_seconds)
    pipe = Pipeline(cfg, provider=MockProvider(),
                    tokenizer=HeuristicTokenizer("test"), cache=cache,
                    registry=default_registry(cache))

    def _allows(before: str, after: str) -> bool:
        ctx = pipe.build_context(before)
        return pipe.gate.check(ctx, replace(ctx, query=after),
                               TransformKind.REWRITE, "M1").passed

    return _allows


class TestTheGateRefusesAChangeOfMeaning:
    CHANGES = [
        ("negation added", "Visitors must sign in at reception.",
         "Visitors must not sign in at reception."),
        ("morphological negation", "The change is possible.",
         "The change is impossible."),
        ("morphological negation", "The fee is refundable.",
         "The fee is non-refundable."),
        ("modal weakened", "Laptops must be encrypted before connecting.",
         "Laptops may be encrypted before connecting."),
        ("obligation dropped", "Encryption is required.", "Encryption is optional."),
        ("opposition reversed", "The maximum nightly rate is 140 pounds.",
         "The minimum nightly rate is 140 pounds."),
        ("opposition reversed", "Flights under six hours are economy.",
         "Flights over six hours are economy."),
        ("opposition reversed", "Loans longer than 7 days need approval.",
         "Loans shorter than 7 days need approval."),
        ("temporal reversed", "Report damage before 24 hours have passed.",
         "Report damage after 24 hours have passed."),
        ("scope dropped", "Only firmware engineers take the secure coding course.",
         "Firmware engineers take the secure coding course."),
        ("quantifier weakened", "All employees must book through Voyant.",
         "Some employees must book through Voyant."),
        ("condition dropped", "If the ticket is escalated, Tallinn engineers join.",
         "Tallinn engineers join."),
        ("hedge dropped", "Late returns may be flagged to the site manager.",
         "Late returns are flagged to the site manager."),
        ("currency swapped", "Hotel costs are reimbursed up to 140 pounds.",
         "Hotel costs are reimbursed up to 140 euros."),
        ("currency swapped", "The daily allowance is 45 pounds.",
         "The daily allowance is 45 dollars."),
        ("number changed", "The annual travel budget is 84000 pounds.",
         "The annual travel budget is 48000 pounds."),
        ("entity changed", "The Leeds site manager is Priya Raman.",
         "The Porto site manager is Priya Raman."),
    ]

    @pytest.mark.parametrize("why,before,after", CHANGES,
                             ids=[f"{w}:{b[:24]}" for w, b, _ in CHANGES])
    def test_it_is_refused(self, gate_allows, why, before, after):
        assert not gate_allows(before, after), why


class TestTheGateStillAllowsHonestTightening:
    """A gate that refuses everything is not a safe gate, it is a broken
    compressor. Politeness removal is M1 tier 1's commonest edit and every one
    of these must survive."""

    PRESERVES = [
        ("courtesy", "Please could you tell me the travel budget? Thanks!",
         "What is the travel budget?"),
        ("courtesy", "Hello, I was wondering if you could please explain the policy.",
         "Explain the policy."),
        ("filler", "I was just wondering what the meal allowance is.",
         "What is the meal allowance?"),
        ("impersonal opener", "It should be noted that parking is limited to 20 spaces.",
         "Parking is limited to 20 spaces."),
        ("whitespace", "The   budget   is  84000   pounds.",
         "The budget is 84000 pounds."),
        ("contraction expanded", "The lab isn't available on Fridays.",
         "The lab is not available on Fridays."),
        ("punctuation", "The budget is 84000 pounds!!!", "The budget is 84000 pounds."),
        ("repeat removed", "The deadline is 15 March. The deadline is 15 March.",
         "The deadline is 15 March."),
    ]

    @pytest.mark.parametrize("why,before,after", PRESERVES,
                             ids=[f"{w}:{b[:24]}" for w, b, _ in PRESERVES])
    def test_it_is_allowed(self, gate_allows, why, before, after):
        assert gate_allows(before, after), why


class TestWhatTheGateStillCannotSee:
    """Recorded rather than hidden. Each needs structure the gate does not
    model, and a test asserting the current behaviour is how a later change
    gets noticed (ADR-060)."""

    @pytest.mark.parametrize("before,after", [
        ("The handbook states that parking is allocated by lottery.",
         "Parking is allocated by lottery."),
        ("Prototypes are designed in Tallinn and built in Leeds.",
         "Prototypes are designed in Leeds and built in Tallinn."),
    ], ids=["attribution-dropped", "order-swapped"])
    def test_these_still_pass_and_should_not(self, gate_allows, before, after):
        assert gate_allows(before, after), (
            "this now fails, which is an improvement -- move it into the "
            "refusal suite and update ADR-060")


class TestTheOtherTransformKindsAreUntouched:
    """SELECT exists to drop whole turns and EXTRACT to drop whole sentences.
    Neither is a change of meaning, and a uniform gate would veto M3 and M1
    permanently (ADR-003)."""

    def test_select_may_still_drop_a_turn_carrying_a_qualifier(self):
        from parsimony.core.types import Turn
        cfg = full_stack()
        cache = SemanticCache(cfg.cache.ttl_seconds)
        pipe = Pipeline(cfg, provider=MockProvider(),
                        tokenizer=HeuristicTokenizer("test"), cache=cache,
                        registry=default_registry(cache))
        history = (Turn("a", "user", "Only managers may approve this."),
                   Turn("b", "user", "Something else entirely."))
        ctx = pipe.build_context("carry on", history)
        after = replace(ctx, history=(history[1],))
        assert pipe.gate.check(ctx, after, TransformKind.SELECT, "M3").passed
