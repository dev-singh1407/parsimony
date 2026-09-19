"""The encoder, the thresholds calibrated for it, and the host it talks to.

ADR-041 turned on three things that are easy to get wrong again:

  * verification runs on every candidate, so the cache's safety no longer
    depends on the encoder being too weak to score a lookalike pair highly;
  * a threshold belongs to an encoder, so swapping one without the other is a
    silent behaviour change;
  * `localhost` costs two seconds per call on this machine, and nothing in the
    runtime's own counters can see it.

The tests that need a model server skip without one, so the suite still runs on
a machine that has never installed Ollama.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from parsimony.core.config import NEURAL_EMBEDDER, full_stack, neural
from parsimony.core.types import Turn
from parsimony.eval.calibration import _lookup_hits, evaluate_point
from parsimony.eval.corpus import load_adversarial
from parsimony.infra.embedding import OllamaEmbedder, get_embedder
from parsimony.infra.providers import DEFAULT_OLLAMA_HOST, fast_host

BLEACH = ("Is it safe to mix bleach and vinegar?", "Is it not safe to mix bleach and vinegar?")
needs_embedder = pytest.mark.skipif(not OllamaEmbedder.available(),
                                    reason="no embedding model reachable")


class TestTheHost:
    def test_the_default_is_the_ip_not_the_name(self):
        """localhost resolves to ::1 first and costs ~2 s per call here."""
        assert "127.0.0.1" in DEFAULT_OLLAMA_HOST
        assert "localhost" not in DEFAULT_OLLAMA_HOST

    def test_a_localhost_url_a_caller_passes_is_rewritten(self):
        assert fast_host("http://localhost:11434") == "http://127.0.0.1:11434"
        assert fast_host("http://example.test:1234") == "http://example.test:1234"

    def test_providers_and_embedders_both_use_it(self):
        from parsimony.infra.providers import OllamaProvider

        assert OllamaProvider(host="http://localhost:11434").host == "http://127.0.0.1:11434"
        assert OllamaEmbedder(host="http://localhost:11434").host == "http://127.0.0.1:11434"


class TestVerificationIsUnconditional:
    @staticmethod
    @pytest.fixture(scope="class")
    def pairs():
        return load_adversarial()

    def test_it_is_on_by_default(self):
        assert full_stack().cache.verify_always is True

    def test_no_threshold_admits_a_false_answer(self, pairs):
        """With the accept zone gone, the threshold stops being a safety parameter."""
        embedder = get_embedder(full_stack().embedder_id)
        for tau_hi in (0.75, 0.85, 0.92, 0.97, 0.99):
            cfg = full_stack()
            cfg = replace(cfg, cache=replace(cfg.cache, tau_hi=tau_hi))
            assert evaluate_point(pairs, cfg, embedder, verifier_on=True).false_hits == 0

    def test_the_zone_it_replaces_is_still_reachable_for_comparison(self, pairs):
        """The three-zone design is what the literature uses; it stays measurable."""
        cfg = full_stack()
        zoned = replace(cfg, cache=replace(cfg.cache, verify_always=False, tau_hi=0.90))
        embedder = get_embedder(cfg.embedder_id)
        assert evaluate_point(pairs, zoned, embedder, verifier_on=True).false_hits > 0

    def test_a_high_scoring_negation_is_refused_rather_than_served(self, make_pipeline):
        pipe = make_pipeline(full_stack())
        pipe.run(BLEACH[0])
        second = pipe.run(BLEACH[1])
        assert second.generated, "the opposite question must not be served from the cache"


@needs_embedder
class TestTheNeuralEncoder:
    @staticmethod
    @pytest.fixture(scope="class")
    def embedder():
        return get_embedder(NEURAL_EMBEDDER)

    def test_it_scores_the_adversarial_pair_higher_than_the_lexical_one_does(self, embedder):
        """The reason the accept zone had to go: a better space makes a negation
        look MORE similar, not less."""
        neural_score = float(embedder.embed(list(BLEACH))[0] @ embedder.embed(list(BLEACH))[1])
        lexical = get_embedder("content-v1")
        lexical_score = float(lexical.embed(list(BLEACH))[0] @ lexical.embed(list(BLEACH))[1])
        assert neural_score > lexical_score > 0.85

    def test_it_connects_meanings_no_lexical_score_can(self, embedder):
        """'daily dose' against 'once a day' -- the miss that cost an answer in ADR-040."""
        question = "What daily dose of velastrin do adults take?"
        sentence = "Participants took either 25 mg of velastrin or a placebo once a day."
        a, b = embedder.embed([question, sentence])
        assert float(a @ b) > 0.6

    def test_the_preset_carries_its_own_calibration(self):
        cfg = neural()
        assert cfg.embedder_id == NEURAL_EMBEDDER
        assert cfg.cache.tau_lo == 0.70, "a threshold belongs to the encoder it was set for"

    def test_it_is_safe_and_reuses_more_at_its_own_calibration(self, embedder):
        pairs = load_adversarial()
        point = evaluate_point(pairs, neural(), embedder, verifier_on=True)
        lexical_point = evaluate_point(pairs, full_stack(),
                                       get_embedder("content-v1"), verifier_on=True)
        assert point.false_hits == 0
        assert point.true_hits > lexical_point.true_hits

    def test_every_free_typing_pair_is_decided_correctly(self, embedder):
        import json
        from pathlib import Path

        corpus = Path(__file__).resolve().parents[2] / "corpus" / "interactive_pairs.jsonl"
        pairs = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        cfg = neural()
        wrong = [p["pair_id"] for p in pairs
                 if _lookup_hits(p["b"], p["a"], embedder, cfg, True) != (p["expect"] == "hit")]
        assert not wrong, f"decided wrongly: {wrong}"

    def test_the_cli_picks_it_up_and_says_so(self, capsys):
        from parsimony.surfaces.cli.main import pick_encoder

        cfg = pick_encoder(full_stack())
        assert cfg.embedder_id == NEURAL_EMBEDDER

    def test_a_pipeline_using_it_still_refuses_the_opposite_question(self):
        from parsimony.infra.providers import MockProvider
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(neural(), provider=MockProvider())
        pipe.run(BLEACH[0])
        assert pipe.run(BLEACH[1]).generated

    def test_a_rephrasing_the_lexical_encoder_misses_is_reused(self):
        from parsimony.infra.providers import MockProvider
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(neural(), provider=MockProvider())
        pipe.run("What is the capital city of Australia?")
        second = pipe.run("Which city is Australia's capital?")
        assert not second.generated, "a genuine rephrasing should be served from memory"

    def test_it_is_fast_enough_for_one_question(self, embedder):
        """A third of the 120 ms overhead budget, not two seconds (the host bug)."""
        import time

        embedder.embed(["warm"])
        start = time.perf_counter()
        embedder.embed([f"a fresh question about budgets {time.time()}"])
        assert (time.perf_counter() - start) * 1000 < 500


class TestHistoryTurnsStillWork:
    def test_the_selector_runs_under_either_encoder(self, make_pipeline):
        history = tuple(Turn(f"t{i}", "user" if i % 2 == 0 else "assistant", text)
                        for i, text in enumerate(
                            ["My server runs Ubuntu 22.04 with 8 GB of RAM.", "Noted.",
                             "Name a cat.", "Whiskers.", "Recommend a film.", "Inception."]))
        out = make_pipeline(full_stack()).run("Which database suits my server?", history)
        assert out.row.tokens_in_final > 0


@needs_embedder
class TestTheBenchmarkHarnessHonoursTheEncoder:
    """A harness bug worth a permanent test.

    The first neural long-context run produced output identical to the lexical
    one, token for token: `Methods` passed the neural CONFIG to the stage while
    the derived cache the stage reads vectors from still belonged to a pipeline
    built with the lexical encoder. An arm that silently measures another arm
    is worse than a missing arm.
    """

    def test_the_two_encoders_select_different_sentences(self):
        """Asserted over the corpus, not one item.

        It was one item, until section anchors (ADR-044) made the two encoders
        agree on exactly that item. A regression test that can be satisfied by
        a single question is a test of that question.
        """
        from parsimony.eval.longctx import Methods, load_longctx

        methods = Methods()
        items = list(load_longctx())
        differ = sum(methods.parsimony(i, full_stack()).documents
                     != methods.parsimony(i, neural()).documents for i in items)
        assert differ > len(items) // 2, (
            f"only {differ}/{len(items)} items differ between encoders -- the neural arm "
            "is probably scoring with the lexical encoder's cached vectors again")

    def test_the_neural_arm_keeps_the_answer_sentence_the_lexical_one_loses(self):
        """'daily dose' against 'once a day' -- ADR-040's named failure.

        Section anchors now rescue this item for the lexical encoder too: the
        question names OB-114, and the sentence sits under the heading "Trial
        OB-114: velastrin", so it inherits the anchor it never spells. The
        encoder gap ADR-040 measured is therefore pinned in the configuration
        ADR-040 measured it in, and the rescue is pinned beside it -- both are
        facts, and dropping either would let a real regression through.
        """
        from dataclasses import replace

        from parsimony.eval.longctx import Methods, evidence_kept, load_longctx

        items = {i.item_id: i for i in load_longctx()}
        methods = Methods()
        item = items["orrin_q1"]

        def without_sections(cfg):
            return replace(cfg, compression=replace(cfg.compression,
                                                    context_section_anchors=False))

        assert evidence_kept(
            item, methods.parsimony(item, without_sections(full_stack())).documents)[0] == 0
        assert evidence_kept(
            item, methods.parsimony(item, without_sections(neural())).documents)[0] == 1
        # And with section anchors on, the lexical encoder no longer loses it.
        assert evidence_kept(item, methods.parsimony(item, full_stack()).documents)[0] == 1


class TestTheChoiceIsGlobal:
    """One resolution point, used by every surface and every study.

    The encoder decides cache hits, history selection and sentence scoring
    alike, so a benchmark run on one encoder while `ask` used the other would
    be reporting a different system from the one in use.
    """

    def test_an_already_neural_config_is_left_alone(self):
        from parsimony.infra.embedding import best_config

        cfg = neural()
        assert best_config(cfg) is cfg

    def test_it_resolves_to_a_concrete_encoder_never_to_auto(self):
        """`config_hash` is experiment identity: two runs on different encoders
        must not share one, so nothing may stay unresolved."""
        from parsimony.infra.embedding import best_config

        chosen = best_config(full_stack())
        assert chosen.embedder_id in {"content-v1", NEURAL_EMBEDDER}
        assert chosen.config_hash == best_config(chosen).config_hash

    @needs_embedder
    def test_it_upgrades_and_recalibrates_together(self):
        from parsimony.infra.embedding import best_config

        chosen = best_config(full_stack())
        assert chosen.embedder_id == NEURAL_EMBEDDER
        assert chosen.cache.tau_lo == neural().cache.tau_lo

    def test_commands_say_which_encoder_they_used(self):
        from typer.testing import CliRunner

        from parsimony.surfaces.cli.main import app

        result = CliRunner().invoke(app, ["calibrate"])
        assert result.exit_code == 0, result.output
        assert "Encoder:" in result.output
