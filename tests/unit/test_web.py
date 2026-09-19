"""The local visualiser: its endpoints, and the honesty of its counters.

A page that shows a saving is a page that can overstate one. The counter tests
here are the important ones: seconds may only be called measured when this
machine timed them, and a mock model or a cache hit must flip that label rather
than quietly inflating the total. The rest pins that the page serves from the
same pipeline the terminal uses, and that a pasted document cannot inject script
into it.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from parsimony.surfaces import web

HANDBOOK = Path(__file__).resolve().parents[2] / "examples/staff-handbook.md"
QUESTION = "What is the annual travel budget for the Tallinn office?"


@pytest.fixture(scope="module")
def handbook_text():
    return HANDBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def vis():
    return web.Visualiser(provider_name="mock")


@pytest.fixture(scope="module")
def server(vis):
    """A real socket: the handler's job is HTTP, so test it over HTTP."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(vis))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.read()


def post_json(base: str, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestTheCounters:
    def test_pruned_tokens_are_exactly_written_minus_sent(self):
        totals = web.SessionTotals()
        totals.record(written=900, sent=300, prefill_s=1.0, rate_ms=10.0, timed_here=True)
        totals.record(written=100, sent=100, prefill_s=0.5, rate_ms=10.0, timed_here=True)
        snap = totals.snapshot()
        assert snap["tokens_pruned"] == 600
        assert snap["tokens_written"] == 1000 and snap["tokens_sent"] == 400

    def test_seconds_are_the_gap_between_what_was_read_and_what_was_written(self):
        totals = web.SessionTotals()
        # 1000 tokens at 10 ms each would be 10 s; 3 s were actually spent.
        totals.record(written=1000, sent=300, prefill_s=3.0, rate_ms=10.0, timed_here=True)
        assert totals.snapshot()["seconds_saved"] == pytest.approx(7.0)

    def test_a_saving_is_never_reported_as_negative(self):
        totals = web.SessionTotals()
        totals.record(written=10, sent=10, prefill_s=9.0, rate_ms=1.0, timed_here=True)
        assert totals.snapshot()["seconds_saved"] == 0.0

    def test_seconds_are_flagged_estimated_unless_this_machine_timed_them(self):
        totals = web.SessionTotals()
        totals.record(written=900, sent=300, prefill_s=1.0, rate_ms=10.0, timed_here=True)
        assert totals.snapshot()["timed_here"] is True
        totals.record(written=900, sent=300, prefill_s=1.0, rate_ms=10.0, timed_here=False)
        assert totals.snapshot()["timed_here"] is False, (
            "one untimed request makes the total an estimate, and the page must say so")

    def test_nothing_run_is_not_a_measurement(self):
        assert web.SessionTotals().snapshot()["timed_here"] is False

    def test_a_simulated_model_never_claims_measured_seconds(self, vis, handbook_text):
        """The mock's rate is the project's recorded constant, not this laptop's."""
        vis.totals = web.SessionTotals()
        for label, cfg, docs in vis.arms(QUESTION, handbook_text):
            vis.run_arm(label, cfg, docs, QUESTION, lambda *a: None)
        snap = vis.totals.snapshot()
        assert snap["requests"] == 2
        assert snap["tokens_pruned"] > 0
        assert snap["timed_here"] is False


class TestTheHeatmap:
    def test_it_reports_the_same_decisions_the_audit_made(self, vis, handbook_text):
        from parsimony.modules.m1_context import audit
        from parsimony.pipeline.orchestrator import Pipeline
        from parsimony.core.types import split_into_documents

        data = vis.compress(QUESTION, handbook_text)
        pipe = Pipeline(vis.cfg, provider=vis.provider)
        ctx = pipe.build_context(QUESTION, documents=split_into_documents(handbook_text, "x"))
        report = audit(ctx, vis.cfg)
        assert data["tokens_before"] == report.tokens_before
        assert data["tokens_after"] == report.tokens_after
        assert [u["tag"] for u in data["units"]] == [u.tag for u in report.units]

    def test_every_sentence_carries_a_score_a_tag_and_a_reason(self, vis, handbook_text):
        data = vis.compress(QUESTION, handbook_text)
        assert data["units"]
        for unit in data["units"]:
            assert 0.0 <= unit["score"] <= 1.0
            assert unit["tag"] and unit["detail"]
            assert isinstance(unit["kept"], bool)

    def test_the_encoder_cost_is_reported_beside_what_it_bought(self, vis, handbook_text):
        data = vis.compress(QUESTION, handbook_text)
        assert data["encoder_ms"] >= 0 and data["decide_ms"] >= 0
        assert data["removed_pct"] > 0

    def test_an_off_topic_question_is_labelled_rather_than_answered_from_nothing(
            self, vis, handbook_text):
        data = vis.compress("What is the capital of Peru?", handbook_text)
        assert data["off_topic"] is True
        assert sum(u["kept"] for u in data["units"]) >= 1, (
            "an empty context reads as an instruction with a missing attachment")


class TestTheEndpoints:
    def test_the_page_is_served_and_names_itself(self, server):
        status, body = get(server, "/")
        assert status == 200
        assert b"<title>Parsimony" in body

    def test_the_page_loads_no_third_party_code(self, server):
        _, body = get(server, "/")
        text = body.decode("utf-8")
        for scheme in ("https://", "http://cdn", "//cdn", "//unpkg"):
            assert scheme not in text, (
                "the visualiser must work with the network off, like the middleware")

    def test_state_says_whether_the_model_is_real(self, server):
        _, body = get(server, "/api/state")
        state = json.loads(body)
        assert state["simulated"] is True
        assert state["encoder"]

    def test_compress_requires_both_a_question_and_a_document(self, server):
        status, payload = post_json(server, "/api/compress", {"question": "", "text": ""})
        assert status == 400 and "error" in payload

    def test_compress_returns_the_decisions(self, server, handbook_text):
        status, payload = post_json(server, "/api/compress",
                                    {"question": QUESTION, "text": handbook_text})
        assert status == 200
        assert payload["tokens_after"] < payload["tokens_before"]
        assert any(u["tag"] == "ANCHOR" for u in payload["units"])

    def test_a_broken_request_answers_rather_than_hanging(self, server):
        status, payload = post_json(server, "/api/compress", {"question": QUESTION})
        assert status == 400 and "error" in payload

    def test_an_unknown_path_is_a_clean_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/whatever")
        assert exc.value.code == 404


class TestTheRace:
    def test_it_streams_both_arms_and_ends_with_a_summary(self, server, handbook_text):
        query = urllib.parse.urlencode({"question": QUESTION, "text": handbook_text})
        events, payload = [], None
        with urllib.request.urlopen(f"{server}/api/race?{query}", timeout=300) as r:
            name = None
            for raw in r:
                line = raw.decode("utf-8").strip()
                if line.startswith("event:"):
                    name = line.split(": ", 1)[1]
                    events.append(name)
                elif line.startswith("data:") and name in ("done", "failed"):
                    payload = json.loads(line.split(": ", 1)[1])
                    break
        assert "failed" not in events, payload
        assert events.count("arm_done") == 2
        assert "token" in events, "a race with no streamed tokens shows nothing happening"
        arms = {a["arm"]: a for a in payload["arms"]}
        assert set(arms) == {"parsimony", "baseline"}
        assert arms["parsimony"]["prompt_tokens"] < arms["baseline"]["prompt_tokens"]

    def test_the_compressed_arm_runs_first_so_it_pays_any_cold_start(self, vis,
                                                                    handbook_text):
        assert [a[0] for a in vis.arms(QUESTION, handbook_text)] == ["parsimony", "baseline"]

    def test_a_simulated_model_is_not_warmed_up(self, vis):
        assert vis.warm_up() is False


class TestThePage:
    def test_it_escapes_document_text_before_drawing_it(self):
        text = (web.PAGE).read_text(encoding="utf-8")
        assert "const esc = (s) =>" in text
        assert "esc(u.text)" in text, "sentence text goes through the escaper, not innerHTML raw"

    def test_it_ships_beside_the_module_that_serves_it(self):
        assert web.PAGE.exists() and web.PAGE.suffix == ".html"

    def test_the_sample_document_is_found_from_a_checkout(self):
        assert web.sample_document() is not None


class TestThePipelineView:
    """The stage-by-stage stream behind the flow graph.

    The graph's claim is that it draws what the orchestrator did, so the tests
    that matter are the ones tying each drawn thing to a real trace: the nodes
    to the registry's own order, the numbers to the ledger row, and the tier
    ladder to stages that actually exist.
    """

    def test_the_plan_is_the_registry_not_a_hand_written_list(self, vis):
        from parsimony.pipeline.orchestrator import Pipeline

        pipe = Pipeline(vis.cfg, provider=vis.provider)
        expected = [getattr(getattr(p, "stage", p), "name", "")
                    for p in pipe.registry.ordered(vis.cfg)]
        assert [s["name"] for s in vis.plan(vis.cfg)] == expected

    def test_every_stage_is_labelled_for_a_reader(self, vis):
        for stage in vis.plan(vis.cfg):
            assert stage["label"] and not stage["label"].startswith("m")
            assert stage["module"].startswith("M")
            assert stage["does"], f"{stage['name']} has no plain-language description"

    def test_the_compression_tiers_are_the_ladder(self, vis):
        tiers = [s["name"] for s in vis.plan(vis.cfg) if s["role"] == "compress"]
        assert tiers == ["m1_context", "m1_tier1", "m1_tier2", "m1_tier3"]

    def test_it_streams_a_stage_for_every_stage_that_ran(self, vis, handbook_text):
        events = []
        summary = vis.run_pipeline(QUESTION, handbook_text,
                                   lambda name, data: events.append((name, data)))
        names = [n for n, _ in events]
        assert names[0] == "plan" and names[1] == "begin" and names[-1] == "done"
        streamed = [d["name"] for n, d in events if n == "stage"]
        assert streamed == [s["name"] for s in vis.plan(vis.cfg)]
        assert len(summary["stages"]) == len(streamed)

    def test_the_numbers_on_screen_come_from_the_ledger(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        t = summary["tokens"]
        assert t["removed"] == t["original"] - t["final"]
        assert 0 < t["final"] < t["original"]
        assert t["ratio"] == pytest.approx(t["final"] / t["original"])

    def test_each_stage_carries_the_reason_it_gives_in_the_ledger(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        for stage in summary["stages"]:
            assert stage["outcome"] in {"applied", "noop", "skipped", "reverted",
                                        "short_circuit", "error", "not_implemented"}
            assert stage["rationale"], f"{stage['name']} reported no reason"
            assert stage["ms"] >= 0

    def test_the_compressor_is_the_stage_that_removes_the_tokens(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        by = {s["name"]: s for s in summary["stages"]}
        assert by["m1_context"]["after"] < by["m1_context"]["before"]

    def test_a_question_with_no_documents_still_runs(self, vis):
        summary = vis.run_pipeline("What is 17 times 4?", "", lambda *a: None)
        assert summary["tokens"]["final"] > 0
        assert summary["route"]["tier"]

    def test_evidence_is_trimmed_to_something_a_page_can_render(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        for stage in summary["stages"]:
            assert len(json.dumps(stage["evidence"])) < 4000

    def test_the_decisions_panel_has_something_to_show(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        assert "consulted" in summary["cache"] and "hit" in summary["cache"]
        assert "fired" in summary["gate"]
        assert summary["route"]["tier"]
        assert summary["timing"]["middleware_ms"] >= 0

    def test_the_endpoint_streams_over_http(self, server, handbook_text):
        query = urllib.parse.urlencode({"question": QUESTION, "text": handbook_text})
        seen, payload = [], None
        with urllib.request.urlopen(f"{server}/api/pipeline?{query}", timeout=300) as r:
            name = None
            for raw in r:
                line = raw.decode("utf-8").strip()
                if line.startswith("event:"):
                    name = line.split(": ", 1)[1]
                    seen.append(name)
                elif line.startswith("data:") and name in ("done", "failed"):
                    payload = json.loads(line.split(": ", 1)[1])
                    break
        assert "failed" not in seen, payload
        assert seen.count("stage") >= 8
        assert payload["tokens"]["removed"] > 0

    def test_a_request_with_no_question_is_refused(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server, "/api/pipeline?question=&text=x")
        assert exc.value.code == 400


class TestTheFlowDrawing:
    """Two failures the page hit in testing, pinned so they cannot return."""

    def test_the_fit_refuses_a_zero_width_container(self):
        page = web.PAGE.read_text(encoding="utf-8")
        assert "if (avail < 80) return;" in page, (
            "a hidden tab reports clientWidth 0, which produced a negative scale "
            "and flipped the diagram inside out")

    def test_counters_do_not_depend_on_animation_frames_running(self):
        page = web.PAGE.read_text(encoding="utf-8")
        assert "if (document.hidden) { show(target); return; }" in page
        assert "_safety" in page, (
            "requestAnimationFrame does not run in a background tab; the number "
            "must arrive even when the animation does not")
