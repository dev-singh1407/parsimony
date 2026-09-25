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


@pytest.fixture(autouse=True)
def _fresh_cache(vis):
    """A new cache per test.

    The visualiser shares one cache across a session, which is the point -- ask
    the same thing twice and the second is served without the model. That also
    makes any test asking the same question twice depend on its neighbours, so
    each starts from empty and the sharing gets its own test below.
    """
    from parsimony.modules.m2_cache import SemanticCache

    vis.cache = SemanticCache(vis.cfg.cache.ttl_seconds,
                              max_entries=vis.cfg.cache.max_entries)


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
        app = (web.PAGE.parent / "web_app.js").read_text(encoding="utf-8")
        assert "const esc = (s) =>" in app
        assert "esc(u.text)" in app, "sentence text goes through the escaper, not raw"

    def test_it_ships_beside_the_module_that_serves_it(self):
        assert web.PAGE.exists() and web.PAGE.suffix == ".html"
        for name, _kind in web.ASSETS.values():
            assert (web.PAGE.parent / name).exists(), name

    def test_the_sample_document_is_found_from_a_checkout(self):
        assert web.sample_document() is not None

    def test_the_assets_are_served_locally(self, server):
        for path, (_name, kind) in web.ASSETS.items():
            status, body = get(server, path)
            assert status == 200 and body, path
        # and nothing on the page reaches for a network
        _, page = get(server, "/")
        text = page.decode("utf-8")
        for scheme in ("https://", "http://cdn", "//cdn", "//unpkg"):
            assert scheme not in text

    def test_counters_do_not_depend_on_animation_frames_running(self):
        """rAF is suspended in a background tab; the number must still arrive."""
        app = (web.PAGE.parent / "web_app.js").read_text(encoding="utf-8")
        assert "if (document.hidden) { show2(target); return; }" in app
        assert "_safety" in app


class TestTheSentenceLife:
    """The document view's data: every sentence, and the stage that removed it."""

    def test_every_sentence_is_accounted_for(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        assert summary["units"]
        for unit in summary["units"]:
            assert unit["text"].strip()
            assert "removed_at" in unit

    def test_the_context_tier_is_what_removed_them(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        removed = [u for u in summary["units"] if u["removed_at"]]
        assert removed, "this question must drop sentences for the view to show anything"
        assert {u["removed_at"] for u in removed} <= {s["name"] for s in summary["stages"]}
        assert any(u["removed_at"] == "m1_context" for u in removed)

    def test_a_removed_sentence_carries_its_score_and_reason(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        scored = [u for u in summary["units"] if u.get("tag")]
        assert scored, "the compressor's own scores must be joined onto the sentences"
        for unit in scored:
            assert unit["detail"], unit["text"][:40]
            assert 0.0 <= unit["score"] <= 1.0

    def test_a_sentence_that_only_moved_is_not_reported_as_cut(self, vis, handbook_text):
        """M1 regroups documents and M4 moves blocks; neither is a removal."""
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        survivors = [u for u in summary["units"] if not u["removed_at"]]
        assert survivors, "everything was reported as cut, which cannot be right"
        sent = summary["prompt"] or ""
        if sent:
            for unit in survivors[:5]:
                assert " ".join(unit["text"].split())[:40] in " ".join(sent.split())

    def test_no_documents_means_no_sentences_rather_than_a_crash(self, vis):
        summary = vis.run_pipeline("What is 17 times 4?", "", lambda *a: None)
        assert summary["units"] == []


class TestTheTerminalParity:
    """The web table must be the terminal's table, not a second opinion."""

    def test_it_uses_the_same_labels_the_terminal_prints(self, vis, handbook_text):
        from parsimony.surfaces.cli.explain import _NAME

        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        assert summary["layers"]
        for layer in summary["layers"]:
            expected = _NAME.get(layer["name"], (layer["name"], ""))
            assert layer["label"] == expected[0]
            assert layer["job"] == expected[1]

    def test_every_layer_explains_itself_in_plain_english(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        for layer in summary["layers"]:
            assert layer["why"], f"{layer['name']} had no explanation"
            assert layer["name"] not in layer["why"], "that is the code name, not English"

    def test_the_table_covers_every_planned_stage(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        assert [l["name"] for l in summary["layers"]] == [s["name"]
                                                          for s in vis.plan(vis.cfg)]

    def test_the_sentence_count_agrees_with_the_token_change(self, vis, handbook_text):
        summary = vis.run_pipeline(QUESTION, handbook_text, lambda *a: None)
        for layer in summary["layers"]:
            if layer.get("removed"):
                assert layer["delta"] < 0, (
                    f"{layer['name']} removed sentences but reported no token drop")


class TestTheCacheIsSharedAcrossRequests:
    """One cache for the session, not one per request.

    Every request built its own Pipeline, and a Pipeline builds its own cache
    when it is not handed one -- so the Memory layer was given an empty cache
    every time and reported the miss honestly, however often you asked. The
    layer could never demonstrate itself in the page it exists to be shown in.
    """

    def test_asking_twice_serves_the_second_without_the_model(self, vis):
        question = "What is the travel budget for Porto?"
        first = vis.run_pipeline(question, "", lambda *a: None)
        assert first["cache"]["hit"] is False
        second = vis.run_pipeline(question, "", lambda *a: None)
        assert second["cache"]["hit"] is True
        assert second["route"]["tier"].startswith("CACHE")
        assert second["tokens"]["final"] == 0, "a cache hit must not send a prompt"

    def test_a_rewording_is_verified_rather_than_assumed(self, vis):
        vis.run_pipeline("What is the travel budget for Porto?", "", lambda *a: None)
        near = vis.run_pipeline("What is the Porto travel budget?", "", lambda *a: None)
        if near["cache"]["hit"]:
            assert near["cache"]["zone"] in ("verify", "accept")

    def test_a_different_question_still_misses(self, vis):
        vis.run_pipeline("What is the travel budget for Porto?", "", lambda *a: None)
        other = vis.run_pipeline("How many people work in Leeds?", "", lambda *a: None)
        assert other["cache"]["hit"] is False


class TestThePageAndItsStylesheetAgree:
    """Markup, script and stylesheet must describe the same page.

    Written because they had drifted: `error`, `not_implemented` and
    `not_reached` were states the script could put on a node, with no rule
    anywhere to style them -- a stage that failed would have rendered
    indistinguishable from one that succeeded, which is the single worst thing
    this page could get wrong.
    """

    @staticmethod
    def _sources():
        d = web.PAGE.parent
        return (web.PAGE.read_text(encoding="utf-8"),
                (d / "web_app.js").read_text(encoding="utf-8"),
                (d / "web_app.css").read_text(encoding="utf-8"))

    #: Names the script builds by interpolation, which no regex over the source
    #: can see. Each must correspond to a value the server actually sends.
    BUILT = {
        "applied", "noop", "skipped", "reverted", "short_circuit", "error",
        "not_implemented", "not_reached",          # StageOutcome values
        "at", "past", "future", "sel",             # scrub position
        "kept", "cut", "alive", "flash",           # sentence states
        "ok", "no", "star", "stop",                # table marks
        "faded", "plain", "warn", "below", "dim", "keep", "drop", "gone",
    }

    def test_every_class_the_page_uses_has_a_rule(self):
        import re

        html, js, css = self._sources()
        used = set()
        for m in re.finditer(r'class="([^"{}$]+)"', html):
            used.update(m.group(1).split())
        for m in re.finditer(r'class="([^"$`{]*)', js):
            used.update(w for w in m.group(1).split() if w)
        defined = set(re.findall(r"\.([A-Za-z][\w-]*)", css))
        missing = sorted((used | self.BUILT) - defined)
        assert not missing, f"no style for: {', '.join(missing)}"

    def test_every_stage_outcome_can_be_told_apart(self):
        """A state the server can send must be visible as its own thing."""
        from parsimony.core.ledger import StageOutcome

        _html, _js, css = self._sources()
        for outcome in StageOutcome:
            assert f".{outcome.value}" in css, (
                f"{outcome.value} has no styling; it would look like a success")

    def test_every_id_the_script_reaches_for_exists(self):
        import re

        html, js, _css = self._sources()
        page_ids = set(re.findall(r'id="([^"$]+)"', html))
        built = {m + "-" for m in re.findall(r'id="([a-z-]+)-\$\{', js)}
        asked = set(re.findall(r'\$\("([^"$]+)"\)', js))
        unknown = sorted(i for i in asked - page_ids
                         if not any(i.startswith(p) for p in built))
        assert not unknown, f"the script reaches for ids the page lacks: {unknown}"

    def test_the_stylesheet_and_script_are_actually_served(self, server):
        for path in web.ASSETS:
            status, body = get(server, path)
            assert status == 200 and len(body) > 200, path


class TestARequestTheModelNeverSaw:
    """A cache hit and a calculator answer send no prompt at all.

    Both panels got this wrong: "What the AI actually received" listed the whole
    document as though it had been sent, and the timing line blamed a simulated
    model for the absent prefill. Overstating what was sent, in the two places a
    reader trusts most, is the worst available failure for this page.
    """

    def test_the_received_panel_says_nothing_was_sent(self, vis):
        summary = vis.run_pipeline("What is 17 * 4?", "", lambda *a: None)
        assert summary["served_without_model"] is True
        assert summary["tokens"]["final"] == 0
        kinds = {r["kind"] for r in summary["received"]}
        assert kinds == {"unsent"}, "no section may be listed as received"
        assert "no prompt" in summary["received"][0]["note"]

    def test_it_does_not_list_the_document_as_received(self, vis, handbook_text):
        """The cache answers the second ask; the document is not sent again."""
        question = "What is the annual travel budget for the Tallinn office?"
        vis.run_pipeline(question, handbook_text, lambda *a: None)
        again = vis.run_pipeline(question, handbook_text, lambda *a: None)
        if not again["served_without_model"]:
            pytest.skip("the cache did not serve this one")
        assert {r["kind"] for r in again["received"]} == {"unsent"}
        assert not any("Leeds" in (r["note"] or "") for r in again["received"])

    def test_the_measured_panel_says_it_answered_without_the_model(self, vis):
        summary = vis.run_pipeline("What is 17 * 4?", "", lambda *a: None)
        joined = " ".join(summary["measured"])
        assert "without the AI" in joined
        assert "removed" not in joined, "there was no prompt to remove anything from"

    def test_a_normal_request_still_lists_its_sections(self, vis, handbook_text):
        summary = vis.run_pipeline("Who runs the Porto office?", handbook_text,
                                   lambda *a: None)
        assert summary["served_without_model"] is False
        kinds = {r["kind"] for r in summary["received"]}
        assert "doc" in kinds or "doc-dropped" in kinds

    def test_the_page_has_a_branch_for_it(self):
        app = (web.PAGE.parent / "web_app.js").read_text(encoding="utf-8")
        assert "served_without_model" in app
        assert "the model was never called" in app

    def test_the_estimate_points_at_the_surface_you_are_on(self, vis, handbook_text):
        """The terminal says "type 'compare'"; a browser cannot."""
        summary = vis.run_pipeline("How many people work in Leeds?", handbook_text,
                                   lambda *a: None)
        joined = " ".join(summary["measured"])
        if "estimate" in joined:
            assert "type 'compare'" not in joined
            assert "A/B tab" in joined


class TestTheScriptActuallyParses:
    """A syntax error in the page script is invisible to every other test here.

    They read the file as text, so they pass while the page is dead: one bad
    token and nothing runs, the header sticks on "connecting...", and the only
    symptom is in a console nobody opened. This actually happened -- an escaped
    newline was written into the file as a real one, and 47 green tests said
    the page was fine.
    """

    def test_the_page_script_parses(self):
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            pytest.skip("no node available to parse the script")
        app = web.PAGE.parent / "web_app.js"
        result = subprocess.run([node, "--check", str(app)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr

    def test_the_stylesheet_has_balanced_braces(self):
        """A stylesheet cannot be parsed here, but an unclosed rule silently
        drops every rule after it, which is worth one cheap check."""
        css = (web.PAGE.parent / "web_app.css").read_text(encoding="utf-8")
        assert css.count("{") == css.count("}"), "unbalanced braces in the stylesheet"


class TestItWouldSurviveBeingInstalled:
    """Everything the server can serve has to ship with it.

    The page was one self-contained HTML file, so `package-data` declared
    `*.html` and that was enough. Splitting the stylesheet and script out --
    necessary once the page grew a scrubbable pipeline view -- silently broke
    every installed copy: `pip install .` shipped the markup alone, and the
    visualiser came up unstyled and inert. Nothing failed in the repo, where the
    files are simply present on disk.
    """

    @staticmethod
    def _declared_suffixes() -> set[str]:
        import re

        root = Path(__file__).resolve().parents[2]
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        block = re.search(r'"parsimony\.surfaces"\s*=\s*\[([^\]]*)\]', text)
        assert block, "parsimony.surfaces has no package-data entry at all"
        return {m.lower() for m in re.findall(r'"\*(\.[a-z0-9]+)"', block.group(1))}

    def test_every_servable_file_is_declared_as_package_data(self):
        needed = {web.PAGE.suffix.lower()}
        for name, _kind in web.ASSETS.values():
            needed.add(Path(name).suffix.lower())
        missing = needed - self._declared_suffixes()
        assert not missing, (
            f"pyproject.toml ships {sorted(self._declared_suffixes())} but the server also "
            f"serves {sorted(missing)}; an installed copy would 500 on those paths")

    def test_the_asset_table_and_the_files_on_disk_agree(self):
        """A file in the directory that nothing serves is dead weight; a served
        file that is not there is a crash."""
        surfaces = web.PAGE.parent
        on_disk = {p.name for p in surfaces.glob("web_*")}
        served = {web.PAGE.name} | {name for name, _ in web.ASSETS.values()}
        assert served <= on_disk, f"serves files that do not exist: {served - on_disk}"
        assert on_disk <= served, f"unserved files left behind: {on_disk - served}"


class TestTheDemoScreenCounts:
    """The demo screen is the one read from across a room, so every figure on
    it has to be one the session actually measured -- including the two that
    make the system look busy rather than clever."""

    def test_a_request_the_model_never_saw_is_counted_as_one(self, vis):
        vis.totals = web.SessionTotals()
        vis.run_pipeline("What is 17 * 4?", "", lambda *a: None)
        snap = vis.totals.snapshot()
        assert snap["requests"] == 1
        assert snap["without_model"] == 1

    def test_a_normal_request_is_not(self, vis, handbook_text):
        vis.totals = web.SessionTotals()
        vis.run_pipeline("Who runs the Porto office?", handbook_text, lambda *a: None)
        assert vis.totals.snapshot()["without_model"] == 0

    def test_the_gate_is_counted_when_it_refuses(self, vis):
        vis.totals = web.SessionTotals()
        vis.run_pipeline("Explain the deadline. The deadline is 15 March. "
                         "The deadline is 16 March.", "", lambda *a: None)
        snap = vis.totals.snapshot()
        assert snap["gate_refused"] >= 1
        assert snap["gate_checked"] >= snap["gate_refused"], (
            "a refusal is also a check; counting it only as a refusal loses the denominator")

    def test_a_refusal_is_never_reported_without_what_it_was_out_of(self, vis, handbook_text):
        vis.totals = web.SessionTotals()
        vis.run_pipeline("Who runs the Porto office?", handbook_text, lambda *a: None)
        snap = vis.totals.snapshot()
        assert snap["gate_checked"] >= 1
        assert snap["gate_refused"] == 0

    def test_kv_is_zero_rather_than_guessed_when_the_model_cannot_say(self, vis, handbook_text):
        """MockProvider reports no key-value geometry, so there is nothing to
        price the pruned tokens at. The screen shows a dash, not a number."""
        vis.totals = web.SessionTotals()
        vis.run_pipeline("Who runs the Porto office?", handbook_text, lambda *a: None)
        assert vis.totals.snapshot()["kv_mb_saved"] == 0.0

    def test_every_figure_the_screen_shows_is_in_the_payload(self, server):
        _status, body = get(server, "/api/session")
        snap = json.loads(body)
        for key in ("requests", "tokens_pruned", "seconds_saved", "timed_here",
                    "without_model", "gate_checked", "gate_refused", "kv_mb_saved"):
            assert key in snap, key


class TestTheFloorDial:
    """The page can sweep the relevance floor; it must not re-tune the system.

    ADR-050 found the floor is what decides how much survives, with a measured
    curve behind it. Making it draggable turns that table into something a
    reader can check -- but a control that quietly changed the shipped default
    would turn a demonstration into a configuration change nobody recorded.
    """

    def test_a_lower_floor_keeps_more(self, vis, handbook_text):
        high = vis.compress(QUESTION, handbook_text, 'h.md', 0.15)
        low = vis.compress(QUESTION, handbook_text, 'h.md', 0.05)
        assert sum(u['kept'] for u in low['units']) > sum(u['kept'] for u in high['units'])
        assert low['tokens_after'] > high['tokens_after']

    def test_the_shipped_default_is_never_mutated(self, vis, handbook_text):
        before = vis.cfg.compression.context_relevance_floor
        vis.compress(QUESTION, handbook_text, 'h.md', 0.0)
        assert vis.cfg.compression.context_relevance_floor == before

    def test_the_payload_reports_both_floors(self, vis, handbook_text):
        data = vis.compress(QUESTION, handbook_text, 'h.md', 0.05)
        assert data['floor'] == 0.05
        assert data['shipped_floor'] == vis.cfg.compression.context_relevance_floor
        assert data['floor'] != data['shipped_floor'], 'the page must be able to say so'

    def test_omitting_it_uses_the_shipped_value(self, vis, handbook_text):
        data = vis.compress(QUESTION, handbook_text, 'h.md')
        assert data['floor'] == vis.cfg.compression.context_relevance_floor

    def test_a_nonsense_floor_is_ignored_rather_than_crashing(self, server, handbook_text):
        status, payload = post_json(server, '/api/compress',
                                    {'question': QUESTION, 'text': handbook_text,
                                     'floor': 'banana'})
        assert status == 200
        assert payload['floor'] == payload['shipped_floor']

    def test_a_floor_out_of_range_is_clamped(self, server, handbook_text):
        for sent, expect in ((5.0, 1.0), (-2.0, 0.0)):
            status, payload = post_json(server, '/api/compress',
                                        {'question': QUESTION, 'text': handbook_text,
                                         'floor': sent})
            assert status == 200 and payload['floor'] == expect


class TestTheFloorModeOnThePage:
    """ADR-051 put a rule behind the floor, and a rule has to be legible or it
    is just a number the page cannot account for."""

    @staticmethod
    def _sources():
        from pathlib import Path

        here = Path(web.__file__).parent
        return (
            (here / "web_page.html").read_text(encoding="utf-8"),
            (here / "web_app.js").read_text(encoding="utf-8"),
            (here / "web_app.css").read_text(encoding="utf-8"),
        )

    def test_one_request_path_so_the_dials_cannot_be_left_out_of_one(self):
        """The Compress button used to issue its own fetch. It sent neither
        dial and never recorded the result, so the button ran at the shipped
        settings whatever the sliders said and switching the floor mode
        afterwards did nothing at all -- silently, because both paths worked."""
        _html, js, _css = self._sources()
        assert js.count('fetch("/api/compress"') == 1, (
            "two paths issuing the same request is two things to keep in step")

    def test_the_page_asks_for_the_mode_it_offers(self):
        html, js, _css = self._sources()
        assert 'id="mode-adaptive"' in html and 'id="mode-fixed"' in html
        assert "adaptive: adaptiveFloor" in js

    def test_the_computed_floor_is_reported_rather_than_drawn_from_the_slider(self):
        """In adaptive mode the slider is an output. A page that drew a line the
        run did not use would be worse than one that drew none."""
        _html, js, css = self._sources()
        assert "if (adaptiveFloor) floor = null;" in js
        assert ".dial.reading input[type=range]" in css

    def test_the_canvas_is_sized_from_its_box_not_from_fixed_attributes(self):
        """The backing store was 1200x500 against a stylesheet box of 100% by
        250: every label came out twice as wide as it was tall, the markers were
        ellipses, and the whole thing was blurred on any HiDPI screen. Nothing
        failed, so nothing noticed."""
        import re

        html, js, _css = self._sources()
        canvas = re.search(r"<canvas[^>]*>", html)
        assert canvas, "the A/B tab must still have a canvas"
        assert "width=" not in canvas.group(0), (
            "a fixed backing-store width is what the stylesheet then stretches")
        assert "canvasBox" in js and "devicePixelRatio" in js
        assert "setTransform(dpr, 0, 0, dpr, 0, 0)" in js, (
            "the context has to be scaled or every coordinate below is in the "
            "wrong units")

    def test_the_endpoint_reports_the_reading_it_used(self, vis, handbook_text):
        data = vis.compress("When does the office close for Christmas?", handbook_text,
                            "handbook", None, None, True)
        assert data["adaptive"] is True
        assert data["floor_why"]
        assert data["floor"] > 0
        # A rank the page can draw, or -1 meaning there was nothing to draw.
        assert data["cliff_rank"] >= -1

    def test_a_reading_and_a_constant_are_told_apart_in_what_is_sent(self, vis,
                                                                     handbook_text):
        q = "When does the office close for Christmas?"
        fixed = vis.compress(q, handbook_text, "handbook", None, None, False)
        read = vis.compress(q, handbook_text, "handbook", None, None, True)
        assert fixed["adaptive"] is False and read["adaptive"] is True
        assert fixed["floor_why"] != read["floor_why"]

    def test_asking_for_the_mode_does_not_change_what_the_system_ships(self, vis,
                                                                       handbook_text):
        """The same guarantee the floor slider carries: this page sweeps, it
        does not re-tune."""
        data = vis.compress("What is the travel budget for Porto?", handbook_text,
                            "handbook", None, None, True)
        assert data["shipped_adaptive"] is False
        assert vis.cfg.compression.context_adaptive_floor is False


class TestScoringWithEitherEncoder:
    """ADR-041, ADR-046 and ADR-051 all say the same thing from three
    directions: the encoder decides the scores, thresholds do not transfer
    between encoders, and a result that does not name its encoder says nothing.
    None of it was visible on the page it is a page about."""

    @staticmethod
    def _sources():
        from pathlib import Path

        here = Path(web.__file__).parent
        return ((here / "web_page.html").read_text(encoding="utf-8"),
                (here / "web_app.js").read_text(encoding="utf-8"),
                (here / "web_app.css").read_text(encoding="utf-8"))

    def test_the_switch_is_offered_only_when_there_is_something_to_switch_to(self, vis):
        """A build running the lexical encoder has no other one to offer, and a
        control that cannot work is worse than no control."""
        other = vis.other_encoder()
        assert other is None or other != vis.cfg.embedder_id
        if vis.cfg.embedder_id == vis.LEXICAL:
            assert other is None

    def test_the_page_hides_the_bar_until_the_server_names_a_second_encoder(self):
        html, js, _css = self._sources()
        assert 'id="encbar" hidden' in html
        assert "data.other_encoder" in js, (
            "the page must take the choice from the server, not assume one exists")

    def test_a_request_can_only_ask_for_an_encoder_the_page_offers(self, server,
                                                                   handbook_text):
        """An arbitrary id from a request is a way to ask the server to load
        anything it can find."""
        status, data = post_json(server, "/api/compress", {
            "question": "What is the travel budget for Porto?",
            "text": handbook_text, "encoder": "../../etc/passwd"})
        assert status == 200
        assert data["encoder"] == data["shipped_encoder"], (
            "an encoder the page does not offer must fall back, not be loaded")

    def test_the_answer_names_the_encoder_that_produced_it(self, vis, handbook_text):
        data = vis.compress("Who manages the Porto office?", handbook_text, "handbook")
        assert data["encoder"] and data["shipped_encoder"]

    def test_scoring_with_the_lexical_encoder_is_always_possible(self, vis, handbook_text):
        """It needs nothing running, which is what makes it the honest
        comparison arm rather than a second thing that can be unavailable."""
        data = vis.compress("Who manages the Porto office?", handbook_text, "handbook",
                            None, None, None, vis.LEXICAL)
        assert data["encoder"] == vis.LEXICAL
        assert data["units"], "the lexical encoder must still score every sentence"

    def test_asking_for_an_encoder_does_not_change_what_the_system_ships(self, vis,
                                                                         handbook_text):
        before = vis.cfg.embedder_id
        vis.compress("Who manages the Porto office?", handbook_text, "handbook",
                     None, None, None, vis.LEXICAL)
        assert vis.cfg.embedder_id == before


class TestWhatAReaderMeetsBeforeAndBetweenRuns:
    """Five things the page got wrong in the states it spends most of its time
    in: before anything has run, and between the stages of a finished run.
    None of them threw, so none of them was noticed."""

    @staticmethod
    def _sources():
        from pathlib import Path

        here = Path(web.__file__).parent
        return ((here / "web_page.html").read_text(encoding="utf-8"),
                (here / "web_app.js").read_text(encoding="utf-8"),
                (here / "web_app.css").read_text(encoding="utf-8"))

    def test_the_question_box_holds_a_question_rather_than_looking_like_it_does(self):
        """It shipped with a placeholder that reads as a filled field. The first
        thing anyone does is press Run and be told to ask something first."""
        import re

        html, _js, _css = self._sources()
        tag = re.search(r"<input[^>]*id=\"pq\"[^>]*>", html, re.S)
        assert tag, "the pipeline question input must still exist"
        assert 'value="' in tag.group(0), (
            "a placeholder that looks like a value is not a value")
        assert "data-default=" in tag.group(0), (
            "recall has to tell the shipped default from a question someone typed")

    def test_a_remembered_question_outranks_the_shipped_default(self):
        _html, js, _css = self._sources()
        assert "dataset.default" in js

    def test_nothing_run_yet_is_not_zero(self):
        """A zero is a measurement. The ring already said so with an em dash
        while three counters beside it claimed to have counted nothing."""
        import re

        html, _js, _css = self._sources()
        for metric in ("m-before", "m-after", "m-removed", "m-saved"):
            cell = re.search(rf'id="{metric}"[^>]*>([^<]*)<', html)
            assert cell, metric
            assert cell.group(1).strip() == "—", (
                f"{metric} starts at {cell.group(1)!r}, which claims a measurement")

    def test_a_word_sits_in_its_lane_without_the_animation(self):
        """`--x` reached the chip only through the keyframes, and the keyframes
        only run on a channel the request has reached -- so every channel below
        the scrub head, every channel before a run, and every channel for a
        reader who asked for reduced motion stacked all eleven chips on the
        centre line. That pile was the panel's normal appearance."""
        _html, _js, css = self._sources()
        base = css.split(".vword {")[1].split("}")[0]
        assert "translate(calc(-50% + var(--x" in base, (
            "the lane belongs to the element, not to the animation")

    def test_every_stage_of_the_transport_can_be_clicked(self):
        """The context selector is over 90% of middleware time, so at a 3px
        floor the other ten stages were three pixels wide -- on a control whose
        whole purpose is to be clicked through."""
        import re

        _html, _js, css = self._sources()
        rule = css.split(".seg { position: relative;")[1].split("}")[0]
        floor = re.search(r"min-width:\s*(\d+)px", rule)
        assert floor and int(floor.group(1)) >= 12, (
            "a segment has to be wide enough to hit")

    def test_the_proportions_are_still_proportions(self):
        """Equal-width segments would hide the finding, which is that one stage
        IS the time. Shrink lets the clamped ones take their floor out of the
        dominant segment instead."""
        _html, js, _css = self._sources()
        assert "flex:1 1 ${pct}%" in js

    def test_attaching_the_sample_cannot_be_raced_by_running(self):
        """Attach is a fetch. Pressing Run inside it read an empty question box,
        complained, and then the document landed -- leaving a complaint about an
        empty field under a filled one."""
        _html, js, _css = self._sources()
        assert "button.disabled = true" in js
        assert "clearPipeError" in js, "and the complaint has to go when it stops being true"
