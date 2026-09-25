"""The CLI surface, checked against what the documents claim it is.

Two failures motivated this file.

The architecture figure -- which is embedded in the project report, the slide
deck and the paper -- listed the command line as "run, sweep, compare, ask,
tour, ...". Neither `run` nor `sweep` has ever existed; the commands are `chat`
and `bench`. The figure was drawn from memory, three documents inherited it,
and nothing in the build could notice, because a diagram is a PNG and a PNG
does not import anything.

And `parsimony chat ""` returned a confident paragraph. Nothing on the way in
treated an empty payload as special -- the fidelity gate guards against a
module *producing* one, which is the other direction -- so the pipeline
compressed nothing, retrieved nothing, and the provider answered the empty
string.

Both are cheap to assert and expensive to find by eye.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from parsimony.surfaces.cli.main import app, require_query

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


def cli_commands() -> set[str]:
    """Every command name the app actually exposes."""
    return {
        (c.name or c.callback.__name__).replace("_", "-")
        for c in app.registered_commands
    }


class TestTheCommandsTheDocumentsPromise:
    """Any command name a document prints must exist."""

    # Where a document lists the command line, and the pattern that finds the
    # list. Kept explicit rather than scanning all prose: "run" and "chat" are
    # ordinary English words, and a greedy scan would drown in false positives.
    SOURCES = (
        (ROOT / "tools" / "make_diagrams.py",
         r"Command line interface</text>\s*<text[^>]*>([^<]+)</text>"),
        (ROOT / "docs" / "14-project-report.md",
         r"The command-line interface: ([^.|]+)"),
        # Stops at the LaTeX row terminator, not at the first backslash --
        # every name in the list is wrapped in \texttt{}.
        (ROOT / "latex" / "report.tex",
         r"The command line: (.+?)\.?\s*\\\\"),
    )

    @pytest.mark.parametrize("path,pattern", SOURCES,
                             ids=[p.name for p, _ in SOURCES])
    def test_every_command_named_in_the_document_exists(self, path, pattern):
        if not path.exists():                      # optional document
            pytest.skip(f"{path.name} not present")
        text = path.read_text(encoding="utf-8")
        match = re.search(pattern, text)
        assert match, f"no command list found in {path.name}"

        # Strip markup the diagrams and LaTeX use around the names.
        raw = match.group(1)
        raw = raw.replace("&#183;", ",").replace("·", ",")
        raw = re.sub(r"\\texttt\{([^}]*)\}", r"\1", raw)
        raw = raw.replace("`", "").replace("$", "")
        named = {n.strip() for n in re.split(r"[,;]", raw) if n.strip()}
        assert named, f"parsed an empty command list from {path.name}"

        missing = named - cli_commands()
        assert not missing, (
            f"{path.name} names {sorted(missing)}, which the CLI does not "
            f"provide. Available: {sorted(cli_commands())}"
        )


class TestAnEmptyQuestionIsRefused:
    """A question with no content is a user error, not a prompt."""

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  \n"])
    def test_require_query_rejects_blank_input(self, blank):
        with pytest.raises(Exception) as exc:      # typer.BadParameter
            require_query(blank)
        assert "empty" in str(exc.value).lower()

    def test_require_query_returns_stripped_text(self):
        assert require_query("  What is recursion?  ") == "What is recursion?"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_chat_exits_nonzero_on_a_blank_question(self, blank):
        result = runner.invoke(app, ["chat", blank])
        assert result.exit_code != 0, (
            "a blank question produced a successful run; the pipeline will "
            "answer the empty string if nothing stops it"
        )

    def test_chat_still_accepts_a_real_question(self):
        result = runner.invoke(app, ["chat", "What is 2 + 2?", "--no-trace"])
        assert result.exit_code == 0, result.output


class TestUnknownProviderIsReportedNotRaised:
    def test_a_misspelled_provider_does_not_reach_the_user_as_a_traceback(self):
        result = runner.invoke(app, ["chat", "hello", "--provider", "bogus"])
        assert result.exit_code != 0
        # The message must name the thing that was wrong and the valid values.
        blob = result.output + str(result.exception or "")
        assert "bogus" in blob
        assert "mock" in blob and "ollama" in blob


class TestTheLauncherAndTheCliAgree:
    """`demo.ps1` is what runs in the room, and it reaches the CLI by name.

    Nothing in the build could notice a launcher act invoking a command that
    does not exist, or a help line advertising an act the switch does not
    handle -- a PowerShell switch falls through to its default and a typo
    becomes a stack trace in front of the guide.
    """

    LAUNCHER = ROOT / "demo.ps1"

    def _text(self):
        if not self.LAUNCHER.exists():
            pytest.skip("demo.ps1 not present")
        return self.LAUNCHER.read_text(encoding="utf-8")

    def test_every_command_the_launcher_invokes_exists(self):
        text = self._text()
        invoked = {m.group(1) for m in re.finditer(r"Run-Cli ([a-z][\w-]*)", text)}
        invoked -= {"@args"}
        missing = invoked - cli_commands()
        assert not missing, (
            f"demo.ps1 runs {sorted(missing)}, which the CLI does not provide. "
            f"Available: {sorted(cli_commands())}")

    def test_every_act_the_help_advertises_is_handled(self):
        text = self._text()
        advertised = {m.group(1) for m in re.finditer(r"demo\.ps1 ([a-z0-9]+)", text)}
        assert advertised, "no help lines found; this test would assert nothing"
        handled = {m.group(1) for m in re.finditer(r'^\s{4}"([a-z0-9]*)" \{', text, re.M)}
        # A name the switch does not handle is fine IF the CLI provides it:
        # `default` forwards anything unrecognised straight through. What must
        # not happen is a name that is neither, which reaches the CLI as an
        # unknown command in front of the room.
        missing = advertised - handled - cli_commands()
        assert not missing, (
            f"the launcher mentions {sorted(missing)}, which is neither an act it "
            f"handles nor a command the CLI provides, so `default` would forward it "
            f"and it would fail as an unknown command")


class TestOneFloorExperimentNotTwo:
    """`parsimony floor` and the reproduction run both report ADR-051's
    comparison. They each computed it themselves at first, and had already
    drifted -- different column names, and both writing
    `figures/adaptive_floor.csv`, so whichever ran last decided what the
    committed table meant.
    """

    def test_both_entry_points_call_the_shared_measurement(self):
        from pathlib import Path

        import parsimony.surfaces.cli.main as cli

        root = Path(cli.__file__).resolve().parents[4]
        for path in (Path(cli.__file__), root / "reproduce.py"):
            if not path.exists():
                pytest.skip(f"{path.name} not present")
            source = path.read_text(encoding="utf-8")
            assert "floor_rows" in source, (
                f"{path.name} must go through longctx.floor_rows, not measure it again")

    def test_the_splits_are_named_once(self):
        from parsimony.eval.longctx import FLOOR_SPLITS

        assert set(FLOOR_SPLITS) == {"development", "held out", "off topic"}
        # Every split the corpus has is accounted for, so no items are quietly
        # left out of the comparison.
        from parsimony.eval.longctx import load_longctx

        named = {s for names in FLOOR_SPLITS.values() for s in names}
        present = {i.split for i in load_longctx()}
        assert not present - named - {"offtopic_dev"}, (
            f"these splits are in the corpus and in no floor group: "
            f"{sorted(present - named - {'offtopic_dev'})}")

    def test_the_shipped_arm_is_labelled_from_the_configuration(self):
        """A literal "0.15 (shipped)" would keep that name after the default
        moved, which is the one thing a results table must never do."""
        from dataclasses import replace

        from parsimony.core.config import full_stack
        from parsimony.eval.longctx import floor_arms

        cfg = full_stack()
        moved = replace(cfg, compression=replace(cfg.compression,
                                                 context_relevance_floor=0.22))
        assert any("0.22" in label for label in floor_arms(moved))
        assert not any("0.15" in label for label in floor_arms(moved))

    def test_the_two_callers_cannot_write_the_same_file(self):
        """The committed table carries both encoders; the command runs one. A
        single path for both is how the committed one lost half its rows."""
        from pathlib import Path

        import parsimony.surfaces.cli.main as cli

        source = Path(cli.__file__).read_text(encoding="utf-8")
        assert 'Path("figures/adaptive_floor.csv")' not in source, (
            "the command must not default to the path reproduce.py writes")
        assert "adaptive_floor_{base.embedder_id" in source
