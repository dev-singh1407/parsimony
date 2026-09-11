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
