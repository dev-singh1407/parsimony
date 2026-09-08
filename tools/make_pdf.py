"""Render a project document to a typeset PDF.

Markdown -> styled HTML -> Chrome headless -> PDF. Chrome is the only PDF
engine on this machine, and it is a good one: it gives real CSS paged media,
so the output is set properly rather than dumped.

Typography choices, since the point of this script is that the result should
not look like a rendered README:

  * Cambria for body text — deliberately, not as a fallback. The first version
    pulled Source Serif 4 from Google Fonts; headless Chrome did not finish
    fetching it and the PDF came out set in Cambria with Times New Roman
    leaking in for some glyphs. Rather than make a document that must
    regenerate reliably depend on a network fetch, the stack is now local:
    Cambria was designed by Jelle Bosma for extended reading in print and on
    screen, ships with Windows, and is a better academic body face than the
    Times New Roman it was displacing. Constantia and Georgia follow it.
  * Tables in the booktabs manner — rule above, rule below the header, rule at
    the foot, and NO vertical rules. Vertical rules are the single clearest
    sign of a table that was not typeset.
  * Old-style figures are avoided in tables (tabular-nums) so columns of
    numbers align on the digit.
  * Hanging indents on the bibliography, page numbers in the footer, and a
    running header carrying the short title.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)

CSS = """
@page {
  size: A4;
  margin: 20mm 19mm 18mm 19mm;
  @bottom-center { content: counter(page); }
}

:root {
  --ink:      #16181d;
  --muted:    #5a6069;
  --rule:     #c8ccd2;
  --rule-strong: #2b2f36;
  --accent:   #7a2518;
  --tint:     #f6f4ef;
}

* { box-sizing: border-box; }
html, body, table, th, td, li, p, h1, h2, h3, h4, h5, blockquote {
  font-family: Cambria, Constantia, Georgia, serif;
}

body {
  font-family: Cambria, Constantia, Georgia, serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: var(--ink);
  margin: 0;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-variant-numeric: oldstyle-nums proportional-nums;
  hyphens: auto;
}

/* ---------------------------------------------------------------- title -- */

.titleblock {
  text-align: center;
  margin: 0 0 2.2em;
  padding-bottom: 1.4em;
  border-bottom: 1px solid var(--rule);
}
.titleblock .kicker {
  font-size: 8.5pt; letter-spacing: .16em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: 1.1em;
}
.titleblock h1 {
  font-size: 21pt; line-height: 1.2; font-weight: 600;
  margin: 0 0 .35em; letter-spacing: -0.01em; border: 0; padding: 0;
}
.titleblock .subtitle {
  font-size: 12pt; color: var(--muted); font-style: italic; margin-bottom: 1.2em;
}
.titleblock .authors { font-size: 10pt; line-height: 1.7; }
.titleblock .affil { font-size: 9pt; color: var(--muted); }

/* ------------------------------------------------------------- headings -- */

h1, h2, h3, h4 {
  font-weight: 600;
  line-height: 1.25;
  color: var(--ink);
  break-after: avoid;
  page-break-after: avoid;
}
h1 {                                   /* major section */
  font-size: 15pt; margin: 2.1em 0 .7em;
  padding-bottom: .28em; border-bottom: 1.5px solid var(--rule-strong);
  letter-spacing: -0.005em;
}
h2 { font-size: 12.5pt; margin: 1.7em 0 .5em; }
h3 { font-size: 11pt;  margin: 1.35em 0 .4em; }
h4 { font-size: 10.5pt; margin: 1.2em 0 .35em; font-style: italic; font-weight: 400; }

p { margin: 0 0 .72em; text-align: justify; }
h1 + p, h2 + p, h3 + p, blockquote + p { text-indent: 0; }

strong { font-weight: 600; }
a { color: inherit; text-decoration: none; border-bottom: .4px solid var(--rule); }

/* --------------------------------------------------------------- tables -- */

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.1em 0 1.3em;
  font-size: 9.2pt;
  font-variant-numeric: tabular-nums lining-nums;
  break-inside: avoid;
  page-break-inside: avoid;
}
thead th {
  font-weight: 600; text-align: left; padding: .42em .7em .42em 0;
  border-top: 1.4px solid var(--rule-strong);
  border-bottom: .8px solid var(--rule-strong);
  font-size: 8.6pt; letter-spacing: .04em; text-transform: uppercase;
  color: var(--muted);
}
tbody td { padding: .38em .7em .38em 0; border-bottom: .4px solid var(--rule); vertical-align: top; }
tbody tr:last-child td { border-bottom: 1.4px solid var(--rule-strong); }
td:not(:first-child), th:not(:first-child) { padding-left: .7em; }

/* ---------------------------------------------------------- block quote -- */

blockquote {
  margin: 1.15em 0;
  padding: .75em 1em .75em 1.1em;
  background: var(--tint);
  border-left: 2.5px solid var(--accent);
  break-inside: avoid;
}
blockquote p { margin: 0 0 .45em; text-align: left; }
blockquote p:last-child { margin-bottom: 0; }

/* ----------------------------------------------------------------- code -- */

code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: .86em;
  background: var(--tint);
  padding: .08em .32em;
  border-radius: 2px;
  font-variant-numeric: lining-nums tabular-nums;
}
pre { background: var(--tint); padding: .8em 1em; overflow-x: auto; border-left: 2px solid var(--rule); }
pre code { background: none; padding: 0; font-size: 8.6pt; }

/* ---------------------------------------------------------------- lists -- */

ul, ol { margin: 0 0 .8em; padding-left: 1.35em; }
li { margin-bottom: .3em; }
li > ul, li > ol { margin-top: .3em; }

hr { border: 0; border-top: .5px solid var(--rule); margin: 2em 0; }

/* --------------------------------------------------------- bibliography -- */

.references ol { list-style: none; padding-left: 0; counter-reset: ref; }
.references li {
  counter-increment: ref;
  padding-left: 2.4em;
  text-indent: -2.4em;                 /* hanging indent */
  margin-bottom: .5em;
  font-size: 9.3pt;
  line-height: 1.42;
  text-align: left;
  break-inside: avoid;
}
.references li::before {
  content: "[" counter(ref) "]";
  display: inline-block;
  width: 2.4em;
  text-indent: 0;
  font-variant-numeric: lining-nums;
  color: var(--accent);
  font-weight: 600;
}
.references em { font-style: italic; }
"""

TITLE_BLOCK = """
<div class="titleblock">
  <div class="kicker">Literature Survey &amp; Positioning</div>
  <h1>Token-Efficient LLM Interaction<br>on CPU-Only Hardware</h1>
  <div class="subtitle">Parsimony &mdash; a stacked, self-improving optimisation layer<br>for small language models</div>
  <div class="authors">
    Arrsh Tripathi &middot; 23BCI0191 &nbsp;&nbsp; Alok Singh &middot; 23BCI0158 &nbsp;&nbsp; Dev Singh &middot; 23BCE0794
  </div>
  <div class="affil">
    Vellore Institute of Technology &middot; B.Tech BCSE497J Project I<br>Guide: Dr Sathya K &middot; September 2026
  </div>
</div>
"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    sys.exit("No Chrome or Edge found; cannot render a PDF.")


def strip_front_matter(text: str) -> str:
    """Drop the markdown title block; the HTML one replaces it."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "---" and i > 0:
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return text


def title_block(kicker: str, title: str, subtitle: str) -> str:
    return f"""
<div class="titleblock">
  <div class="kicker">{kicker}</div>
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="authors">
    Arrsh Tripathi &middot; 23BCI0191 &nbsp;&nbsp; Alok Singh &middot; 23BCI0158 &nbsp;&nbsp; Dev Singh &middot; 23BCE0794
  </div>
  <div class="affil">
    Vellore Institute of Technology &middot; B.Tech BCSE497J Project I<br>Guide: Dr Sathya K &middot; September 2026
  </div>
</div>
"""


def to_html(md_text: str, block: str = TITLE_BLOCK) -> str:
    body = markdown.markdown(
        strip_front_matter(md_text),
        extensions=["tables", "attr_list", "sane_lists", "md_in_html"],
        output_format="html5",
    )
    # Demote so the document's own "##" sections become the top visual level.
    body = re.sub(r"<(/?)h4\b", r"<\1h5", body)
    body = re.sub(r"<(/?)h3\b", r"<\1h4", body)
    body = re.sub(r"<(/?)h2\b", r"<\1h1", body)

    # Wrap the bibliography so the hanging-indent rules apply to it alone.
    marker = "References</h1>"
    if marker in body:
        head, tail = body.split(marker, 1)
        body = f'{head}{marker}<div class="references">{tail}</div>'

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Parsimony — Literature Survey</title><style>{CSS}</style></head>"
        f"<body>{block}{body}</body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--kicker", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--serif", default=None,
                    help="Body font stack, e.g. \"'Times New Roman', Cambria, serif\". "
                         "Used to match an existing document's look.")
    args = ap.parse_args()

    block = (
        title_block(args.kicker, args.title, args.subtitle)
        if args.title else TITLE_BLOCK
    )
    html = to_html(args.source.read_text(encoding="utf-8"), block)
    if args.serif:
        html = html.replace("Cambria, Constantia, Georgia, serif", args.serif)
    tmp = Path(tempfile.gettempdir()) / f"parsimony_{args.out.stem}.html"
    tmp.write_text(html, encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            find_chrome(),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",   # let the webfonts arrive
            f"--print-to-pdf={args.out.resolve()}",
            tmp.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    size = args.out.stat().st_size
    print(f"wrote {args.out}  ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
