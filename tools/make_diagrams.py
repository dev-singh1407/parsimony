"""Render the report's mandatory diagrams to PNG.

The template requires a system architecture figure, a data-flow diagram and a
use-case diagram. There is no drawing library on this machine and installing one
for four pictures is not worth it, so the diagrams are authored as SVG and
screenshotted by headless Chrome -- the same engine already used to typeset the
PDFs. That keeps the toolchain to one dependency and the diagrams to text files
that diff properly in git.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)

FONT = "font-family='Segoe UI, Calibri, sans-serif'"

# --------------------------------------------------------------------------
# 1. System architecture -- the layered view the report's 4.1 describes
# --------------------------------------------------------------------------
ARCHITECTURE = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="760" {FONT}>
<rect width="1180" height="760" fill="#ffffff"/>
<text x="590" y="34" font-size="21" font-weight="700" text-anchor="middle" fill="#16181d">
  Parsimony &#8212; Layered System Architecture</text>
<text x="590" y="56" font-size="13" text-anchor="middle" fill="#5a6069">
  A layer may depend only on layers below it. Enforced by an automated test.</text>

<!-- L5 -->
<rect x="60" y="80" width="1060" height="72" rx="6" fill="#eceff1" stroke="#37474f" stroke-width="1.4"/>
<text x="78" y="106" font-size="14" font-weight="700" fill="#37474f">L5 &#183; SURFACES</text>
<text x="78" y="128" font-size="12.5" fill="#16181d">Command line interface</text>
<text x="78" y="145" font-size="11.5" fill="#5a6069">chat &#183; bench &#183; compare &#183; ask &#183; tour &#183; judge &#183; learning &#183; latency</text>
<rect x="700" y="96" width="400" height="42" rx="4" fill="#ffffff" stroke="#90a4ae"/>
<text x="900" y="113" font-size="11.5" text-anchor="middle" fill="#37474f">User query + conversation history in</text>
<text x="900" y="129" font-size="11.5" text-anchor="middle" fill="#37474f">Response + token ledger out</text>

<!-- L4 -->
<rect x="60" y="164" width="1060" height="78" rx="6" fill="#f3e5f5" stroke="#6a1b9a" stroke-width="1.4"/>
<text x="78" y="190" font-size="14" font-weight="700" fill="#6a1b9a">L4 &#183; EVALUATION HARNESS</text>
<g font-size="11.5" fill="#16181d">
<rect x="78"  y="200" width="190" height="30" rx="4" fill="#fff" stroke="#ba68c8"/><text x="173" y="219" text-anchor="middle">Factorial sweep runner</text>
<rect x="278" y="200" width="180" height="30" rx="4" fill="#fff" stroke="#ba68c8"/><text x="368" y="219" text-anchor="middle">Quality measures</text>
<rect x="468" y="200" width="180" height="30" rx="4" fill="#fff" stroke="#ba68c8"/><text x="558" y="219" text-anchor="middle">Bootstrap statistics</text>
<rect x="658" y="200" width="200" height="30" rx="4" fill="#fff" stroke="#ba68c8"/><text x="758" y="219" text-anchor="middle">Threshold calibration</text>
<rect x="868" y="200" width="232" height="30" rx="4" fill="#fff" stroke="#ba68c8"/><text x="984" y="219" text-anchor="middle">Generalisation &#183; latency &#183; learning</text>
</g>

<!-- L3 -->
<rect x="60" y="254" width="1060" height="66" rx="6" fill="#fff8e1" stroke="#b8860b" stroke-width="1.6"/>
<text x="78" y="280" font-size="14" font-weight="700" fill="#7a5c00">L3 &#183; ORCHESTRATOR</text>
<text x="78" y="302" font-size="12" fill="#16181d">
  Applies stages in the configured order &#183; submits every proposal to the fidelity gate &#183; commits or refuses &#183; writes one ledger row per stage</text>

<!-- L2 -->
<rect x="60" y="332" width="1060" height="176" rx="6" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.4"/>
<text x="78" y="358" font-size="14" font-weight="700" fill="#1b5e20">L2 &#183; OPTIMISATION MODULES &#8212; each returns a PROPOSAL, never mutates the request</text>
<g font-size="11.5">
<rect x="78"  y="370" width="200" height="60" rx="4" fill="#fff" stroke="#4a148c" stroke-width="1.2"/>
<text x="90" y="388" font-weight="700" fill="#4a148c">M1 Compressor</text>
<text x="90" y="404" fill="#5a6069">Tiers 1&#8211;3, sentence aware</text>
<text x="90" y="419" fill="#5a6069">Negative-yield guard</text>
<rect x="288" y="370" width="200" height="60" rx="4" fill="#fff" stroke="#1b5e20" stroke-width="1.2"/>
<text x="300" y="388" font-weight="700" fill="#1b5e20">M2 Semantic cache</text>
<text x="300" y="404" fill="#5a6069">Exact + similarity tiers</text>
<text x="300" y="419" fill="#5a6069">Three-zone policy</text>
<rect x="498" y="370" width="200" height="60" rx="4" fill="#fff" stroke="#0d47a1" stroke-width="1.2"/>
<text x="510" y="388" font-weight="700" fill="#0d47a1">M3 History manager</text>
<text x="510" y="404" fill="#5a6069">Selector + arranger</text>
<text x="510" y="419" fill="#5a6069">4 strategies</text>
<rect x="708" y="370" width="196" height="60" rx="4" fill="#fff" stroke="#b71c1c" stroke-width="1.2"/>
<text x="720" y="388" font-weight="700" fill="#b71c1c">M4 Assembler</text>
<text x="720" y="404" fill="#5a6069">Prefix-stable ordering</text>
<text x="720" y="419" fill="#5a6069">KV-reuse instrument</text>
<rect x="914" y="370" width="186" height="60" rx="4" fill="#fff" stroke="#e65100" stroke-width="1.2"/>
<text x="926" y="388" font-weight="700" fill="#e65100">M5 Output budgeter</text>
<text x="926" y="404" fill="#5a6069">Per-class budget</text>
<text x="926" y="419" fill="#5a6069">Streaming early stop</text>

<rect x="78"  y="440" width="200" height="56" rx="4" fill="#fff" stroke="#37474f" stroke-width="1.2"/>
<text x="90" y="458" font-weight="700" fill="#37474f">M6 Router</text>
<text x="90" y="474" fill="#5a6069">M6a deterministic tier</text>
<text x="90" y="489" fill="#5a6069">M6b escalation tier</text>
<rect x="288" y="440" width="200" height="56" rx="4" fill="#fff" stroke="#00695c" stroke-width="1.2"/>
<text x="300" y="458" font-weight="700" fill="#00695c">M7 Policy learner</text>
<text x="300" y="474" fill="#5a6069">Offline counterfactual</text>
<text x="300" y="489" fill="#5a6069">replay &#8594; PolicyBundle</text>
<rect x="498" y="440" width="602" height="56" rx="4" fill="#fff8e1" stroke="#b8860b" stroke-width="1.6"/>
<text x="510" y="458" font-weight="700" fill="#7a5c00">M8 Fidelity gate &#8212; ALWAYS ON, runs on every proposal from every module</text>
<text x="510" y="474" fill="#5a4a00">Refuses any edit that drops a number, named entity, negation or operative modifier, or empties the payload</text>
<text x="510" y="489" fill="#5a4a00">Measured effect: false-hit rate 51.1% &#8594; 0.0% on the adversarial set</text>
</g>

<!-- L1 -->
<rect x="60" y="520" width="1060" height="76" rx="6" fill="#e3f2fd" stroke="#0d47a1" stroke-width="1.4"/>
<text x="78" y="546" font-size="14" font-weight="700" fill="#0d47a1">L1 &#183; INFRASTRUCTURE</text>
<g font-size="11.5" fill="#16181d">
<rect x="78"  y="556" width="200" height="30" rx="4" fill="#fff" stroke="#64b5f6"/><text x="178" y="575" text-anchor="middle">Tokeniser (Qwen2.5, 151,665)</text>
<rect x="288" y="556" width="180" height="30" rx="4" fill="#fff" stroke="#64b5f6"/><text x="378" y="575" text-anchor="middle">Encoders (content-v1)</text>
<rect x="478" y="556" width="190" height="30" rx="4" fill="#fff" stroke="#64b5f6"/><text x="573" y="575" text-anchor="middle">Unicode sanitisation</text>
<rect x="678" y="556" width="210" height="30" rx="4" fill="#fff" stroke="#64b5f6"/><text x="783" y="575" text-anchor="middle">Providers: Ollama / Mock</text>
<rect x="898" y="556" width="202" height="30" rx="4" fill="#fff" stroke="#64b5f6"/><text x="999" y="575" text-anchor="middle">Generation memoisation</text>
</g>

<!-- L0 -->
<rect x="60" y="608" width="1060" height="62" rx="6" fill="#eceff1" stroke="#37474f" stroke-width="1.4"/>
<text x="78" y="634" font-size="14" font-weight="700" fill="#37474f">L0 &#183; CONTRACTS &#8212; pure types and protocols, no logic, no upward imports</text>
<text x="78" y="656" font-size="12" fill="#16181d">
  Request &#183; Turn &#183; Proposal (ContextPatch | ShortCircuit | NoOp) &#183; LedgerRow &#183; TransformKind &#183; PolicyBundle</text>

<!-- external -->
<rect x="60" y="686" width="520" height="52" rx="6" fill="#ffffff" stroke="#37474f" stroke-width="1.4" stroke-dasharray="4 3"/>
<text x="320" y="708" font-size="13" font-weight="700" text-anchor="middle" fill="#16181d">EXTERNAL &#183; Ollama runtime (CPU only, offline)</text>
<text x="320" y="727" font-size="11.5" text-anchor="middle" fill="#5a6069">qwen2.5:1.5b-instruct, Q4_K_M quantisation</text>
<rect x="600" y="686" width="520" height="52" rx="6" fill="#ffffff" stroke="#37474f" stroke-width="1.4" stroke-dasharray="4 3"/>
<text x="860" y="708" font-size="13" font-weight="700" text-anchor="middle" fill="#16181d">PERSISTENCE &#183; Token ledger (JSONL)</text>
<text x="860" y="727" font-size="11.5" text-anchor="middle" fill="#5a6069">One row per stage per request, including no-ops</text>
</svg>"""

# --------------------------------------------------------------------------
# 2. Data-flow diagram (mandatory) -- level 0 and level 1
# --------------------------------------------------------------------------
DFD = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="690" {FONT}>
<rect width="1180" height="690" fill="#ffffff"/>
<defs>
  <marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#37474f"/></marker>
</defs>
<text x="590" y="32" font-size="20" font-weight="700" text-anchor="middle" fill="#16181d">
  Data Flow Diagram</text>

<!-- ---------------- level 0 ---------------- -->
<text x="60" y="70" font-size="15" font-weight="700" fill="#7a2518">Level 0 &#8212; Context diagram</text>
<rect x="70"  y="90" width="150" height="58" fill="#eceff1" stroke="#37474f" stroke-width="1.6"/>
<text x="145" y="115" font-size="13" font-weight="600" text-anchor="middle" fill="#16181d">User</text>
<text x="145" y="133" font-size="11" text-anchor="middle" fill="#5a6069">external entity</text>

<circle cx="560" cy="119" r="70" fill="#fff8e1" stroke="#b8860b" stroke-width="1.8"/>
<text x="560" y="112" font-size="13" font-weight="700" text-anchor="middle" fill="#7a5c00">0</text>
<text x="560" y="130" font-size="12.5" font-weight="600" text-anchor="middle" fill="#16181d">Parsimony</text>
<text x="560" y="146" font-size="11" text-anchor="middle" fill="#5a6069">middleware</text>

<rect x="930" y="90" width="180" height="58" fill="#eceff1" stroke="#37474f" stroke-width="1.6"/>
<text x="1020" y="115" font-size="13" font-weight="600" text-anchor="middle" fill="#16181d">Local SLM (Ollama)</text>
<text x="1020" y="133" font-size="11" text-anchor="middle" fill="#5a6069">external entity</text>

<line x1="220" y1="106" x2="486" y2="106" stroke="#37474f" marker-end="url(#a)"/>
<text x="353" y="99"  font-size="11.5" text-anchor="middle" fill="#5a6069">query + history</text>
<line x1="486" y1="134" x2="220" y2="134" stroke="#37474f" marker-end="url(#a)"/>
<text x="353" y="150" font-size="11.5" text-anchor="middle" fill="#5a6069">answer + token ledger</text>

<line x1="632" y1="106" x2="926" y2="106" stroke="#37474f" marker-end="url(#a)"/>
<text x="779" y="99"  font-size="11.5" text-anchor="middle" fill="#5a6069">optimised prompt + budget</text>
<line x1="926" y1="134" x2="632" y2="134" stroke="#37474f" marker-end="url(#a)"/>
<text x="779" y="150" font-size="11.5" text-anchor="middle" fill="#5a6069">generated tokens + timings</text>

<line x1="60" y1="200" x2="1120" y2="200" stroke="#c8ccd2"/>

<!-- ---------------- level 1 ---------------- -->
<text x="60" y="232" font-size="15" font-weight="700" fill="#7a2518">Level 1 &#8212; Decomposition of process 0</text>

<rect x="60" y="256" width="120" height="50" fill="#eceff1" stroke="#37474f" stroke-width="1.5"/>
<text x="120" y="286" font-size="12.5" font-weight="600" text-anchor="middle" fill="#16181d">User</text>

<g font-size="11.5">
<circle cx="330" cy="281" r="52" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.5"/>
<text x="330" y="273" font-size="11" font-weight="700" text-anchor="middle" fill="#1b5e20">1.0</text>
<text x="330" y="288" font-weight="600" text-anchor="middle" fill="#16181d">Route</text>
<text x="330" y="302" text-anchor="middle" fill="#5a6069">M6a</text>

<circle cx="500" cy="281" r="52" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.5"/>
<text x="500" y="273" font-size="11" font-weight="700" text-anchor="middle" fill="#1b5e20">2.0</text>
<text x="500" y="288" font-weight="600" text-anchor="middle" fill="#16181d">Look up</text>
<text x="500" y="302" text-anchor="middle" fill="#5a6069">M2</text>

<circle cx="670" cy="281" r="52" fill="#e3f2fd" stroke="#0d47a1" stroke-width="1.5"/>
<text x="670" y="273" font-size="11" font-weight="700" text-anchor="middle" fill="#0d47a1">3.0</text>
<text x="670" y="288" font-weight="600" text-anchor="middle" fill="#16181d">Trim</text>
<text x="670" y="302" text-anchor="middle" fill="#5a6069">M3</text>

<circle cx="840" cy="281" r="52" fill="#f3e5f5" stroke="#4a148c" stroke-width="1.5"/>
<text x="840" y="273" font-size="11" font-weight="700" text-anchor="middle" fill="#4a148c">4.0</text>
<text x="840" y="288" font-weight="600" text-anchor="middle" fill="#16181d">Compress</text>
<text x="840" y="302" text-anchor="middle" fill="#5a6069">M1</text>

<circle cx="1010" cy="281" r="52" fill="#ffebee" stroke="#b71c1c" stroke-width="1.5"/>
<text x="1010" y="273" font-size="11" font-weight="700" text-anchor="middle" fill="#b71c1c">5.0</text>
<text x="1010" y="288" font-weight="600" text-anchor="middle" fill="#16181d">Assemble</text>
<text x="1010" y="302" text-anchor="middle" fill="#5a6069">M4</text>

<circle cx="840" cy="452" r="52" fill="#fff3e0" stroke="#e65100" stroke-width="1.5"/>
<text x="840" y="444" font-size="11" font-weight="700" text-anchor="middle" fill="#e65100">6.0</text>
<text x="840" y="459" font-weight="600" text-anchor="middle" fill="#16181d">Budget</text>
<text x="840" y="473" text-anchor="middle" fill="#5a6069">M5</text>

<circle cx="1010" cy="452" r="52" fill="#eceff1" stroke="#37474f" stroke-width="1.5"/>
<text x="1010" y="444" font-size="11" font-weight="700" text-anchor="middle" fill="#37474f">7.0</text>
<text x="1010" y="459" font-weight="600" text-anchor="middle" fill="#16181d">Generate</text>
<text x="1010" y="473" text-anchor="middle" fill="#5a6069">via Ollama</text>

<circle cx="560" cy="452" r="56" fill="#fff8e1" stroke="#b8860b" stroke-width="1.8"/>
<text x="560" y="443" font-size="11" font-weight="700" text-anchor="middle" fill="#7a5c00">8.0</text>
<text x="560" y="459" font-weight="600" text-anchor="middle" fill="#16181d">Verify</text>
<text x="560" y="473" text-anchor="middle" fill="#5a6069">M8 gate</text>
</g>

<line x1="180" y1="281" x2="274" y2="281" stroke="#37474f" marker-end="url(#a)"/>
<text x="227" y="274" font-size="10.5" text-anchor="middle" fill="#5a6069">query</text>
<line x1="382" y1="281" x2="444" y2="281" stroke="#37474f" marker-end="url(#a)"/>
<line x1="552" y1="281" x2="614" y2="281" stroke="#37474f" marker-end="url(#a)"/>
<line x1="722" y1="281" x2="784" y2="281" stroke="#37474f" marker-end="url(#a)"/>
<line x1="892" y1="281" x2="954" y2="281" stroke="#37474f" marker-end="url(#a)"/>
<line x1="1010" y1="333" x2="1010" y2="396" stroke="#37474f" marker-end="url(#a)"/>
<line x1="954" y1="452" x2="896" y2="452" stroke="#37474f" marker-end="url(#a)"/>
<text x="925" y="444" font-size="10.5" text-anchor="middle" fill="#5a6069">cap</text>
<line x1="788" y1="452" x2="620" y2="452" stroke="#37474f" marker-end="url(#a)"/>
<line x1="504" y1="452" x2="180" y2="452" stroke="#37474f" marker-end="url(#a)"/>
<text x="342" y="444" font-size="10.5" text-anchor="middle" fill="#5a6069">verified answer to user</text>
<line x1="120" y1="306" x2="120" y2="452" stroke="#37474f"/>

<!-- gate feedback -->
<path d="M560 396 L560 360 L330 360 L330 333" stroke="#b8860b" stroke-width="1.4" fill="none" stroke-dasharray="4 3" marker-end="url(#a)"/>
<text x="445" y="354" font-size="10.5" text-anchor="middle" fill="#7a5c00">refuse &#8594; proposal not committed (applies to processes 2.0&#8211;6.0)</text>

<!-- data stores -->
<g font-size="11.5">
<rect x="60" y="560" width="300" height="40" fill="#f6f4ef" stroke="#37474f" stroke-width="1.4"/>
<line x1="60" y1="560" x2="60" y2="600" stroke="#37474f" stroke-width="4"/>
<text x="78" y="585" font-weight="600" fill="#16181d">D1 &#183; Semantic cache store</text>
<rect x="400" y="560" width="300" height="40" fill="#f6f4ef" stroke="#37474f" stroke-width="1.4"/>
<line x1="400" y1="560" x2="400" y2="600" stroke="#37474f" stroke-width="4"/>
<text x="418" y="585" font-weight="600" fill="#16181d">D2 &#183; Token ledger (JSONL)</text>
<rect x="740" y="560" width="360" height="40" fill="#f6f4ef" stroke="#37474f" stroke-width="1.4"/>
<line x1="740" y1="560" x2="740" y2="600" stroke="#37474f" stroke-width="4"/>
<text x="758" y="585" font-weight="600" fill="#16181d">D3 &#183; PolicyBundle (mined from past chats)</text>
</g>
<line x1="500" y1="504" x2="210" y2="556" stroke="#37474f" marker-end="url(#a)"/>
<text x="330" y="524" font-size="10.5" fill="#5a6069">read / write past Q&amp;A</text>
<line x1="560" y1="508" x2="550" y2="556" stroke="#37474f" marker-end="url(#a)"/>
<text x="600" y="534" font-size="10.5" fill="#5a6069">one row per stage</text>
<line x1="900" y1="556" x2="880" y2="504" stroke="#37474f" marker-end="url(#a)"/>
<text x="930" y="534" font-size="10.5" fill="#5a6069">warm start (M7, offline)</text>

<text x="60" y="650" font-size="11.5" fill="#5a6069">
  Note: processes 1.0 and 2.0 may terminate the flow early &#8212; a deterministic answer or a verified cache hit returns to the user without reaching 7.0.</text>
<text x="60" y="670" font-size="11.5" fill="#5a6069">
  Every process writes to D2 whether or not it changed anything, so the trace is complete and the evaluation is auditable.</text>
</svg>"""

# --------------------------------------------------------------------------
# 3. Use-case diagram (mandatory)
# --------------------------------------------------------------------------
USECASE = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="720" {FONT}>
<rect width="1120" height="720" fill="#ffffff"/>
<defs>
  <marker id="b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#7a2518"/></marker>
  <marker id="c" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto">
    <path d="M0,0 L9,3.5 L0,7" fill="none" stroke="#5a6069"/></marker>
</defs>
<text x="560" y="32" font-size="20" font-weight="700" text-anchor="middle" fill="#16181d">Use Case Diagram</text>

<!-- system boundary -->
<rect x="270" y="60" width="580" height="620" rx="6" fill="#fbfaf7" stroke="#37474f" stroke-width="1.6"/>
<text x="560" y="86" font-size="14" font-weight="700" text-anchor="middle" fill="#37474f">Parsimony middleware</text>

<!-- actor: end user -->
<g stroke="#16181d" stroke-width="1.8" fill="none">
  <circle cx="120" cy="180" r="16" fill="#fff"/>
  <line x1="120" y1="196" x2="120" y2="240"/>
  <line x1="92"  y1="212" x2="148" y2="212"/>
  <line x1="120" y1="240" x2="98"  y2="278"/>
  <line x1="120" y1="240" x2="142" y2="278"/>
</g>
<text x="120" y="300" font-size="13" font-weight="600" text-anchor="middle" fill="#16181d">End user</text>

<!-- actor: researcher -->
<g stroke="#16181d" stroke-width="1.8" fill="none">
  <circle cx="120" cy="450" r="16" fill="#fff"/>
  <line x1="120" y1="466" x2="120" y2="510"/>
  <line x1="92"  y1="482" x2="148" y2="482"/>
  <line x1="120" y1="510" x2="98"  y2="548"/>
  <line x1="120" y1="510" x2="142" y2="548"/>
</g>
<text x="120" y="570" font-size="13" font-weight="600" text-anchor="middle" fill="#16181d">Researcher</text>
<text x="120" y="588" font-size="11" text-anchor="middle" fill="#5a6069">(evaluator)</text>

<!-- actor: local model -->
<g stroke="#16181d" stroke-width="1.8" fill="none">
  <circle cx="1000" cy="330" r="16" fill="#fff"/>
  <line x1="1000" y1="346" x2="1000" y2="390"/>
  <line x1="972"  y1="362" x2="1028" y2="362"/>
  <line x1="1000" y1="390" x2="978"  y2="428"/>
  <line x1="1000" y1="390" x2="1022" y2="428"/>
</g>
<text x="1000" y="450" font-size="13" font-weight="600" text-anchor="middle" fill="#16181d">Local SLM</text>
<text x="1000" y="468" font-size="11" text-anchor="middle" fill="#5a6069">(Ollama, CPU)</text>

<!-- use cases -->
<g font-size="12">
<ellipse cx="470" cy="130" rx="150" ry="30" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.4"/>
<text x="470" y="127" text-anchor="middle" font-weight="600" fill="#16181d">UC1 Ask a question</text>
<text x="470" y="143" text-anchor="middle" font-size="10.5" fill="#5a6069">optimised end to end</text>

<ellipse cx="470" cy="205" rx="150" ry="30" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.4"/>
<text x="470" y="202" text-anchor="middle" font-weight="600" fill="#16181d">UC2 Reuse a past answer</text>
<text x="470" y="218" text-anchor="middle" font-size="10.5" fill="#5a6069">verified cache hit</text>

<ellipse cx="470" cy="280" rx="150" ry="30" fill="#e8f5e9" stroke="#1b5e20" stroke-width="1.4"/>
<text x="470" y="277" text-anchor="middle" font-weight="600" fill="#16181d">UC3 Answer without the model</text>
<text x="470" y="293" text-anchor="middle" font-size="10.5" fill="#5a6069">arithmetic / date routing</text>

<ellipse cx="470" cy="368" rx="150" ry="30" fill="#e3f2fd" stroke="#0d47a1" stroke-width="1.4"/>
<text x="470" y="365" text-anchor="middle" font-weight="600" fill="#16181d">UC4 Run a factorial sweep</text>
<text x="470" y="381" text-anchor="middle" font-size="10.5" fill="#5a6069">17 configurations</text>

<ellipse cx="470" cy="443" rx="150" ry="30" fill="#e3f2fd" stroke="#0d47a1" stroke-width="1.4"/>
<text x="470" y="440" text-anchor="middle" font-weight="600" fill="#16181d">UC5 Calibrate a threshold</text>
<text x="470" y="456" text-anchor="middle" font-size="10.5" fill="#5a6069">adversarial sweep</text>

<ellipse cx="470" cy="518" rx="150" ry="30" fill="#e3f2fd" stroke="#0d47a1" stroke-width="1.4"/>
<text x="470" y="515" text-anchor="middle" font-weight="600" fill="#16181d">UC6 Inspect the ledger</text>
<text x="470" y="531" text-anchor="middle" font-size="10.5" fill="#5a6069">audit every decision</text>

<ellipse cx="470" cy="600" rx="150" ry="30" fill="#e0f2f1" stroke="#00695c" stroke-width="1.4"/>
<text x="470" y="597" text-anchor="middle" font-weight="600" fill="#16181d">UC7 Mine past conversations</text>
<text x="470" y="613" text-anchor="middle" font-size="10.5" fill="#5a6069">offline warm start</text>

<ellipse cx="740" cy="243" rx="86" ry="27" fill="#fff8e1" stroke="#b8860b" stroke-width="1.5"/>
<text x="740" y="240" text-anchor="middle" font-weight="600" fill="#7a5c00">UC8 Verify</text>
<text x="740" y="255" text-anchor="middle" font-size="10.5" fill="#5a4a00">fidelity gate</text>

<ellipse cx="740" cy="330" rx="86" ry="27" fill="#eceff1" stroke="#37474f" stroke-width="1.4"/>
<text x="740" y="327" text-anchor="middle" font-weight="600" fill="#16181d">UC9 Generate</text>
<text x="740" y="342" text-anchor="middle" font-size="10.5" fill="#5a6069">call the model</text>
</g>

<!-- associations -->
<g stroke="#7a2518" stroke-width="1.2">
<line x1="152" y1="196" x2="320" y2="136" marker-end="url(#b)"/>
<line x1="152" y1="212" x2="320" y2="205" marker-end="url(#b)"/>
<line x1="152" y1="230" x2="320" y2="274" marker-end="url(#b)"/>
<line x1="152" y1="470" x2="320" y2="372" marker-end="url(#b)"/>
<line x1="152" y1="482" x2="320" y2="445" marker-end="url(#b)"/>
<line x1="152" y1="496" x2="320" y2="516" marker-end="url(#b)"/>
<line x1="152" y1="516" x2="320" y2="596" marker-end="url(#b)"/>
<line x1="968" y1="350" x2="826" y2="332" marker-end="url(#b)"/>
</g>

<!-- include relationships -->
<g stroke="#5a6069" stroke-width="1.1" stroke-dasharray="5 3" font-size="10" fill="#5a6069">
<line x1="620" y1="140" x2="660" y2="232" marker-end="url(#c)"/>
<text x="655" y="182" >&#171;include&#187;</text>
<line x1="620" y1="212" x2="656" y2="240" marker-end="url(#c)"/>
<line x1="620" y1="145" x2="662" y2="322" marker-end="url(#c)"/>
<text x="600" y="300">&#171;include&#187;</text>
</g>

<text x="290" y="666" font-size="11" fill="#5a6069">
  UC1 includes UC8 and UC9. UC2 includes UC8: no cached answer is returned unless the verifier passes it.</text>
</svg>"""


# --------------------------------------------------------------------------
# 4. Gantt chart -- section 2.5 asks for one explicitly
# --------------------------------------------------------------------------
def _gantt() -> str:
    tasks = [
        ("Literature survey: 40 papers, limitations, six gaps", 1, 3, "#4a148c"),
        ("Contracts (L0) and infrastructure (L1)", 3, 5, "#0d47a1"),
        ("Modules M1-M8 and the orchestrator", 4, 8, "#1b5e20"),
        ("Corpus: 151 conversations, 45 pairs, 40 gold items", 5, 8, "#00695c"),
        ("Evaluation harness and factorial sweep runner", 7, 10, "#6a1b9a"),
        ("Experiments E1-E5 and statistical analysis", 9, 12, "#b71c1c"),
        ("Attach the real model; re-measure latency and quality", 11, 13, "#e65100"),
        ("Reporting: report, paper, reproducibility package", 12, 15, "#37474f"),
        ("Energy instrumentation, per-cell quality (Project-II)", 15, 17, "#9e9e9e"),
    ]
    x0, colw, y0, rowh = 430, 42.0, 96, 34
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="{y0 + rowh * len(tasks) + 50}" {FONT}>',
           f'<rect width="1180" height="{y0 + rowh * len(tasks) + 50}" fill="#ffffff"/>',
           '<text x="590" y="34" font-size="20" font-weight="700" text-anchor="middle"'
           ' fill="#16181d">Project Plan &#8212; Gantt Chart</text>',
           '<text x="590" y="56" font-size="12" text-anchor="middle" fill="#5a6069">'
           'Weeks 1&#8211;16 of the Project-I schedule. The final bar is carried into Project-II.</text>']
    for w in range(1, 18):
        x = x0 + (w - 1) * colw
        out.append(f'<line x1="{x}" y1="{y0 - 18}" x2="{x}" y2="{y0 + rowh * len(tasks)}"'
                   f' stroke="#e0e0e0" stroke-width="1"/>')
        if w < 17:
            out.append(f'<text x="{x + colw / 2}" y="{y0 - 24}" font-size="10.5"'
                       f' text-anchor="middle" fill="#5a6069">W{w}</text>')
    for i, (name, start, end, colour) in enumerate(tasks):
        y = y0 + i * rowh
        out.append(f'<text x="24" y="{y + 21}" font-size="11.5" fill="#16181d">{name}</text>')
        bx = x0 + (start - 1) * colw
        bw = (end - start + 1) * colw - 6
        out.append(f'<rect x="{bx + 3}" y="{y + 7}" width="{bw}" height="18" rx="4"'
                   f' fill="{colour}" opacity="0.85"/>')
        out.append(f'<text x="{bx + bw / 2 + 3}" y="{y + 20}" font-size="10"'
                   f' text-anchor="middle" fill="#ffffff">W{start}&#8211;W{end}</text>')
        out.append(f'<line x1="24" y1="{y + 31}" x2="1150" y2="{y + 31}"'
                   f' stroke="#f0f0f0" stroke-width="1"/>')
    out.append('</svg>')
    return "\n".join(out)


GANTT = _gantt()

# --------------------------------------------------------------------------
# 5. Sequence diagram -- section 4.2.4
# --------------------------------------------------------------------------
SEQUENCE = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="720" {FONT}>
<rect width="1180" height="720" fill="#ffffff"/>
<defs>
  <marker id="s" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#37474f"/></marker>
  <marker id="sr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#b8860b"/></marker>
</defs>
<text x="590" y="30" font-size="20" font-weight="700" text-anchor="middle" fill="#16181d">
  Sequence Diagram &#8212; one request through the pipeline</text>

<g font-size="11.5" font-weight="700" fill="#16181d">
<rect x="40"  y="52" width="130" height="34" rx="4" fill="#eceff1" stroke="#37474f"/><text x="105" y="74" text-anchor="middle">User / CLI</text>
<rect x="230" y="52" width="150" height="34" rx="4" fill="#fff8e1" stroke="#b8860b"/><text x="305" y="74" text-anchor="middle">Orchestrator</text>
<rect x="440" y="52" width="150" height="34" rx="4" fill="#e8f5e9" stroke="#1b5e20"/><text x="515" y="74" text-anchor="middle">Module M1..M6</text>
<rect x="650" y="52" width="150" height="34" rx="4" fill="#fff8e1" stroke="#b8860b"/><text x="725" y="74" text-anchor="middle">M8 Fidelity gate</text>
<rect x="860" y="52" width="140" height="34" rx="4" fill="#eceff1" stroke="#37474f"/><text x="930" y="74" text-anchor="middle">Ollama (CPU)</text>
<rect x="1030" y="52" width="120" height="34" rx="4" fill="#f6f4ef" stroke="#37474f"/><text x="1090" y="74" text-anchor="middle">Ledger</text>
</g>

<g stroke="#b0bec5" stroke-width="1" stroke-dasharray="4 4">
<line x1="105" y1="86" x2="105" y2="690"/><line x1="305" y1="86" x2="305" y2="690"/>
<line x1="515" y1="86" x2="515" y2="690"/><line x1="725" y1="86" x2="725" y2="690"/>
<line x1="930" y1="86" x2="930" y2="690"/><line x1="1090" y1="86" x2="1090" y2="690"/>
</g>

<g font-size="10.5" fill="#37474f" stroke="#37474f" stroke-width="1.2">
<line x1="105" y1="118" x2="299" y2="118" marker-end="url(#s)"/>
<text x="112" y="112" stroke="none">1: Request(query, history)</text>

<line x1="305" y1="152" x2="509" y2="152" marker-end="url(#s)"/>
<text x="312" y="146" stroke="none">2: propose(request)   [for each stage, in configured order]</text>

<line x1="509" y1="186" x2="311" y2="186" marker-end="url(#s)"/>
<text x="330" y="180" stroke="none">3: Proposal = ContextPatch | ShortCircuit | NoOp</text>

<line x1="305" y1="220" x2="719" y2="220" marker-end="url(#s)"/>
<text x="312" y="214" stroke="none">4: check(before, after)   [skipped for NoOp]</text>
</g>

<rect x="240" y="238" width="700" height="86" rx="4" fill="none" stroke="#b8860b" stroke-dasharray="5 3"/>
<text x="252" y="256" font-size="10.5" font-weight="700" fill="#7a5c00">alt  [invariants preserved]</text>
<g font-size="10.5" fill="#7a5c00" stroke="#b8860b" stroke-width="1.2">
<line x1="719" y1="276" x2="311" y2="276" marker-end="url(#sr)"/>
<text x="330" y="270" stroke="none">5a: accept &#8594; orchestrator commits the patch to the Request</text>
</g>
<line x1="240" y1="288" x2="940" y2="288" stroke="#b8860b" stroke-dasharray="3 3"/>
<text x="252" y="304" font-size="10.5" font-weight="700" fill="#7a5c00">else  [a number, entity, negation or modifier would be lost]</text>
<g font-size="10.5" fill="#7a5c00" stroke="#b8860b" stroke-width="1.2">
<line x1="719" y1="318" x2="311" y2="318" marker-end="url(#sr)"/>
<text x="330" y="312" stroke="none">5b: refuse &#8594; proposal discarded, Request unchanged</text>
</g>

<rect x="240" y="340" width="700" height="80" rx="4" fill="none" stroke="#1b5e20" stroke-dasharray="5 3"/>
<text x="252" y="358" font-size="10.5" font-weight="700" fill="#1b5e20">opt  [M6a solved it, or M2 returned a verified hit]</text>
<g font-size="10.5" fill="#1b5e20" stroke="#1b5e20" stroke-width="1.2">
<line x1="305" y1="382" x2="111" y2="382" marker-end="url(#s)"/>
<text x="120" y="376" stroke="none">6: ShortCircuit &#8212; answer returned, model never called (0 model tokens)</text>
<line x1="305" y1="406" x2="1084" y2="406" marker-end="url(#s)"/>
<text x="330" y="400" stroke="none">7: write LedgerRow(outcome = short-circuit)</text>
</g>

<g font-size="10.5" fill="#37474f" stroke="#37474f" stroke-width="1.2">
<line x1="305" y1="446" x2="924" y2="446" marker-end="url(#s)"/>
<text x="312" y="440" stroke="none">8: generate(assembled prompt, num_predict from M5)</text>

<line x1="924" y1="486" x2="311" y2="486" marker-end="url(#s)"/>
<text x="330" y="480" stroke="none">9: streamed tokens + prompt_eval_duration / eval_duration</text>
</g>

<rect x="240" y="504" width="700" height="46" rx="4" fill="none" stroke="#e65100" stroke-dasharray="5 3"/>
<text x="252" y="522" font-size="10.5" font-weight="700" fill="#e65100">loop  [while streaming]</text>
<text x="330" y="540" font-size="10.5" fill="#e65100">10: M5 early stop when trigram novelty falls below threshold</text>

<g font-size="10.5" fill="#37474f" stroke="#37474f" stroke-width="1.2">
<line x1="305" y1="576" x2="1084" y2="576" marker-end="url(#s)"/>
<text x="312" y="570" stroke="none">11: write one LedgerRow per stage &#8212; including stages that did nothing</text>

<line x1="305" y1="610" x2="111" y2="610" marker-end="url(#s)"/>
<text x="120" y="604" stroke="none">12: answer + token ledger</text>
</g>

<text x="40" y="654" font-size="11" fill="#5a6069">
  Steps 2 to 5 repeat for every stage in the configured order. The gate is consulted on every proposal, so no edit reaches the prompt unchecked.</text>
<text x="40" y="674" font-size="11" fill="#5a6069">
  Because a module only ever returns a proposal, disabling one removes its effect entirely &#8212; which is what makes the factorial ablation sound.</text>
</svg>"""


def find_chrome() -> str:
    for path in CHROME:
        if Path(path).exists():
            return path
    sys.exit("No Chrome or Edge found; cannot render diagrams.")


def render(svg: str, out: Path, size: tuple[int, int]) -> None:
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>html,body{{margin:0;padding:0;background:#fff}}</style></head>"
            f"<body>{svg}</body></html>")
    tmp = Path(tempfile.gettempdir()) / f"pz_{out.stem}.html"
    tmp.write_text(html, encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [find_chrome(), "--headless", "--disable-gpu", "--hide-scrollbars",
         "--default-background-color=FFFFFFFF",
         f"--window-size={size[0]},{size[1]}",
         f"--screenshot={out.resolve()}", tmp.resolve().as_uri()],
        check=True, capture_output=True, timeout=120)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "figures" / "diagrams"
    render(ARCHITECTURE, out / "architecture.png", (1180, 760))
    render(DFD, out / "dfd.png", (1180, 690))
    render(USECASE, out / "usecase.png", (1120, 720))
    render(GANTT, out / "gantt.png", (1180, 452))
    render(SEQUENCE, out / "sequence.png", (1180, 700))


if __name__ == "__main__":
    main()
