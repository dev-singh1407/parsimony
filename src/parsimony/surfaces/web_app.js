/* Parsimony visualiser.
 *
 * Everything drawn here comes from an event the orchestrator emitted while it
 * ran: the stack from the stage registry, the token counts from the ledger,
 * each layer's sentence from the same `_reason()` the terminal prints, and the
 * life of every sentence from the before/after text each stage recorded.
 *
 * Two things are NOT measurements, and both are labelled on screen:
 *   - the pacing. The middleware takes ~100 ms end to end, which is too fast to
 *     watch, so the transport holds each stage for a beat. Every duration shown
 *     is the stage's own measured one; the segment WIDTHS are its real share of
 *     middleware time, so the picture stays honest even while it is slowed.
 *   - the prefill seconds, when the model is simulated or the runtime served a
 *     prompt from cache. Those say "estimated" rather than quietly counting.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");
const secs = (s) => (s >= 1 ? s.toFixed(1) + "s" : Math.round(s * 1000) + "ms");
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ══════════════════════════ tabs ══════════════════════════════════ */
const TABS = { pipe: "view-pipe", map: "view-map", ab: "view-ab", demo: "view-demo" };
function show(which) {
  for (const [k, v] of Object.entries(TABS)) {
    $("tab-" + k).setAttribute("aria-selected", String(k === which));
    $(v).hidden = k !== which;
  }
  if (which === "demo") refreshSession();
  if (which === "ab") drawChart();
}
for (const k of Object.keys(TABS)) $("tab-" + k).onclick = () => show(k);

/* ══════════════════════════ header ════════════════════════════════ */
(function connect() {
  const giveUp = setTimeout(() => {
    $("state").textContent = "not connected — no server on port " + (location.port || "80");
    $("state").style.color = "var(--drop)";
  }, 3000);
  fetch("/api/state").then((r) => r.json()).then((s) => {
    clearTimeout(giveUp);
    $("state").textContent = (s.simulated ? "simulated model" : s.model)
      + "  ·  encoder " + s.encoder;
    $("state").style.color = s.simulated ? "var(--warn)" : "";
  }).catch(() => {
    clearTimeout(giveUp);
    $("state").textContent = "not connected — restart `parsimony web`";
    $("state").style.color = "var(--drop)";
  });
})();

async function refreshSession() {
  try {
    const t = await (await fetch("/api/session")).json();
    // Before the first request every one of these is an em dash, not a zero:
    // the line under them already says nothing has run, and six counters
    // claiming to have counted nothing contradicted it. After a run a zero is
    // a real measurement and stays -- no edits refused is a result.
    if (!t.requests) {
      for (const id of ["c-tokens", "c-secs", "d-requests", "d-nomodel",
                        "d-gate", "d-refused", "d-kv"]) {
        $(id).textContent = "—";
      }
      $("demo-sub").textContent = "nothing run yet — use the Pipeline or A/B tab";
      // What the counters will mean is worth having before they mean anything;
      // whether the rate was timed here or estimated needs a run to answer, so
      // that half waits.
      $("demo-basis").innerHTML =
        "<b>Tokens pruned</b> will be measured exactly: what the prompt was written with, "
        + "minus what was sent. <b>Seconds saved</b> is those pruned tokens at the prefill "
        + "rate, and this line will say whether that rate was timed on this machine or "
        + "taken from the project's recorded one.";
      $("demo-extra").innerHTML = "";
      return;
    }
    $("c-tokens").textContent = fmt(t.tokens_pruned);
    $("c-secs").textContent = t.seconds_saved.toFixed(1) + "s";
    $("c-secs-k").textContent = t.timed_here ? "seconds saved" : "seconds saved (estimated)";
    $("demo-basis").innerHTML =
      "<b>Tokens pruned</b> is measured exactly: what the prompt was written with, minus "
      + "what was sent. <b>Seconds saved</b> is those pruned tokens at the prefill rate — "
      + (t.timed_here
        ? "the milliseconds per token this machine actually took on the prompt it did read."
        : "<b>and that rate was not timed here.</b> Either the model is simulated or the "
          + "runtime served a prompt from its cache, so the figure falls back to the "
          + "project's recorded rate and is an estimate.");
    $("d-requests").textContent = fmt(t.requests);
    $("d-nomodel").textContent = fmt(t.without_model || 0);
    $("d-gate").textContent = fmt(t.gate_checked || 0);
    $("d-refused").textContent = fmt(t.gate_refused || 0);
    $("d-kv").textContent = t.kv_mb_saved ? t.kv_mb_saved.toFixed(0) + " MB" : "—";
    $("demo-extra").innerHTML = t.requests
      ? "<b>Answered without the model</b> counts requests the calculator or the cache settled "
        + "outright: zero prompt tokens, zero generated. <b>Edits the gate refused</b> are "
        + "savings the system declined in order to stay correct, which is the number a "
        + "compression project is least likely to show you. <b>KV cache never allocated</b> is "
        + "the pruned tokens priced at this model's own key-value footprint, read from the "
        + "runtime's metadata rather than assumed; it shows a dash where the model cannot "
        + "report it."
      : "";
    $("demo-sub").textContent = t.requests
      ? `${t.requests} request${t.requests > 1 ? "s" : ""} — ${fmt(t.tokens_written)} tokens `
        + `written, ${fmt(t.tokens_sent)} sent (${t.percent.toFixed(0)}% never read)`
      : "nothing run yet — use the Pipeline or A/B tab";
  } catch (e) { /* server gone; keep the last honest figure */ }
}

/* Counters must arrive even when the animation cannot run: rAF is suspended in
   a background tab, which otherwise leaves every headline reading 0. */
function countTo(el, target, suffix) {
  const show2 = (v) => { el.textContent = suffix === "s" ? v.toFixed(1) + "s"
                                                         : fmt(Math.round(v)); };
  const from = parseFloat((el.textContent || "0").replace(/[^\d.]/g, "")) || 0;
  const dur = 600;
  if (document.hidden) { show2(target); return; }
  clearTimeout(el._safety);
  el._safety = setTimeout(() => show2(target), dur + 150);
  const t0 = performance.now();
  (function step(now) {
    const k = Math.min(1, (now - t0) / dur);
    show2(from + (target - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  })(performance.now());
}

/* ══════════════════════ the pipeline view ═════════════════════════ */

const RUN = {
  plan: [],          // stage descriptors, from the registry
  stages: [],        // what each stage did, in order, as it ran
  layers: [],        // the terminal's own table rows
  units: [],         // every sentence, and the stage that removed it
  summary: null,
  at: -1,            // scrub position: -1 = before anything ran
  selected: null,    // stage the inspector is pinned to, else follows `at`
  playing: false,
  timer: null,
  live: false,       // true while the server is still streaming
};

const OUTCOME_MARK = {
  applied: ["✓", "ok"], noop: ["·", "no"], skipped: ["·", "no"],
  reverted: ["✋", "stop"], short_circuit: ["★", "star"], error: ["!", "stop"],
  not_implemented: ["·", "no"], not_reached: ["·", "no"],
};
const OUTCOME_WORD = {
  applied: "changed the request", noop: "chose to do nothing", skipped: "not applicable",
  reverted: "blocked by the gate", short_circuit: "answered it here",
  error: "errored", not_implemented: "not built", not_reached: "never reached",
};

/* ── the stack, drawn downward ───────────────────────────────────── */
function drawStack() {
  const rows = [];
  let prevModule = null;
  RUN.plan.forEach((s, i) => {
    const firstOfModule = s.module !== prevModule;
    prevModule = s.module;
    rows.push(
      `<div class="vrail${firstOfModule ? " first" : ""}">`
      + `${firstOfModule ? esc(s.module) : ""}</div>`
      + `<div class="vstage" id="vs-${s.name}" data-stage="${esc(s.name)}">`
      + `<div class="vs-head"><span class="vs-mod">${esc(s.module)}</span>`
      + `<span class="vs-name">${esc(s.label)}</span>`
      + `<span class="vs-time" id="vt-${s.name}"></span></div>`
      + `<div class="vs-job">${esc(s.does)}</div>`
      + `<div class="vs-out"><span id="vk-${s.name}">—</span>`
      + `<span class="vs-delta" id="vd-${s.name}"></span>`
      + `<span class="vs-gate" id="vg-${s.name}"></span></div>`
      + `<div class="vs-why" id="vw-${s.name}"></div></div>`);
    if (i < RUN.plan.length - 1) {
      rows.push(`<div class="vrail"></div>`
        + `<div class="vflow" id="vf-${i}"><div class="vband"></div>`
        + `<div class="vline"></div><div class="vhead"></div></div>`);
    }
  });
  $("stack").className = "vstack";
  $("stack").innerHTML = rows.join("");
  for (const s of RUN.plan) {
    const el = $("vs-" + s.name);
    if (el) el.onclick = () => { RUN.selected = s.name; paintAll(); };
  }
}

/* The gate, per stage. Three states, and the dim one matters as much as the
   others: a stage that proposed nothing never reached M8, and saying "passed"
   there would claim a check that did not happen. */
const GATE = {
  applied: ["checked", "ok", "M8 inspected this edit and let it through"],
  reverted: ["refused", "stop", "M8 blocked this edit: it would have changed the meaning"],
  short_circuit: ["not asked", "dim", "answered before any edit was proposed"],
};

function paintGate(name, stage) {
  const el = $("vg-" + name);
  if (!el) return;
  if (!stage) { el.textContent = ""; el.className = "vs-gate"; return; }
  const [word, cls, why] = GATE[stage.outcome]
    || ["not asked", "dim", "this layer proposed no edit, so the gate was never consulted"];
  el.textContent = "M8 " + word;
  el.className = "vs-gate " + cls;
  el.title = why;
}

/* How often the gate was consulted this request, and what it did. The page
   claimed a safety component; this is the line that shows it working. */
function gateSummary() {
  const asked = RUN.stages.filter((s) => s.outcome === "applied" || s.outcome === "reverted");
  const refused = asked.filter((s) => s.outcome === "reverted");
  if (!RUN.stages.length) return "";
  if (!asked.length) return "No layer proposed an edit, so the fidelity gate was not consulted.";
  const names = refused.map((s) => labelFor(s.name)).join(", ");
  return refused.length
    ? `The fidelity gate checked ${asked.length} proposed edit`
      + `${asked.length === 1 ? "" : "s"} and refused ${refused.length} — ${names}. `
      + `A refusal is the system declining tokens to stay correct.`
    : `The fidelity gate checked ${asked.length} proposed edit`
      + `${asked.length === 1 ? "" : "s"} and let all of them through: every kept sentence is `
      + `verbatim and in order, and nothing the question names was lost.`;
}

/* Words for the channels, taken from the reader's own text. A stream of dots
   would show that something moves without showing what. */
function wordPool() {
  const alive = [], cut = {};
  for (const u of RUN.units) {
    const words = u.text.split(/\s+/).filter((w) => w.length > 3 && /[A-Za-z]/.test(w))
      .map((w) => w.replace(/^[^\w£€$]+|[^\w%]+$/g, "")).filter(Boolean);
    if (!words.length) continue;
    if (u.removed_at) (cut[u.removed_at] ||= []).push(...words);
    else alive.push(...words);
  }
  return { alive, cut };
}

const MAX_WORDS = 11;
/* Width of one character of the chip font, measured once off the page rather
   than guessed: the cap below is derived from it, so changing the font cannot
   quietly make every chip wider than the lane holding it. */
let CHIP_CH = 0;
function chipCharWidth() {
  if (CHIP_CH) return CHIP_CH;
  const probe = document.createElement("span");
  probe.className = "vword";
  probe.style.cssText = "position:absolute;visibility:hidden;opacity:0;transform:none";
  probe.textContent = "M".repeat(10);
  document.body.appendChild(probe);
  CHIP_CH = probe.getBoundingClientRect().width / 10 || 6.3;
  probe.remove();
  return CHIP_CH;
}

/* A chip is one word of the reader's own text, and one word can be
   "responsibilities". The cap comes from the lane it has to fit in, so the two
   cannot disagree; the whole word is still in the document panel below. */
function clipTo(word, lane) {
  const max = Math.max(4, Math.floor((lane - 16) / chipCharWidth()));
  return word.length > max ? word.slice(0, max - 1) + "…" : word;
}
function fillChannels() {
  const pool = wordPool();
  const start = RUN.stages.length ? RUN.stages[0].before : 1;
  RUN.plan.forEach((p, i) => {
    const flow = $("vf-" + i);
    if (!flow) return;
    flow.querySelectorAll(".vword").forEach((w) => w.remove());
    const si = RUN.stages.findIndex((s) => s.name === p.name);
    if (si < 0) return;
    const carried = RUN.stages[si].after;
    // How many words in flight is how much prompt is still being carried.
    const n = Math.max(2, Math.round(MAX_WORDS * Math.min(1, carried / (start || 1))));
    const kept = pool.alive.length ? pool.alive : ["your", "question"];
    const frag = document.createDocumentFragment();
    // The lanes have to be far enough apart that chips cannot land on each
    // other, and there are only so many pixels: the lane is the room this
    // channel actually has, divided by the words in flight. How many words
    // there are is how much prompt is still carried, so the channel narrows as
    // the prompt does.
    const room = Math.max(280, flow.clientWidth - 30);
    const lane = Math.max(40, Math.min(92, room / n));
    const width = lane * (n - 1);
    flow.querySelector(".vband").style.width = (width + lane).toFixed(0) + "px";
    for (let k = 0; k < n; k++) {
      const el = document.createElement("span");
      el.className = "vword";
      el.textContent = clipTo(kept[(k * 7 + i * 3) % kept.length], lane);
      // Evenly across the band, not pseudo-randomly: a random offset does not
      // avoid a collision, it only makes one harder to predict.
      el.style.setProperty("--x", (k * lane - width / 2).toFixed(0) + "px");
      // Neighbouring lanes fall at different speeds, so two chips side by side
      // are never level and the whole thing reads as a stream rather than a row.
      const dur = 2.1 + ((k * 3) % 7) / 5;
      el.style.setProperty("--dur", dur.toFixed(2) + "s");
      el.style.animationDelay = ((k * 0.37) % dur).toFixed(2) + "s";
      frag.appendChild(el);
    }
    // Whatever this stage removed leaves here, in red, sideways.
    const removed = pool.cut[p.name] || [];
    removed.slice(0, 5).forEach((w, k) => {
      const el = document.createElement("span");
      el.className = "vword cutword";
      el.textContent = w;
      el.style.animationDelay = (k * 0.45).toFixed(2) + "s";
      el.style.setProperty("--dur", "2.0s");
      el.style.setProperty("--x", ((k % 3) - 1) * 22 + "px");
      el.style.setProperty("--side", (k % 2 ? -1 : 1) * (86 + k * 20) + "px");
      frag.appendChild(el);
    });
    flow.appendChild(frag);
  });
}

/* ── the transport ────────────────────────────────────────────────── */
function drawTrack() {
  const total = RUN.stages.reduce((a, s) => a + Math.max(s.ms, 0.05), 0) || 1;
  $("track").innerHTML = RUN.stages.map((s, i) => {
    const pct = (Math.max(s.ms, 0.05) / total) * 100;
    const label = pct > 7 ? esc(labelFor(s.name)) : "";
    return `<div class="seg ${s.outcome}" id="seg-${i}" style="flex:1 1 ${pct}%"`
         + ` tabindex="0" role="button"`
         + ` aria-label="${esc(labelFor(s.name))}, ${s.ms.toFixed(1)} milliseconds"`
         + ` title="${esc(labelFor(s.name))} — ${s.ms.toFixed(1)}ms">`
         + `<div class="segfill"></div><div class="seglab">${label}</div></div>`;
  }).join("");
  RUN.stages.forEach((s, i) => {
    const seg = $("seg-" + i);
    seg.onclick = () => goTo(i);
    seg.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") { goTo(i); e.preventDefault(); }
    };
  });
  $("tick-r").textContent = total.toFixed(0) + " ms of middleware";

  /* And say which stage that is. A bar with one block in it has already made
     the point to anyone who reads bars; the sentence is for everyone else, and
     it is the number this whole tier exists to justify. */
  const top = RUN.stages.reduce((a, b) => (b.ms > a.ms ? b : a),
                                RUN.stages[0] || { ms: 0, name: "" });
  const rest = RUN.stages.length - 1;
  // Only worth saying when there is time to dominate. A cache hit spends under
  // a millisecond in total, and "69% of 0 ms" is not a finding.
  const ms = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1));
  $("track-note").innerHTML = (!RUN.stages.length || total < 5 || rest < 1) ? "" :
    `<b>${esc(labelFor(top.name))}</b> is ${(100 * top.ms / total).toFixed(0)}% of it `
    + `(${ms(top.ms)} ms) — the other ${rest} stage${rest === 1 ? "" : "s"} `
    + `share ${ms(total - top.ms)} ms between them.`;
}

function labelFor(name) {
  const l = RUN.layers.find((x) => x.name === name);
  if (l) return l.label;
  const p = RUN.plan.find((x) => x.name === name);
  return p ? p.label : name;
}

function goTo(i) {
  RUN.at = Math.max(-1, Math.min(RUN.stages.length - 1, i));
  RUN.selected = null;
  paintAll();
}
function stepBy(d) { stopPlay(); goTo(RUN.at + d); }

function play() {
  if (RUN.playing) { stopPlay(); return; }
  if (RUN.at >= RUN.stages.length - 1) RUN.at = -1;
  RUN.playing = true;
  $("t-play").textContent = "❚❚ Pause";
  const beat = parseInt($("t-speed").value, 10);
  RUN.timer = setInterval(() => {
    if (RUN.at >= RUN.stages.length - 1) { stopPlay(); return; }
    goTo(RUN.at + 1);
  }, beat);
}
function stopPlay() {
  RUN.playing = false;
  clearInterval(RUN.timer);
  $("t-play").textContent = "▶ Play";
}
$("t-play").onclick = play;
$("t-next").onclick = () => stepBy(1);
$("t-prev").onclick = () => stepBy(-1);
$("t-first").onclick = () => { stopPlay(); goTo(-1); };
$("t-last").onclick = () => { stopPlay(); goTo(RUN.stages.length - 1); };
$("t-speed").onchange = () => { if (RUN.playing) { stopPlay(); play(); } };
addEventListener("keydown", (e) => {
  if ($("view-pipe").hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowRight") { stepBy(1); e.preventDefault(); }
  if (e.key === "ArrowLeft") { stepBy(-1); e.preventDefault(); }
  if (e.key === " ") { play(); e.preventDefault(); }
});

/* ── painting one moment of the run ───────────────────────────────── */
function paintAll() {
  const at = RUN.at;
  // transport
  RUN.stages.forEach((s, i) => {
    const seg = $("seg-" + i);
    if (!seg) return;
    seg.classList.toggle("at", i === at);
    seg.classList.toggle("past", i <= at);
    seg.classList.toggle("future", i > at);
  });
  // stack
  for (const p of RUN.plan) {
    const node = $("vs-" + p.name);
    if (!node) continue;
    const i = RUN.stages.findIndex((s) => s.name === p.name);
    const ran = i >= 0 && i <= at;
    const s = i >= 0 ? RUN.stages[i] : null;
    node.className = "vstage"
      + (ran ? " past " + s.outcome : "")
      + (i === at ? " at" : "")
      + (RUN.selected === p.name ? " sel" : "");
    $("vk-" + p.name).textContent = ran ? `${fmt(s.after)} tokens carried on` : "—";
    $("vt-" + p.name).textContent = ran ? s.ms.toFixed(1) + " ms" : "";
    const d = ran ? s.after - s.before : 0;
    $("vd-" + p.name).textContent = ran && d < 0 ? `−${fmt(-d)}` : "";
    const layer = RUN.layers.find((l) => l.name === p.name);
    $("vw-" + p.name).textContent = ran && layer ? layer.why : "";
    paintGate(p.name, ran ? s : null);
    // Only the channels the request has actually reached are in motion.
    const flow = $("vf-" + RUN.plan.indexOf(p));
    if (flow) flow.classList.toggle("running", ran);
  }
  // the now-line
  const s = at >= 0 ? RUN.stages[at] : null;
  $("t-now").innerHTML = s
    ? `<b>${esc(labelFor(s.name))}</b> <span class="sub">· ${s.ms.toFixed(1)} ms · `
      + `${esc(OUTCOME_WORD[s.outcome] || s.outcome)} · ${fmt(s.before)} → ${fmt(s.after)} tokens`
      + `</span>`
    : `<span class="sub">${RUN.stages.length ? "before anything ran — press play, or use ← →"
        : "run something, then scrub through it stage by stage"}</span>`;
  const gs = $("gate-summary");
  if (gs) gs.textContent = gateSummary();
  paintDoc();
  paintInspector();
  paintLayers();
}

/* ── the document, losing sentences as the stages run ─────────────── */
let docBuilt = false;
function buildDoc() {
  if (!RUN.units.length) {
    $("docwrap").innerHTML = '<div class="dimtext" style="font-size:13px">'
      + 'This request carried no attached document — nothing for the context tier to trim. '
      + 'Attach the handbook and run again to watch it shrink.</div>';
    docBuilt = false;
    return;
  }
  const groups = [];
  for (const u of RUN.units) {
    const src = u.source || "your text";
    const last = groups[groups.length - 1];
    if (last && last.src === src) last.items.push(u);
    else groups.push({ src, items: [u] });
  }
  $("docwrap").innerHTML = groups.map((g) =>
    `<div class="docgroup"><div class="gt">${esc(g.src)}</div>`
    + g.items.map((u) =>
      `<span class="sent" id="u-${u.order}" data-score="${u.score != null
        ? Number(u.score).toFixed(2) : ""}"`
      + ` data-tag="${esc(u.tag || "")}" data-detail="${esc(u.detail || "")}"`
      + ` data-at="${esc(u.removed_at || "")}">${esc(u.text)}</span>`).join("")
    + `</div>`).join("");
  docBuilt = true;
}

function paintDoc() {
  if (!docBuilt) return;
  const at = RUN.at;
  const doneNames = new Set(RUN.stages.slice(0, at + 1).map((s) => s.name));
  let alive = 0, cut = 0, justCut = 0;
  const cutBy = at >= 0 ? RUN.stages[at].name : null;
  for (const u of RUN.units) {
    const el = $("u-" + u.order);
    if (!el) continue;
    const removed = u.removed_at && doneNames.has(u.removed_at);
    const nowCut = removed && u.removed_at === cutBy;
    el.className = "sent " + (removed ? "cut" : (at >= RUN.stages.length - 1 ? "kept" : "alive"))
      + (nowCut ? " flash" : "");
    if (removed) { cut++; if (nowCut) justCut++; } else alive++;
  }
  $("doc-hint").textContent = at < 0
    ? `${RUN.units.length} sentences, all still in play`
    : `${alive} kept · ${cut} removed` + (justCut ? ` · ${justCut} by this layer` : "");
}

/* ── the inspector: what THIS layer looked at ─────────────────────── */
function paintInspector() {
  const name = RUN.selected || (RUN.at >= 0 ? RUN.stages[RUN.at].name : null);
  if (!name) {
    $("inspector").innerHTML = '<div class="dimtext" style="font-size:13px">Pick a layer on '
      + 'the stack, or press play and watch each one explain itself.</div>';
    return;
  }
  const s = RUN.stages.find((x) => x.name === name);
  const l = RUN.layers.find((x) => x.name === name);
  const p = RUN.plan.find((x) => x.name === name);
  const ev = (s && s.evidence) || {};
  const ran = !!s && RUN.stages.indexOf(s) <= RUN.at;

  let html = `<div class="insp-head"><span class="t">${esc(l ? l.label : p.label)}</span>`
    + `<span class="m">${esc(p ? p.module : "")} · ${esc(name)}</span>`;
  if (s) html += `<span class="verdict ${s.outcome}">${esc(OUTCOME_WORD[s.outcome]
    || s.outcome)}</span>`;
  html += `</div>`;
  html += `<div class="why">${esc(l && ran ? l.why : (p ? p.does : ""))}</div>`;

  if (!ran) {
    html += '<div class="dimtext" style="font-size:12.5px">Has not run yet at this point '
          + 'in the timeline. Scrub forward to see what it decided.</div>';
    $("inspector").innerHTML = html;
    return;
  }

  html += detailFor(name, ev, s, l);
  $("inspector").innerHTML = html;
  // meters animate only once they are in the DOM
  requestAnimationFrame(() => {
    document.querySelectorAll("#inspector .meter > i[data-w]").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

function kv(pairs) {
  const rows = pairs.filter((p) => p[1] !== undefined && p[1] !== null && p[1] !== "")
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  return rows ? `<dl class="kv">${rows}</dl>` : "";
}

function meter(value, max, markAt, leftLab, rightLab) {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  const m = markAt == null ? "" :
    `<b style="left:${Math.min(100, (markAt / max) * 100)}%"></b>`;
  return `<div class="meter"><i data-w="${w.toFixed(1)}"></i>${m}</div>`
       + `<div class="meterlab"><span>${esc(leftLab)}</span><span>${esc(rightLab)}</span></div>`;
}

/* Each layer gets the view that suits what it actually reasons about. A generic
   key/value dump would be the same page for all eleven, which is the thing this
   view exists to avoid. */
function detailFor(name, ev, s, l) {
  const removed = RUN.units.filter((u) => u.removed_at === name);
  switch (name) {
    case "m1_context": {
      const kept = RUN.units.filter((u) => !u.removed_at);
      const shown = removed.slice().sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, 6);
      const top = kept.slice().sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, 6);
      const bars = (list, below) => list.map((u) =>
        `<div class="bar${below ? " below" : ""}"><span>${(u.score || 0).toFixed(2)}</span>`
        + `<span class="t"><i style="width:${Math.round((u.score || 0) * 100)}%"></i></span>`
        + `<span title="${esc(u.text)}">${esc(u.tag || "")}</span></div>`).join("");
      return kv([
        ["sentences in", ev.sentences], ["kept", ev.sentences_kept],
        ["context tokens", `${ev.context_tokens_before} → ${ev.context_tokens_after}`],
        ["documents dropped", ev.documents_dropped],
      ])
      + `<div class="note" style="margin-top:12px">Highest scoring sentences it kept</div>`
      + `<div class="bars">${bars(top, false)}</div>`
      + (shown.length
        ? `<div class="note">Highest scoring it still removed — the budget and the floor</div>`
          + `<div class="bars">${bars(shown, true)}</div>` : "");
    }
    case "m2_cache": {
      const top = (RUN.summary && RUN.summary.cache.top_k) || [];
      return kv([
        ["zone", ev.zone], ["scoped to this conversation", ev.scoped ? "yes" : "no"],
        ["nearest stored question", top.length ? top[0][1] : "nothing stored yet"],
      ]) + (top.length
        ? meter(top[0][1], 1, 0.9, "similarity " + top[0][1], "reuse needs ≈0.90 + verify")
        : `<div class="note">The cache is empty on a fresh server, so this is a miss by `
          + `construction. Ask the same thing twice to see it hit.</div>`);
    }
    case "m6a_deterministic":
      return kv([["handler", ev.handler || "none matched"]])
        + `<div class="note">When this fires the model is never called at all: the answer `
        + `costs zero model tokens.</div>`;
    case "m6b_router":
      return kv([
        ["complexity", ev.complexity], ["threshold", ev.threshold],
        ["escalation enabled", ev.escalation_enabled ? "yes" : "no"],
        ["words in the question", ev.n_words], ["history turns", ev.n_history_turns],
      ]) + (ev.complexity != null
        ? meter(ev.complexity, 0.5, ev.threshold, "complexity " + ev.complexity,
                "escalate above " + ev.threshold) : "");
    case "m5_budgeter":
      return kv([["answer class", ev.response_class], ["tokens allowed", ev.budget]])
        + `<div class="note">The class decides the cap. A factual answer does not need `
        + `the room a code answer does.</div>`;
    case "m4_assembler":
      return kv([["invariant tokens pinned", ev.invariant_tokens],
                 ["prompt tokens", ev.total_tokens]])
        + `<div class="note">Keeping the unchanging part first is what lets the runtime `
        + `reuse its key-value cache across turns.</div>`;
    case "m3_history":
      return kv([["turns kept", ev.kept], ["turns dropped", ev.dropped]]);
    case "m3_arrange":
      return kv([["moved turn", ev.moved_from != null ? ev.moved_from + 1 : null],
                 ["its relevance", ev.relevance]]);
    case "m1_tier1": case "m1_tier2": case "m1_tier3": {
      // A tier can end three ways, and they mean opposite things. Reverted is
      // the gate refusing a saving; saying "nothing here" under it, as this
      // panel first did, contradicts the line directly above it.
      if (s && s.outcome === "reverted") {
        const lost = (s.gate_events || []).join(", ") || "meaning";
        return kv([["edit proposed", "yes"], ["committed", "no — the gate refused it"],
                   ["invariants it would have lost", lost]])
          + `<div class="note">The tier found a saving and the fidelity gate threw it `
          + `away, because the cut would have taken a <b>${esc(lost)}</b> with it. This is `
          + `the system declining tokens to stay correct — the one trade it is never `
          + `allowed to make silently.</div>`;
      }
      const base = kv([
        ["tier", ev.tier], ["candidate edits", ev.candidates],
        ["rejected for negative yield", ev.negative_yield_rejected],
        ["sentences removed", removed.length || null],
      ]);
      const note = ev.negative_yield_rejected
        ? `<div class="note">It found edits and threw them away: in this tokenizer the `
          + `rewrite would have cost more tokens than the original. Declining is the `
          + `correct answer, and the ledger records it as one.</div>`
        : `<div class="note">Nothing here for this tier. On a short, already-clean prompt `
          + `that is the expected result, not a failure.</div>`;
      return base + note;
    }
    default:
      return kv(Object.entries(ev).slice(0, 8));
  }
}

/* ── the prompt bar, the terminal's ###......... line ─────────────── */
function paintTokenBar(t) {
  $("tokenbar").hidden = false;
  $("tb-sent").style.width = (t.ratio * 100).toFixed(1) + "%";
  $("tb-num").innerHTML = `<b>${fmt(t.final)}</b> sent · <i>${fmt(t.removed)} removed `
    + `(${Math.round((1 - t.ratio) * 100)}%)</i> of ${fmt(t.original)} written`;
}

/* Memory, not time. Derived from the model's own key-value geometry, so the
   derivation travels with the number: without it this is a plausible-looking
   megabyte figure and there is no way for a reader to check it. Absent
   entirely when the runtime will not report its geometry. */
function paintKv(kv, tokens) {
  const line = $("kvline");
  if (!kv || !tokens.removed) { line.hidden = true; return; }
  line.hidden = false;
  const mb = kv.bytes_saved / 1048576;
  $("kv-n").textContent = (mb >= 1 ? mb.toFixed(1) + " MB" : Math.round(kv.bytes_saved / 1024)
    + " KB") + " of key-value cache never allocated";
  $("kv-how").textContent = `${fmt(tokens.removed)} tokens never read × `
    + `${fmt(kv.per_token)} bytes each — ${kv.how}`;
}

/* ── what the AI actually received ───────────────────────────────── */
function paintReceived(rows) {
  if (!rows || !rows.length) return;
  $("received").classList.toggle("unsent", rows.length === 1 && rows[0].kind === "unsent");
  $("received").innerHTML = rows.map((r) => {
    const faded = r.kind.endsWith("dropped") || r.kind === "system";
    let body;
    if (r.runs && r.runs.length) {
      body = r.runs.map((run) => run.kept
        ? esc(run.text) + " "
        : `<span class="tagr">${esc(run.label)}</span> `
          + `<span class="gone">${esc(run.text)}</span> `
          + (run.folded ? `<span class="fold">[… ${run.folded} more]</span> ` : "")
      ).join("");
    } else if (r.kind.endsWith("dropped")) {
      body = `<span class="whole">${esc(r.note)}</span>`;
    } else {
      body = `<span class="${r.kind === "system" ? "plain" : ""}">${esc(r.note)}</span>`;
    }
    return `<div class="rl${faded ? " faded" : ""}">${esc(r.label)}</div>`
         + `<div class="rc">${body}</div>`;
  }).join("");
}

/* ── measured ────────────────────────────────────────────────────── */
function paintMeasured(lines) {
  if (!lines || !lines.length) return;
  $("measured").innerHTML = lines.map((l) => {
    const warn = /simulated|estimate|reused that work|not measured/i.test(l);
    return `<li class="${warn ? "warn" : ""}">${esc(l)}</li>`;
  }).join("");
}

/* ── the terminal's table ─────────────────────────────────────────── */
function paintLayers() {
  if (!RUN.layers.length) return;
  const atName = RUN.at >= 0 ? RUN.stages[RUN.at].name : null;
  $("layers-body").innerHTML = RUN.layers.map((l) => {
    const i = RUN.stages.findIndex((s) => s.name === l.name);
    const ran = i >= 0 && i <= RUN.at;
    const [glyph, cls] = OUTCOME_MARK[l.outcome] || ["·", "no"];
    const tok = l.delta == null ? "" : (l.delta < 0 ? fmt(l.delta) : "–");
    const dim = !ran || l.outcome === "skipped" || (l.outcome === "noop" && !l.delta);
    return `<tr class="${l.name === atName ? "at " : ""}${l.outcome}${dim ? " dim" : ""}">`
      + `<td><span class="mark ${cls}">${ran ? glyph : "·"}</span></td>`
      + `<td><b>${esc(l.label)}</b><br><span class="dimtext">${esc(l.job)}</span></td>`
      + `<td>${esc(l.module)}</td>`
      + `<td class="num">${ran && l.ms != null ? l.ms.toFixed(1) + "ms" : ""}</td>`
      + `<td class="num d">${ran ? tok : ""}</td>`
      + `<td class="num">${ran && l.removed ? "−" + l.removed : ""}</td>`
      + `<td>${ran ? esc(l.why) : ""}</td></tr>`;
  }).join("");
}

/* The table as plain text, for pasting into a report. The numbers in a write-up
   should come from the run, not from someone retyping them off a screenshot. */
$("copy-run").onclick = async () => {
  if (!RUN.layers.length) return;
  const d = RUN.summary;
  const w = [22, 6, 9, 8];
  const pad = (s, n) => String(s).padEnd(n).slice(0, n);
  const lines = [
    `Parsimony - ${$("pq").value.trim()}`,
    `${d.tokens.original} tokens written -> ${d.tokens.final} sent `
      + `(${d.tokens.removed} removed, ${Math.round((1 - d.tokens.ratio) * 100)}%)`,
    `route ${d.route.tier}  cache ${d.cache.hit ? "HIT" : "miss"}  `
      + `gate ${d.gate.fired ? "blocked an edit" : "passed"}`,
    "",
    pad("Layer", w[0]) + pad("Mod", w[1]) + pad("Time", w[2]) + pad("Tokens", w[3])
      + "What happened",
  ];
  for (const l of RUN.layers) {
    lines.push(pad(l.label, w[0]) + pad(l.module, w[1])
      + pad(l.ms == null ? "" : l.ms.toFixed(1) + "ms", w[2])
      + pad(l.delta ? String(l.delta) : "-", w[3]) + l.why);
  }
  lines.push("", ...d.measured);
  const text = lines.join("\n");
  try {
    await navigator.clipboard.writeText(text);
    $("copy-run").textContent = "copied";
  } catch (e) {
    // Clipboard access is refused in some contexts; show it instead of failing
    // silently, so the text is still available to select.
    $("pipe-prompt").textContent = text;
    $("copy-run").textContent = "shown below";
  }
  setTimeout(() => { $("copy-run").textContent = "copy as text"; }, 1800);
};

/* ── running one request ──────────────────────────────────────────── */
fetch("/api/plan").then((r) => r.json()).then((p) => {
  RUN.plan = p.stages; drawStack(); paintAll();
}).catch(() => {});

/* Attaching is a fetch, and a fetch takes a moment. Pressing Run inside that
   moment used to read an empty question box and complain about it, and the
   document would land a breath later -- leaving a complaint about an empty
   field sitting under a filled one. The button says what it is doing and
   refuses a second press until it has finished. */
$("pipe-sample").onclick = async () => {
  const button = $("pipe-sample");
  if (button.disabled) return;
  button.disabled = true;
  const label = button.textContent;
  button.innerHTML = '<span class="spinner"></span>attaching';
  try {
    const s = await (await fetch("/api/sample")).json();
    $("pipe-text").value = s.text;
    sizeContext();
    if (!$("pq").value.trim()) $("pq").value = s.question;
    remember();
    clearPipeError();
  } catch (e) { $("pipe-err").textContent = "no sample document on this machine"; }
  button.disabled = false;
  button.textContent = label;
};

/* A complaint that outlives the thing it complained about is worse than none:
   the reader sees "ask something first" beside a question and concludes the
   page is broken. It goes as soon as the field it is about has something in it. */
function clearPipeError() {
  if ($("pq").value.trim() && $("pipe-err").textContent === "ask something first") {
    $("pipe-err").textContent = "";
  }
}
$("pq").addEventListener("input", clearPipeError);
/* Four scenarios worth showing, because the interesting behaviour is not all in
   one request: a compression, a question the model never sees, an edit the gate
   refuses, and an answer served from the cache. */
document.querySelectorAll(".chip-b").forEach((b) => {
  b.onclick = async () => {
    document.querySelectorAll(".chip-b").forEach((o) =>
      o.setAttribute("aria-pressed", String(o === b)));
    $("pq").value = b.dataset.q;
    $("chip-note").textContent = b.dataset.note.replace(/\s+/g, " ").trim();
    if (b.dataset.doc) {
      if (!$("pipe-text").value.trim()) await $("pipe-sample").onclick();
    } else {
      $("pipe-text").value = "";
      sizeContext();
    }
    remember();
    $("run-pipe").click();
  };
});

/* Keep the question and the attached text across a reload. Losing a 4 KB
   handbook to an accidental refresh mid-demo is a small thing that feels like a
   broken tool; browser storage is per-viewer and nothing here leaves the
   machine, and every access is guarded because a private window throws. */
const REMEMBER = "parsimony.pipe.v1";
function remember() {
  try {
    localStorage.setItem(REMEMBER, JSON.stringify({
      q: $("pq").value, text: $("pipe-text").value.slice(0, 400000),
    }));
  } catch (e) { /* private window, or storage disabled: carry on without it */ }
}
function recall() {
  try {
    const saved = JSON.parse(localStorage.getItem(REMEMBER) || "{}");
    // "Untouched" is the empty box OR the shipped default still sitting in it;
    // a question the reader typed before a reload outranks both.
    const untouched = !$("pq").value || $("pq").value === $("pq").dataset.default;
    if (saved.q && untouched) $("pq").value = saved.q;
    if (saved.text && !$("pipe-text").value) $("pipe-text").value = saved.text;
  } catch (e) { /* nothing remembered is a perfectly good state */ }
  sizeContext();
}
function sizeContext() {
  const n = $("pipe-text").value.length;
  $("pipe-ctx-size").textContent = n ? Math.round(n / 1024) + " KB attached" : "nothing yet";
}
$("pipe-text").addEventListener("input", () => { sizeContext(); remember(); });
$("pq").addEventListener("input", remember);
recall();

$("run-pipe").onclick = () => {
  const question = $("pq").value.trim();
  $("pipe-err").textContent = "";
  if (!question) { $("pipe-err").textContent = "ask something first"; return; }
  stopPlay();
  Object.assign(RUN, { stages: [], layers: [], units: [], summary: null, at: -1,
                       selected: null, live: true });
  docBuilt = false;
  $("docwrap").innerHTML = '<div class="dimtext" style="font-size:13px">running…</div>';
  $("layers-body").innerHTML = '<tr class="dim"><td></td><td colspan="6">running…</td></tr>';
  $("pipe-answer").innerHTML = '<span class="cursor"></span>';
  $("pipe-prompt").textContent = "—";
  $("tokenbar").hidden = true;
  $("kvline").hidden = true;
  $("received").innerHTML = '<div class="dimtext" style="font-size:13px">assembling…</div>';
  $("measured").innerHTML = '<li class="dimtext">measuring…</li>';
  $("prompt-tok").textContent = ""; $("answer-tok").textContent = "";
  $("pipe-timing").textContent = "";
  drawStack(); paintAll();
  $("run-pipe").disabled = true;
  $("run-pipe").innerHTML = '<span class="spinner"></span>running';

  const url = "/api/pipeline?question=" + encodeURIComponent(question)
            + "&text=" + encodeURIComponent($("pipe-text").value);
  const es = new EventSource(url);
  let answer = "";

  es.addEventListener("plan", (e) => {
    RUN.plan = JSON.parse(e.data).stages; drawStack(); paintAll();
  });
  es.addEventListener("begin", (e) => {
    const d = JSON.parse(e.data);
    countTo($("m-before"), d.tokens);
  });
  es.addEventListener("stage", (e) => {
    RUN.stages.push(JSON.parse(e.data));
    drawTrack();
  });
  es.addEventListener("prompt", (e) => {
    const d = JSON.parse(e.data);
    $("pipe-prompt").textContent = d.text;
    $("prompt-tok").textContent = fmt(d.tokens) + " tokens";
  });
  es.addEventListener("token", (e) => {
    answer += JSON.parse(e.data).text;
    $("pipe-answer").innerHTML = esc(answer) + '<span class="cursor"></span>';
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    RUN.summary = d; RUN.layers = d.layers; RUN.units = d.units; RUN.live = false;
    $("pipe-answer").textContent = d.answer || answer;
    $("answer-tok").textContent = `${d.tokens.out} tokens`;
    countTo($("m-before"), d.tokens.original);
    countTo($("m-after"), d.tokens.final);
    countTo($("m-removed"), d.tokens.removed);
    countTo($("m-saved"), d.timing.saved_s, "s");
    $("m-saved-k").textContent = d.timing.timed_here
      ? "prefill not spent (measured here)" : "prefill not spent (estimated rate)";
    const C = 2 * Math.PI * 32;
    $("ring-fg").style.strokeDashoffset = String(C * (1 - d.tokens.ratio));
    $("m-ratio").textContent = Math.round(d.tokens.ratio * 100) + "%";
    const t = d.timing;
    // Four cases, and they mean different things. Saying "simulated" for a
    // request the cache answered was simply wrong: there was no prefill at all.
    $("pipe-timing").textContent = d.served_without_model
      ? `Middleware took ${t.middleware_ms.toFixed(0)} ms and the model was never called. `
        + `Sending the question would have meant reading ${fmt(d.tokens.original)} tokens — `
        + `about ${t.saved_s.toFixed(1)} s at ${t.ms_per_token.toFixed(1)} ms/token — and then `
        + `waiting for an answer.`
      : `Middleware took ${t.middleware_ms.toFixed(0)} ms and saved ${t.saved_s.toFixed(1)} s `
        + `of prefill at ${t.ms_per_token.toFixed(1)} ms/token`
        + (t.timed_here ? ", measured on this machine."
          : t.reused ? " — but the runtime reused an earlier prompt, so that rate is not cold."
          : " — the model is simulated, so that rate is the project's recorded figure.");
    paintTokenBar(d.tokens); paintKv(d.kv, d.tokens);
    paintReceived(d.received); paintMeasured(d.measured);
    buildDoc(); fillChannels(); drawTrack(); goTo(-1);
    refreshSession();
    stop();
    // Walk it automatically the first time: the point is to be watched.
    setTimeout(play, 350);
  });
  es.addEventListener("failed", (e) => {
    $("pipe-err").textContent = JSON.parse(e.data).error; stop();
  });
  es.onerror = () => stop();

  function stop() {
    es.close(); RUN.live = false;
    $("run-pipe").disabled = false; $("run-pipe").textContent = "Run";
  }
};

/* ══════════════════════════ heatmap ═══════════════════════════════ */
const SAMPLE_Q = "What is the annual travel budget for the Tallinn office?";

$("file").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => { $("text").value = reader.result; };
  reader.readAsText(f);
};

$("sample").onclick = async () => {
  $("sample").disabled = true;
  try {
    const s = await (await fetch("/api/sample")).json();
    $("text").value = s.text;
    if (!$("q").value.trim()) $("q").value = s.question || SAMPLE_Q;
  } catch (e) { $("map-err").textContent = "no sample document on this machine"; }
  $("sample").disabled = false;
};

const shade = (score) => `rgba(78,201,168,${(0.06 + 0.42 *
  Math.max(0, Math.min(1, score))).toFixed(3)})`;

/* Sentences sorted by score, with the floor drawn across them.

   If selection were nothing but a threshold, kept would be a clean prefix and
   this chart would be boring. It is not: anchors survive below the floor, and
   dependency closure drags in sentences that score near zero because the
   sentence above them needs them to make sense. Those exceptions are the tier,
   and colouring them differently is the whole point of the picture. */
function drawProfile(data) {
  const svg = $("profile");
  const units = data.units.slice().sort((a, b) => b.score - a.score);
  if (!units.length) { svg.innerHTML = ""; return; }

  // 1:1 with the panel it sits in, so no axis is stretched against the other;
  // wider only when there are more sentences than the panel has room for, and
  // then the wrapper scrolls.
  const avail = Math.max(340, (svg.parentElement && svg.parentElement.clientWidth) || 900);
  const W = Math.max(avail, units.length * 9), H = 190;
  const pad = { l: 34, r: 8, t: 12, b: 26 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const bw = Math.max(1.5, plotW / units.length - 1);
  const floor = typeof data.floor === "number" ? data.floor : null;
  const Y = (v) => pad.t + plotH * (1 - Math.max(0, Math.min(1, v)));

  let exceptions = 0;
  const bars = units.map((u, i) => {
    const x = pad.l + i * (plotW / units.length);
    const y = Y(u.score);
    const below = floor !== null && u.score < floor;
    let fill = u.kept ? "var(--keep)" : "#3a4150";
    if (u.kept && below) { fill = "var(--warn)"; exceptions++; }
    if (!u.kept && !below) { fill = "var(--drop)"; exceptions++; }
    const why = `${u.kept ? "KEPT" : "DROPPED"} · ${u.tag} · score ${u.score.toFixed(3)}`
              + `\n${u.detail}\n\n${u.text.slice(0, 160)}`;
    return `<rect class="bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" `
         + `width="${bw.toFixed(1)}" height="${(pad.t + plotH - y).toFixed(1)}" `
         + `fill="${fill}"><title>${esc(why)}</title></rect>`;
  }).join("");

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line class="axis" x1="${pad.l}" y1="${Y(v)}" x2="${W - pad.r}" y2="${Y(v)}"`
    + ` opacity="${v === 0 ? 1 : 0.35}"/>`
    + `<text x="${pad.l - 6}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`).join("");

  /* The cliff the adaptive rule read the floor from. Drawn as the SPAN
     between two adjacent ranks rather than a line, because the claim is about
     a fall between them and a line would read as one more threshold. */
  let cliff = "";
  const drawn = data.adaptive && data.cliff_rank >= 0 && data.cliff_rank + 1 < units.length;
  $("legend-cliff").hidden = !drawn;
  if (drawn) {
    const step = plotW / units.length;
    const x0 = pad.l + data.cliff_rank * step;
    const yHi = Y(units[data.cliff_rank].score);
    const yLo = Y(units[data.cliff_rank + 1].score);
    // Centre of the last kept bar to centre of the first dropped one: the band
    // is the fall between two ranks, so it spans both of them.
    cliff = `<rect class="cliff" x="${(x0 + bw / 2).toFixed(1)}" y="${yHi.toFixed(1)}" `
          + `width="${step.toFixed(1)}" `
          + `height="${(yLo - yHi).toFixed(1)}"/>`
          + `<line class="cliffedge" x1="${pad.l}" y1="${yHi.toFixed(1)}" `
          + `x2="${(x0 + step * 2).toFixed(1)}" y2="${yHi.toFixed(1)}"/>`
          + `<text class="clab" x="${(x0 + step * 2 + 5).toFixed(1)}" `
          + `y="${((yHi + yLo) / 2 + 3).toFixed(1)}">`
          + `↓ ${data.cliff_fall.toFixed(1)}×</text>`;
  }

  const floorMark = floor === null ? "" :
    `<line class="floorline" x1="${pad.l}" y1="${Y(floor)}" x2="${W - pad.r}" y2="${Y(floor)}"/>`
    + `<text class="lab" x="${W - pad.r}" y="${Y(floor) - 5}" text-anchor="end">`
    + `${data.adaptive ? "floor read at" : "relevance floor"} ${floor}</text>`;

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.innerHTML = ticks + cliff + bars + floorMark
    + `<text x="${pad.l}" y="${H - 8}">highest scoring</text>`
    + `<text x="${W - pad.r}" y="${H - 8}" text-anchor="end">lowest</text>`;

  const kept = units.filter((u) => u.kept).length;
  $("profile-note").innerHTML = exceptions
    ? `${kept} of ${units.length} sentences kept. <b>${exceptions}</b> of them `
      + `${exceptions === 1 ? "is" : "are"} not explained by the score alone — an anchor the `
      + `question named, or a sentence dependency closure pulled in so a pronoun still refers `
      + `to something. Hover any bar for its reason. `
      + `Selection stopped because of <b>${esc(data.stopped_by)}</b>.`
    : `${kept} of ${units.length} sentences kept, and the cut is exactly the floor — no anchor `
      + `or closure exception on this question. Selection stopped because of `
      + `<b>${esc(data.stopped_by)}</b>.`;
}

function renderMap(data) {
  $("map-stats").hidden = $("map-doc-panel").hidden = false;
  $("map-profile-panel").hidden = false;
  const kept = data.units.filter((u) => u.kept).length;
  $("map-numbers").innerHTML = [
    ["tokens written", fmt(data.tokens_before)], ["tokens sent", fmt(data.tokens_after)],
    ["removed", data.removed_pct.toFixed(0) + "%"],
    ["sentences kept", `${kept} / ${data.units.length}`],
    ["encoder time", data.encoder_ms.toFixed(0) + "ms"],
    ["deciding", data.decide_ms.toFixed(0) + "ms"],
  ].map(([k, v]) => `<div class="stat"><div class="n">${v}</div><div class="k">${k}</div></div>`)
    .join("");
  const bits = [`stopped by <b>${esc(data.stopped_by)}</b>`,
                `term coverage ${(data.coverage * 100).toFixed(0)}%`];
  if (data.off_topic) bits.push("<b>off topic</b> — the floor keeps one sentence rather "
    + "than inventing relevance");
  if (data.anchors.length) bits.push("anchors held: " + esc(data.anchors.join(", ")));
  $("map-note").innerHTML = bits.join(" · ");

  let html = "", src = null;
  for (const u of data.units) {
    if (u.source !== src) { src = u.source; html += `<div class="src">${esc(src)}</div>`; }
    html += `<span class="hsent ${u.kept ? "keep" : "drop"}"`
         + ` style="background:${u.kept ? shade(u.score) : "transparent"}"`
         + ` data-tag="${esc(u.tag)}" data-detail="${esc(u.detail)}"`
         + ` data-score="${u.score.toFixed(3)}" data-tokens="${u.tokens}">`
         + `${esc(u.text)}</span> `;
  }
  $("map-doc").innerHTML = html;
  drawProfile(data);
  if (typeof data.floor === "number") {
    $("floor").value = data.floor;
    $("floor-v").textContent = Number(data.floor).toFixed(2);
    if (typeof data.ratio === "number") {
      $("ratio").value = data.ratio;
      $("ratio-v").textContent = Math.round(data.ratio * 100) + "%";
    }
    floorLabel(data);
  }
  refreshSession();
}

const tip = $("tip");
function tipFor(el, kept) {
  const detail = el.dataset.detail || "";
  // The second line is the selector's own account of the decision, and for a
  // MATCH that account is the relevance -- so repeating it underneath, less
  // roundly, said the same thing twice. Where the account says something else
  // (an anchor, a closure, a budget) the number still earns its place.
  const parts = [];
  if (!/relevance/i.test(detail)) parts.push("relevance " + el.dataset.score);
  if (el.dataset.tokens) parts.push(el.dataset.tokens + " tokens");
  tip.innerHTML = `<b>[${kept ? "KEEP" : "DROP"}: ${esc(el.dataset.tag)}]</b><br>`
    + esc(detail)
    + (parts.length ? `<br><span class="dimtext">${esc(parts.join(" · "))}</span>` : "");
  tip.style.display = "block";
}
function wireTips(host, sel, keptClass) {
  host.addEventListener("mouseover", (e) => {
    const el = e.target.closest(sel);
    if (el && el.dataset.tag) tipFor(el, el.classList.contains(keptClass));
  });
  host.addEventListener("mouseout", (e) => {
    if (!e.relatedTarget || !e.relatedTarget.closest(sel)) tip.style.display = "none";
  });
}
wireTips($("map-doc"), ".hsent", "keep");
wireTips($("docwrap"), ".sent", "kept");
addEventListener("mousemove", (e) => {
  if (tip.style.display !== "block") return;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});

/* Dragging the floor re-runs the selection at that threshold and nothing else.
   The request is debounced because the encoder is the expensive part and a
   slider emits an event per pixel; the last position always wins. */
let floorTimer = null;
let lastMap = null;
/* The last answer from each mode, so switching can state the difference. Keyed
   by question and budget: a delta against a different question would be a
   comparison of two unrelated things wearing the same label. */
const lastRun = new Map();

const runKey = (d) => `${d.adaptive ? "read" : "fixed"}|${d.encoder}`;

/* The difference between this run and the last one that differed in exactly
   one setting, said in sentences. Both switches make a comparative claim --
   the rule against the constant, one encoder against the other -- and a reader
   holding two numbers in their head to see it is a reader who will not. */
function deltaAgainst(data, key, noun) {
  const other = lastRun.get(key);
  if (!other || other.question !== data.question
      || Math.abs(other.ratio - data.ratio) > 1e-9) return "";
  const ds = data.units.filter((u) => u.kept).length
           - other.units.filter((u) => u.kept).length;
  const dt = data.tokens_after - other.tokens_after;
  if (!ds && !dt) return ` · same selection as the ${noun}`;
  const word = (n, one, many) => `${Math.abs(n)} ${Math.abs(n) === 1 ? one : many}`;
  return ` · <b>${word(ds, "sentence", "sentences")} ${ds < 0 ? "fewer" : "more"}, `
       + `${word(dt, "token", "tokens")} ${dt < 0 ? "fewer" : "more"}</b> than the ${noun}`;
}

function modeDelta(data) {
  return deltaAgainst(data, `${data.adaptive ? "fixed" : "read"}|${data.encoder}`,
                      data.adaptive ? "constant" : "reading");
}

function encoderDelta(data) {
  const other = data.encoder === data.shipped_encoder
    ? data.other_encoder : data.shipped_encoder;
  if (!other) return "";
  return deltaAgainst(data, `${data.adaptive ? "read" : "fixed"}|${other}`, "other encoder");
}

function floorLabel(data) {
  const kept = data.units.filter((u) => u.kept).length;
  const shipped = data.shipped_floor;
  const at = !data.adaptive && Math.abs(data.floor - shipped) < 1e-9;
  const atRatio = Math.abs(data.ratio - data.shipped_ratio) < 1e-9;
  $("floor-read").innerHTML =
    `<b>${kept}</b> of ${data.units.length} sentences kept · `
    + `<b>${fmt(data.tokens_after)}</b> tokens sent · `
    + `<i>${data.removed_pct.toFixed(0)}% removed</i>`
    + (at && atRatio ? " · both dials at the shipped settings"
       : ` · shipped: floor ${shipped}, budget ${Math.round(data.shipped_ratio * 100)}%`
         + " (unchanged; this page only)");
  // `stopped_by` is the selector's own account of which constraint ended it,
  // not something inferred from the numbers on screen.
  const why = {
    "relevance floor": "the <b>floor</b> stopped it — the next sentence scored too low to keep, "
      + "so raising the budget would change nothing",
    budget: "the <b>budget</b> stopped it — sentences above the floor were left behind for want "
      + "of room, so a bigger budget would take more",
    exhausted: "neither dial stopped it: everything above the floor already fit",
  }[data.stopped_by] || esc(data.stopped_by);
  $("binding").innerHTML = "Selection ended because " + why + ".";

  // In adaptive mode the slider reports rather than asks, so it is driven from
  // the answer -- and the reading that produced it is stated in full. A page
  // that showed the computed number without the reading would be asking to be
  // trusted about the one thing worth checking.
  $("dial").classList.toggle("reading", !!data.adaptive);
  $("floor").value = data.floor;
  $("floor-v").textContent = Number(data.floor).toFixed(2);
  // Three states, not two: a cliff was found, the scores were read and had no
  // cliff in them, or the floor was never reached at all. Told apart by the
  // flags the run sends, never by the wording of the sentence.
  const tail = data.cliff_rank >= 0 ? ""
    : data.floor_read ? `, so the constant ${shipped} stands`
    : "";
  $("mode-why").innerHTML = data.adaptive
    ? `floor ${Number(data.floor).toFixed(2)} — ${esc(data.floor_why)}${tail}`
      + modeDelta(data)
    : "";
}

/* Which encoder the page is asking for. `null` means "whatever the server
   ships", so a page that has never switched sends nothing and cannot pin
   itself to an encoder that stopped being reachable. */
let encoderWanted = null;

function showEncoderBar(data) {
  if (!data.other_encoder) { $("encbar").hidden = true; return; }
  $("encbar").hidden = false;
  $("enc-shipped").textContent = data.shipped_encoder;
  $("enc-other").textContent = data.other_encoder;
  const onShipped = data.encoder === data.shipped_encoder;
  $("enc-shipped").classList.toggle("on", onShipped);
  $("enc-other").classList.toggle("on", !onShipped);
  $("enc-shipped").setAttribute("aria-pressed", String(onShipped));
  $("enc-other").setAttribute("aria-pressed", String(!onShipped));
  $("encoder-note").hidden = onShipped;
  $("enc-why").innerHTML = onShipped ? "" :
    `scored by <b>${esc(data.encoder)}</b>, not the encoder this build runs`
    + encoderDelta(data);
}

function setEncoder(which) {
  encoderWanted = which;
  if (lastMap) runMap();
}
$("enc-shipped").onclick = () => setEncoder(null);
$("enc-other").onclick = () => setEncoder(lastMap && lastMap.other_encoder);

let adaptiveFloor = false;

function setMode(on) {
  adaptiveFloor = on;
  $("mode-fixed").classList.toggle("on", !on);
  $("mode-adaptive").classList.toggle("on", on);
  $("mode-fixed").setAttribute("aria-pressed", String(!on));
  $("mode-adaptive").setAttribute("aria-pressed", String(on));
  $("adaptive-note").hidden = !on;
  if (!on) $("mode-why").textContent = "";
  $("dial").classList.toggle("reading", on);
  if (lastMap) runMap();
}
$("mode-fixed").onclick = () => setMode(false);
$("mode-adaptive").onclick = () => setMode(true);

async function runMap(floor, ratio) {
  const question = $("q").value.trim(), text = $("text").value;
  if (!question || !text.trim()) return;
  if (floor === undefined) floor = parseFloat($("floor").value);
  if (ratio === undefined) ratio = parseFloat($("ratio").value);
  // Adaptive mode computes the floor, so sending a slider position would draw
  // a line the run did not use.
  if (adaptiveFloor) floor = null;
  $("dial") && $("dial").classList.add("busy");
  try {
    const r = await fetch("/api/compress", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, text, floor, ratio, adaptive: adaptiveFloor,
                             encoder: encoderWanted }),
    });
    const data = await r.json();
    if (data.error) { $("map-err").textContent = data.error; return; }
    lastMap = data;
    lastRun.set(runKey(data), data);
    renderMap(data);
    floorLabel(data);
    showEncoderBar(data);
  } catch (e) { $("map-err").textContent = String(e); }
  $("dial") && $("dial").classList.remove("busy");
}

function dialsChanged() {
  if (adaptiveFloor) { $("mode-fixed").click(); return; }   // dragging means you want it
  $("floor-v").textContent = parseFloat($("floor").value).toFixed(2);
  $("ratio-v").textContent = Math.round(parseFloat($("ratio").value) * 100) + "%";
  clearTimeout(floorTimer);
  floorTimer = setTimeout(() => runMap(), 260);
}
$("floor").oninput = dialsChanged;
$("ratio").oninput = dialsChanged;

$("floor-reset").onclick = () => {
  $("floor").value = lastMap ? lastMap.shipped_floor : 0.15;
  dialsChanged();
};
$("dials-reset").onclick = () => {
  $("floor").value = lastMap ? lastMap.shipped_floor : 0.15;
  $("ratio").value = lastMap ? lastMap.shipped_ratio : 0.35;
  dialsChanged();
};

$("run-map").onclick = async () => {
  const question = $("q").value.trim(), text = $("text").value;
  $("map-err").textContent = "";
  if (!question || !text.trim()) {
    $("map-err").textContent = "a question and some text are both required"; return;
  }
  $("run-map").disabled = true;
  $("run-map").innerHTML = '<span class="spinner"></span>working';
  // Through `runMap`, not around it: the dials and the floor mode are part of
  // the request, and a second fetch here is a second thing to keep in step.
  await runMap();
  $("run-map").disabled = false;
  $("run-map").textContent = "Compress";
};

/* ════════════════════════════ A/B ═════════════════════════════════ */
const series = { parsimony: [], baseline: [] };
const COLOUR = { parsimony: "#4ec9a8", baseline: "#e5c07b" };

function abSize() {
  const n = $("ab-text").value.length;
  $("ab-ctx-size").textContent = n ? Math.round(n / 1024) + " KB attached" : "nothing yet";
}
$("ab-text").addEventListener("input", abSize);

$("copy-map").onclick = () => {
  $("abq").value = $("pq").value || $("q").value;
  $("ab-text").value = $("pipe-text").value || $("text").value;
  abSize();
};
$("ab-sample").onclick = async () => {
  $("ab-sample").disabled = true;
  try {
    const sample = await (await fetch("/api/sample")).json();
    $("ab-text").value = sample.text;
    if (!$("abq").value.trim()) $("abq").value = sample.question;
    abSize();
  } catch (e) { $("ab-err").textContent = "no sample document on this machine"; }
  $("ab-sample").disabled = false;
};

/* Match the backing store to the box the stylesheet gives the canvas, in
   device pixels, and then draw in CSS pixels. Without this the store's fixed
   1200x500 is stretched across and squashed down by whatever the panel happens
   to be -- labels twice as wide as they are tall, round markers as ellipses --
   and blurred on any screen with a pixel ratio above 1. */
function canvasBox(c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  const W = Math.max(320, Math.round(rect.width) || 320);
  const H = Math.max(160, Math.round(rect.height) || 250);
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
  }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, W, H };
}

function drawChart() {
  const c = $("chart");
  const { g, W, H } = canvasBox(c);
  // Top padding leaves room for the axis title ABOVE the highest gridline;
  // at 20 the title and the top tick label sat on top of each other.
  const pad = { l: 66, r: 20, t: 34, b: 42 };
  g.clearRect(0, 0, W, H);
  const all = [...series.parsimony, ...series.baseline];
  const maxT = Math.max(1, ...all.map((p) => p[0]));
  const maxN = Math.max(8, ...all.map((p) => p[1]));
  const X = (t) => pad.l + (W - pad.l - pad.r) * (t / maxT);
  const Y = (n) => H - pad.b - (H - pad.t - pad.b) * (n / maxN);
  g.strokeStyle = "#252c38"; g.fillStyle = "#8994a6";
  g.font = "13px ui-monospace, Consolas, monospace"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const n = Math.round(maxN * i / 4), y = Y(n);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.textAlign = "right"; g.fillText(String(n), pad.l - 10, y + 4);
  }
  g.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const t = maxT * i / 4;
    g.fillText(t.toFixed(1) + "s", X(t), H - pad.b + 20);
  }
  g.textAlign = "left"; g.fillText("answer tokens", pad.l - 56, pad.t - 14);
  for (const [arm, pts] of Object.entries(series)) {
    if (!pts.length) continue;
    g.strokeStyle = COLOUR[arm]; g.lineWidth = 2.5; g.beginPath();
    pts.forEach(([t, n], i) => (i ? g.lineTo(X(t), Y(n)) : g.moveTo(X(t), Y(n))));
    g.stroke();
    const [t, n] = pts[pts.length - 1];
    g.fillStyle = COLOUR[arm]; g.beginPath(); g.arc(X(t), Y(n), 4, 0, 7); g.fill();
    g.fillText(arm, X(t) + 9, Y(n) + 4);
  }
  if (!all.length) {
    g.fillStyle = "#8994a6"; g.textAlign = "center";
    g.fillText("run both arms to plot them", W / 2, H / 2);
  }
}
drawChart();
// Both the canvas's backing store and the channels' lane widths are computed
// from the box they are given, so the box changing invalidates both. Debounced:
// a drag emits an event per pixel and only the last one matters.
let onResize = null;
function relayout() {
  clearTimeout(onResize);
  onResize = setTimeout(() => {
    drawChart();
    if (RUN.plan && RUN.plan.length) { fillChannels(); paintAll(); }
  }, 120);
}
addEventListener("resize", relayout);

/* And watch the stack itself, because a channel's width changes for reasons a
   window resize never hears about -- the inspector opening beside it, a zoom,
   a font arriving late. The observer fires once on attach, which is harmless:
   there is nothing to lay out until a run has produced a plan. */
if (typeof ResizeObserver === "function" && $("stack")) {
  new ResizeObserver(relayout).observe($("stack"));
}

$("run-ab").onclick = () => {
  const question = $("abq").value.trim() || $("pq").value.trim() || $("q").value.trim();
  const text = $("ab-text").value || $("pipe-text").value || $("text").value;
  $("ab-err").textContent = "";
  if (!question) { $("ab-err").textContent = "ask something first"; return; }
  if (!text.trim()) {
    $("ab-err").textContent = "attach some context — with none, both arms send the same "
      + "prompt and there is nothing to compare";
    return;
  }
  series.parsimony = []; series.baseline = [];
  $("ab-ans-parsimony").textContent = ""; $("ab-ans-baseline").textContent = "";
  $("ab-tag-p").textContent = ""; $("ab-tag-b").textContent = "";
  $("ab-table-panel").hidden = true;
  drawChart();
  $("run-ab").disabled = true;
  $("run-ab").innerHTML = '<span class="spinner"></span>running';

  const es = new EventSource("/api/race?question=" + encodeURIComponent(question)
    + "&text=" + encodeURIComponent(text));
  es.addEventListener("note", (e) => {
    $("ab-live-note").textContent = JSON.parse(e.data).warmed
      ? "Model warmed with a throwaway token first, so neither arm pays the weight load." : "";
  });
  es.addEventListener("prompt", (e) => {
    const d = JSON.parse(e.data);
    $(d.arm === "parsimony" ? "ab-tag-p" : "ab-tag-b").textContent =
      fmt(d.tokens) + " prompt tokens — reading…";
  });
  es.addEventListener("token", (e) => {
    const d = JSON.parse(e.data);
    series[d.arm].push([d.t, d.n]);
    $("ab-ans-" + d.arm).textContent += d.text;
    if (d.n % 3 === 0) drawChart();
  });
  es.addEventListener("arm_done", (e) => {
    const d = JSON.parse(e.data);
    $(d.arm === "parsimony" ? "ab-tag-p" : "ab-tag-b").textContent =
      `${fmt(d.prompt_tokens)} prompt tokens · read in ${secs(d.prefill_s)}`
      + (d.reused ? " (reused earlier work)" : "");
    drawChart();
  });
  es.addEventListener("done", (e) => {
    renderAbTable(JSON.parse(e.data).arms); refreshSession(); finish();
  });
  es.addEventListener("failed", (e) => {
    $("ab-err").textContent = JSON.parse(e.data).error; finish();
  });
  es.onerror = () => finish();
  function finish() {
    es.close(); $("run-ab").disabled = false; $("run-ab").textContent = "Run both arms";
    drawChart();
  }
};

function renderAbTable(arms) {
  const p = arms.find((a) => a.arm === "parsimony");
  const b = arms.find((a) => a.arm === "baseline");
  if (!p || !b) return;
  const pct = (a, x) => {
    const d = 100 * (b[x] - a[x]) / (b[x] || 1);
    return (d > 0 ? "−" : "+") + Math.abs(d).toFixed(0) + "%";
  };
  const rows = [
    ["prompt tokens", fmt(p.prompt_tokens), fmt(b.prompt_tokens), pct(p, "prompt_tokens")],
    ["reading the prompt", secs(p.prefill_s), secs(b.prefill_s), pct(p, "prefill_s")],
    ["time to first token", p.ttft_s ? secs(p.ttft_s) : "—", b.ttft_s ? secs(b.ttft_s) : "—",
     (p.ttft_s && b.ttft_s) ? pct(p, "ttft_s") : ""],
    ["answer tokens", fmt(p.answer_tokens), fmt(b.answer_tokens), ""],
    ["wall clock", secs(p.total_s), secs(b.total_s), pct(p, "total_s")],
  ];
  $("ab-table").innerHTML =
    "<tr><th>measured</th><th>parsimony</th><th>baseline</th><th>change</th></tr>"
    + rows.map((r) => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td>`
      + `<td class="num">${r[2]}</td><td class="num">${r[3]}</td></tr>`).join("");
  $("ab-table-panel").hidden = false;
  const same = p.answer_tokens === b.answer_tokens;
  $("ab-note").textContent = ((p.reused || b.reused)
    ? "One arm reused earlier work from the runtime's cache, so its reading time is not a "
      + "cold measurement — rerun with a different question for a clean pair."
    : "Both arms ran cold, one after the other, on this machine.")
    + " Prompt tokens and reading time are what the layers changed. Wall clock also carries "
    + (same ? "generation" : `the two answers' different lengths (${p.answer_tokens} vs `
        + `${b.answer_tokens} tokens), which compression did not choose`)
    + " — read it as context, not as the result. Whether the two answers agree is for you "
    + "to judge: a faster wrong answer is not an improvement.";
}

refreshSession();
setInterval(() => { if (!$("view-demo").hidden) refreshSession(); }, 2000);
