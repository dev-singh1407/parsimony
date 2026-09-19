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

/* ── the stack ────────────────────────────────────────────────────── */
function drawStack() {
  const byModule = [];
  for (const s of RUN.plan) {
    const last = byModule[byModule.length - 1];
    if (last && last.module === s.module) last.items.push(s);
    else byModule.push({ module: s.module, items: [s] });
  }
  $("stack").innerHTML = byModule.map((lane) =>
    `<div class="lane"><div class="mid">${esc(lane.module)}</div>`
    + `<div class="cells">`
    + lane.items.map((s) =>
      `<div class="cell" id="cell-${s.name}" data-stage="${esc(s.name)}">`
      + `<div class="cn">${esc(s.label)}</div>`
      + `<div class="cj">${esc(s.does)}</div>`
      + `<div class="cs"><span class="w" id="cw-${s.name}">—</span>`
      + `<span class="d" id="cd-${s.name}"></span></div>`
      + `<span class="carry" id="cc-${s.name}"></span></div>`).join("")
    + `</div></div>`).join("");
  for (const s of RUN.plan) {
    $("cell-" + s.name).onclick = () => { RUN.selected = s.name; paintAll(); };
  }
}

/* ── the transport ────────────────────────────────────────────────── */
function drawTrack() {
  const total = RUN.stages.reduce((a, s) => a + Math.max(s.ms, 0.05), 0) || 1;
  $("track").innerHTML = RUN.stages.map((s, i) => {
    const pct = (Math.max(s.ms, 0.05) / total) * 100;
    const label = pct > 7 ? esc(labelFor(s.name)) : "";
    return `<div class="seg ${s.outcome}" id="seg-${i}" style="flex:0 0 ${pct}%"`
         + ` title="${esc(labelFor(s.name))} — ${s.ms.toFixed(1)}ms">`
         + `<div class="segfill"></div><div class="seglab">${label}</div></div>`;
  }).join("");
  RUN.stages.forEach((s, i) => { $("seg-" + i).onclick = () => goTo(i); });
  $("tick-r").textContent = total.toFixed(0) + " ms of middleware";
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
    const cell = $("cell-" + p.name);
    if (!cell) continue;
    const i = RUN.stages.findIndex((s) => s.name === p.name);
    const ran = i >= 0 && i <= at;
    const s = i >= 0 ? RUN.stages[i] : null;
    cell.className = "cell"
      + (ran ? " past " + s.outcome : "")
      + (i === at ? " at" : "")
      + (RUN.selected === p.name ? " sel" : "");
    $("cw-" + p.name).textContent = ran ? `${fmt(s.after)} tok` : "—";
    const d = ran ? s.after - s.before : 0;
    $("cd-" + p.name).textContent = ran && d < 0 ? `−${fmt(-d)}` : "";
  }
  drawCarry();
  // the now-line
  const s = at >= 0 ? RUN.stages[at] : null;
  $("t-now").innerHTML = s
    ? `<b>${esc(labelFor(s.name))}</b> <span class="sub">· ${s.ms.toFixed(1)} ms · `
      + `${esc(OUTCOME_WORD[s.outcome] || s.outcome)} · ${fmt(s.before)} → ${fmt(s.after)} tokens`
      + `</span>`
    : `<span class="sub">${RUN.stages.length ? "before anything ran — press play, or use ← →"
        : "run something, then scrub through it stage by stage"}</span>`;
  paintDoc();
  paintInspector();
  paintLayers();
}

/* The width of each bar is that layer's outgoing token count as a fraction of
   what arrived at the top, so the staircase down the stack is the compression. */
function drawCarry() {
  const start = RUN.stages.length ? RUN.stages[0].before : 0;
  for (const p of RUN.plan) {
    const bar = $("cc-" + p.name);
    if (!bar) continue;
    const i = RUN.stages.findIndex((s) => s.name === p.name);
    const ran = i >= 0 && i <= RUN.at;
    bar.style.width = ran && start
      ? (Math.max(0.03, RUN.stages[i].after / start) * 100).toFixed(1) + "%" : "0";
  }
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

/* ── running one request ──────────────────────────────────────────── */
fetch("/api/plan").then((r) => r.json()).then((p) => {
  RUN.plan = p.stages; drawStack(); paintAll();
}).catch(() => {});

$("pipe-sample").onclick = async () => {
  try {
    const s = await (await fetch("/api/sample")).json();
    $("pipe-text").value = s.text;
    $("pipe-ctx-size").textContent = Math.round(s.text.length / 1024) + " KB attached";
    if (!$("pq").value.trim()) $("pq").value = s.question;
  } catch (e) { $("pipe-err").textContent = "no sample document on this machine"; }
};
$("pipe-text").addEventListener("input", () => {
  const n = $("pipe-text").value.length;
  $("pipe-ctx-size").textContent = n ? Math.round(n / 1024) + " KB attached" : "nothing yet";
});

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
    $("pipe-timing").textContent =
      `Middleware took ${t.middleware_ms.toFixed(0)} ms and saved ${t.saved_s.toFixed(1)} s `
      + `of prefill at ${t.ms_per_token.toFixed(1)} ms/token`
      + (t.timed_here ? ", measured on this machine."
        : t.reused ? " — but the runtime reused an earlier prompt, so that rate is not cold."
        : " — the model is simulated, so that rate is the project's recorded figure.");
    buildDoc(); drawTrack(); goTo(-1);
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

function renderMap(data) {
  $("map-stats").hidden = $("map-doc-panel").hidden = false;
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
  refreshSession();
}

const tip = $("tip");
function tipFor(el, kept) {
  tip.innerHTML = `<b>[${kept ? "KEEP" : "DROP"}: ${esc(el.dataset.tag)}]</b><br>`
    + `${esc(el.dataset.detail)}<br><span class="dimtext">relevance `
    + `${el.dataset.score}${el.dataset.tokens ? " · " + el.dataset.tokens + " tokens" : ""}`
    + `</span>`;
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

$("run-map").onclick = async () => {
  const question = $("q").value.trim(), text = $("text").value;
  $("map-err").textContent = "";
  if (!question || !text.trim()) {
    $("map-err").textContent = "a question and some text are both required"; return;
  }
  $("run-map").disabled = true;
  $("run-map").innerHTML = '<span class="spinner"></span>working';
  try {
    const r = await fetch("/api/compress", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, text }),
    });
    const data = await r.json();
    if (data.error) $("map-err").textContent = data.error; else renderMap(data);
  } catch (e) { $("map-err").textContent = String(e); }
  $("run-map").disabled = false;
  $("run-map").textContent = "Compress";
};

/* ════════════════════════════ A/B ═════════════════════════════════ */
const series = { parsimony: [], baseline: [] };
const COLOUR = { parsimony: "#4ec9a8", baseline: "#e5c07b" };

$("copy-map").onclick = () => show("map");

function drawChart() {
  const c = $("chart"), g = c.getContext("2d");
  const W = c.width, H = c.height, pad = { l: 66, r: 20, t: 20, b: 42 };
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
  g.textAlign = "left"; g.fillText("answer tokens", pad.l - 56, pad.t - 4);
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

$("run-ab").onclick = () => {
  const question = $("pq").value.trim() || $("q").value.trim();
  const text = $("pipe-text").value || $("text").value;
  $("ab-err").textContent = "";
  if (!question || !text.trim()) {
    $("ab-err").textContent = "set a question and some text on the Pipeline or Heatmap tab first";
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
