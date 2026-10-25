"use strict";
// RegLens demo. Vanilla JS, no build step; served under a CSP that forbids inline script.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtDate = (d) => (d ? new Date(d + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric" }) : "");
const shortDate = (d) => new Date(d + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" });
const money = (a) => (a == null ? "" : "$" + Number(a).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 2 }));
const shortCit = (c) => c.replace(/^12 CFR /, "§").replace(/^Supplement I to Part \d+, /, "");
let META = null;

async function api(url, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : r.statusText);
  return data;
}
const highlight = (src, span) => (span ? esc(src.slice(0, span[0])) + "<mark>" + esc(src.slice(span[0], span[1])) + "</mark>" + esc(src.slice(span[1])) : esc(src));

// ---------------------------------------------------------------- tabs
function show(tab) {
  if (!$(`[data-panel="${tab}"]`)) tab = "copilot";
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$("[data-panel]").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== tab));
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", "#" + tab);
  if (tab === "results") loadResults();
  if (tab === "explore" && !$("#q").value) {  // a rich first view: the $50 rule, its interpretations and citers
    $("#q").value = "my debit card was stolen, how much could I lose";
    openGraph("12_CFR_1005_6_b_1").then(() => doSearch(false));
  }
}
$("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(b.dataset.tab); });
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// ---------------------------------------------------------------- meta / hero
async function loadMeta() {
  META = await api("/api/meta");
  const a = META.answering;
  $("#status").className = "status" + (a.enabled ? " on" : "");
  $("#status span").textContent = a.enabled ? `Answering live · ${META.model.provider}:${META.model.chat}` : "Deterministic tools live · answering off";
  $("#status").title = a.enabled ? `Limits: ${a.rate_per_hour} requests/hour per visitor, $${a.daily_usd_cap}/day` : a.reason;
  const k = $$("#kpis b");
  k[0].textContent = META.stats.passages.toLocaleString();
  k[1].textContent = META.graph.edges.toLocaleString();
  k[3].textContent = `${META.rules} / ${META.rules}`;
  api("/api/evals").then((e) => { const d = e.retrieval_all?.dense_bge; if (d) k[2].textContent = d["recall@5"].mean.toFixed(2); }).catch(() => {});
  $("#prov").innerHTML = `<span>eCFR as of <b>${esc(META.as_of)}</b></span>` +
    Object.entries(META.parts).map(([p, m]) => `<span title="${esc(m.name)}">Part ${esc(p)} <b>${esc(m.sha256)}</b></span>`).join("") +
    `<span>graph <b>${esc(META.graph.backend)}</b></span>`;
  const off = a.enabled ? "" : "Live answering is not enabled on this deployment yet; the calculator, checker, search and graph are fully live.";
  $("#runCopilot").disabled = $("#askBtn").disabled = !a.enabled;
  $("#copilotNote").textContent = $("#askNote").textContent = off;
  $("#cases").innerHTML = META.cases.map((c) => `<button data-v="${esc(c.case)}">${esc(c.id.replaceAll("-", " "))}</button>`).join("");
  $("#examples").innerHTML = META.examples.map((e) => `<button data-v="${esc(e.question)}">${esc(e.id)} · ${esc(e.question.slice(0, 56))}…</button>`).join("");
}
$("#cases").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) $("#case").value = b.dataset.v; });
$("#examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) $("#askq").value = b.dataset.v; });

// ---------------------------------------------------------------- deadline timeline (SVG)
const DEADLINE_NAMES = {
  "12 CFR 1005.6(b)(1)": "2-business-day notice", "12 CFR 1005.6(b)(3)": "60-day statement window",
  "12 CFR 1005.11(b)(1)(i)": "error notice deadline", "12 CFR 1005.11(b)(2)": "written confirmation",
  "12 CFR 1005.11(c)(1)": "decide, or credit provisionally", "12 CFR 1005.11(c)(3)(i)": "decide, or credit provisionally",
  "12 CFR 1005.11(c)(2)": "extended investigation ends", "12 CFR 1005.11(c)(3)(ii)": "extended investigation ends",
  "12 CFR 1005.11(c)(2)(i)": "provisional credit if extending",
  "12 CFR 1026.13(b)(1)": "billing notice deadline", "12 CFR 1026.13(c)(1)": "acknowledgment due", "12 CFR 1026.13(c)(2)": "resolution due",
};
function timelineSvg(inputs, items) {
  const groups = new Map(); // one marker per (date, side), listing everything on that day
  const add = (d, side, label, color) => { const k = d + side; if (!groups.has(k)) groups.set(k, { d, side, labels: [], color }); const g = groups.get(k); if (!g.labels.includes(label)) g.labels.push(label); };
  if (inputs.learned_of_loss) add(inputs.learned_of_loss, "up", "learned of loss", "var(--sky)");
  if (inputs.statement_sent) add(inputs.statement_sent, "up", "statement sent", "var(--sky)");
  (inputs.transfers || []).forEach((t) => add(t.date, "up", `${money(t.amount)} transfer`, "var(--rose)"));
  if (inputs.notified) add(inputs.notified, "up", "member notified", "var(--indigo)");
  items.filter((i) => i.due).forEach((i) => add(i.due, "down", DEADLINE_NAMES[i.citation] || shortCit(i.citation), "var(--amber)"));
  const pts = [...groups.values()]; if (!pts.length) return "";
  const t = (d) => new Date(d + "T12:00:00").getTime();
  const lo = Math.min(...pts.map((p) => t(p.d))), hi = Math.max(...pts.map((p) => t(p.d)));
  const W = 560, L = 24, R = 24, AX = 150, span = Math.max(hi - lo, 864e5), x = (d) => L + ((t(d) - lo) / span) * (W - L - R);
  const lanes = { up: [], down: [] }, LANE = 52;
  const lane = (side, px) => { const r = lanes[side]; let i = 0; while (r[i] !== undefined && px - r[i] < 150) i++; r[i] = px; return i; };
  let svg = ""; let minY = AX, maxY = AX;
  pts.sort((a, b) => t(a.d) - t(b.d)).forEach((p) => {
    const px = x(p.d), k = lane(p.side, px), up = p.side === "up";
    const lines = p.labels.slice(0, 3).concat(p.labels.length > 3 ? [`+${p.labels.length - 3} more`] : []);
    const y0 = up ? AX - 24 - k * LANE : AX + 24 + k * LANE;
    const anchor = px < 90 ? "start" : px > W - 90 ? "end" : "middle";
    const block = [`<tspan font-weight="700" fill="${p.color}">${esc(shortDate(p.d))}</tspan>`, ...lines.map((l) => esc(l))];
    const textY = up ? y0 - (block.length - 1) * 14 : y0 + 4;
    svg += `<line x1="${px}" y1="${AX}" x2="${px}" y2="${up ? y0 + 6 : y0 - 8}" stroke="${p.color}" stroke-width="1.5" ${up ? "" : 'stroke-dasharray="3 2"'}/>
      <circle cx="${px}" cy="${AX}" r="6.5" fill="${p.color}" stroke="var(--panel)" stroke-width="2"/>
      <text x="${px}" y="${textY}" font-size="12" text-anchor="${anchor}" fill="var(--ink)">${block.map((b, i) => `<tspan x="${px}" dy="${i ? 14 : 0}">${b}</tspan>`).join("")}</text>`;
    minY = Math.min(minY, textY - 14); maxY = Math.max(maxY, textY + (block.length - 1) * 14 + 6);
  });
  return `<svg class="timeline" viewBox="0 ${minY - 4} ${W} ${maxY - minY + 8}" role="img" aria-label="Case timeline: events above the line, regulatory deadlines below">
    <line x1="${L - 12}" y1="${AX}" x2="${W - R + 12}" y2="${AX}" stroke="var(--line)" stroke-width="3" stroke-linecap="round"/>${svg}</svg>`;
}

function timelineView(t, inputs) {
  const rows = t.items.map((i) => `<tr><td>${esc(i.label)}${i.note ? `<div class="muted">${esc(i.note)}</div>` : ""}</td>
    <td class="num amount">${i.amount != null ? `<b>${money(i.amount)}</b><br>` : ""}${i.due ? `<span class="${i.amount != null ? "muted" : ""}">${i.amount != null ? "notice by " : ""}${esc(fmtDate(i.due))}</span>` : ""}</td>
    <td><details><summary class="mono">${esc(i.citation)}</summary><div class="src">${esc(i.quote)}</div></details></td></tr>`).join("");
  return `${t.warnings.map((w) => `<div class="banner amber">${esc(w)}</div>`).join("")}
    ${inputs ? timelineSvg(inputs, t.items) : ""}
    <div class="tablewrap"><table><tr><th>What</th><th class="num">Limit / date</th><th>Rule (verbatim text)</th></tr>${rows}</table></div>
    <p class="muted" style="margin:10px 0 0">${esc(t.assumption)}</p>`;
}

// ---------------------------------------------------------------- copilot (streamed)
const rail = (stage, cls) => { const el = $(`.stage[data-stage="${stage}"]`); if (!el) return; el.classList.remove("active", "done", "fail"); if (cls) el.classList.add(cls); };
function resetRail() { $$(".stage").forEach((s) => s.classList.remove("active", "done", "fail")); }
const TOOL_CLASS = { search_regulations: "t-search", related_paragraphs: "t-graph", compute_deadlines: "t-deadline", submit_answer: "t-ok" };

function traceItem(step) {
  if (step.node === "agent") {
    if (step.error) return `<li class="t-bad"><b>model</b> ${esc(step.error)}</li>`;
    const calls = step.tool_calls.length ? step.tool_calls.map((c) => `<span class="mono">${esc(c)}</span>`).join(", ") : "no tool call";
    return `<li><b>model</b> → ${calls} <span class="muted">${step.ms ?? ""} ms</span></li>`;
  }
  const res = Array.isArray(step.result) ? step.result.slice(0, 4).map(shortCit).join(" · ") : step.result;
  const bad = String(res).startsWith("error");
  const args = Object.entries(step.args || {}).filter(([k]) => k !== "claims" && k !== "reply_to_member")
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ").slice(0, 160);
  return `<li class="${bad ? "t-bad" : TOOL_CLASS[step.tool] || ""}"><span class="mono"><b>${esc(step.tool)}</b></span> <span class="muted">${step.ms} ms</span>
    <div class="args">${esc(args)}</div><div>${esc(res)}</div></li>`;
}

function renderCopilotResult(r, usage, redactions) {
  const x = r;
  const verdict = x.draft_withheld ? `<span class="pill p-rose">reply withheld</span>` : `<span class="pill p-emerald">${x.claims.length} verified citation${x.claims.length === 1 ? "" : "s"}</span>`;
  let h = `<h2>Result</h2><div class="row" style="margin-top:6px"><span class="pill p-indigo">${esc(x.case_type.replaceAll("_", " "))}</span>${verdict}
    <span class="pill p-sky">${usage.calls} model calls · $${(usage.usd ?? 0).toFixed(4)}</span></div>`;
  if (Object.keys(redactions).length) h += `<div class="banner sky">Masked before the model saw it: ${Object.entries(redactions).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ")}</div>`;
  if (x.draft_withheld) h += `<div class="banner rose"><b>Withheld for a specialist.</b> ${esc(x.review_reason)}</div>`;
  else if (x.needs_human_review) h += `<div class="banner amber"><b>Representative action required.</b> ${esc(x.review_reason)}</div>`;
  if (x.reply_to_member) h += `<h3>Draft reply to member</h3><div class="reply">${esc(x.reply_to_member)}</div><button class="btn" data-copy>Copy reply</button>`;
  if (x.claims.length) h += `<h3>Verified citations</h3>` + x.claims.map((c) => `<div class="claim"><div>${esc(c.statement)}</div><div class="cit">${esc(c.citation)}</div>
    <blockquote>“${esc(c.quote)}”</blockquote><details><summary>Source passage</summary><div class="src">${highlight(c.source, c.span)}</div></details></div>`).join("");
  if (x.handoff) h += `<h3>Handoff ticket</h3><div class="ticket"><div class="meta"><span class="pill ${x.handoff.priority === "urgent" ? "p-rose" : "p-indigo"}">${esc(x.handoff.priority)}</span>
    queue <b>${esc(x.handoff.queue)}</b></div><p style="margin:8px 0 0">${esc(x.handoff.summary)}</p><ul>${x.handoff.next_actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></div>`;
  $("#copilotOut").innerHTML = h;
  const cp = $("[data-copy]"); if (cp) cp.addEventListener("click", () => navigator.clipboard?.writeText(x.reply_to_member));
  const checks = (x.checks || []).map((c) => `<li class="${c.verified ? "t-ok" : "t-bad"}"><b>verify</b> <span class="mono">${esc(shortCit(c.citation || c.chunk_id))}</span> → ${esc(c.reason)}</li>`);
  const events = (x.events || []).map((e) => `<li class="t-bad"><b>finalize</b> ${esc(e)}</li>`);
  $("#trace ul").insertAdjacentHTML("beforeend", checks.join("") + events.join(""));
  $("#timelineCard").classList.toggle("hidden", !x.timelines.length);
  $("#copilotTimeline").innerHTML = x.timelines.map((t) => timelineView(t)).join("<hr>");
}

function beginCopilot(label) {
  const out = $("#copilotOut"); out.classList.remove("hidden");
  out.innerHTML = `<span class="spinner"></span> <span class="muted">${esc(label)}</span>`;
  $("#trace").innerHTML = `<ul class="trace"></ul>`; resetRail(); rail("guard", "active");
}

// One handler for live server-sent events and for replays of recorded runs.
function onCopilotEvent(ev) {
  if (ev.type === "start") { rail("guard", "done"); rail("agent", "active"); }
  else if (ev.type === "step") {
    $("#trace ul").insertAdjacentHTML("beforeend", traceItem(ev.step));
    if (ev.step.node === "agent") { rail("agent", "active"); rail("tools", ""); }
    else { rail("agent", "done"); rail("tools", ev.step.tool === "submit_answer" ? "done" : "active"); if (ev.step.tool === "submit_answer") rail("verify", "active"); }
  } else if (ev.type === "result") {
    ["agent", "tools"].forEach((s) => rail(s, "done"));
    rail("verify", ev.result.events.some((e) => !e.startsWith("handoff")) ? "fail" : "done");
    rail("final", ev.result.draft_withheld ? "fail" : "done");
    renderCopilotResult(ev.result, ev.usage, ev.redactions);
  } else if (ev.type === "error") throw new Error(ev.detail);
}

async function runCopilot() {
  const text = $("#case").value.trim(); if (!text) return;
  beginCopilot("Working…"); $("#replayNote").textContent = "";
  $("#runCopilot").disabled = true;
  try {
    const resp = await fetch("/api/copilot/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case: text }) });
    if (!resp.ok) { const d = await resp.json().catch(() => ({})); throw new Error(d.detail || resp.statusText); }
    const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const data = block.split("\n").find((l) => l.startsWith("data: ")); if (data) onCopilotEvent(JSON.parse(data.slice(6)));
      }
    }
  } catch (e) { $("#copilotOut").innerHTML = `<div class="err">${esc(e.message)}</div>`; $$(".stage.active").forEach((s) => s.classList.replace("active", "fail")); }
  finally { $("#runCopilot").disabled = !META?.answering.enabled; }
}

// Recorded runs: real model output captured by scripts/record_copilot_runs.py, replayed at its recorded pace.
let RECORDED = [];
async function loadRecorded() {
  RECORDED = (await api("/api/copilot/recorded")).runs;
  $("#recorded").innerHTML = RECORDED.map((r, i) => `<button data-i="${i}">▶ ${esc(r.title)}</button>`).join("");
  $("#recordedWrap").classList.toggle("hidden", !RECORDED.length);
}
async function replay(i) {
  const run = RECORDED[i];
  $("#case").value = run.case;
  $("#replayNote").innerHTML = `Replaying a <b>recorded real run</b> · ${esc(run.model)} · ${esc(run.recorded_at)} · commit ${esc(run.commit)}. Nothing is sent to a model.`;
  beginCopilot("Replaying…");
  for (const ev of run.events) {
    const ms = ev.type === "step" ? Math.min(900, Math.max(250, (ev.step.ms || 300) / 6)) : 250;
    await new Promise((r) => setTimeout(r, ms));
    onCopilotEvent(ev);
  }
}
$("#recorded").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) replay(Number(b.dataset.i)); });
$("#runCopilot").addEventListener("click", runCopilot);

// ---------------------------------------------------------------- deadline calculator
const DL_PRESETS = {
  "stolen card, called next morning": { type: "unauthorized_eft", notified: "2026-10-21", learned: "2026-10-20", statement: "", tx: [["2026-10-20", 400]] },
  "reported three weeks late": { type: "unauthorized_eft", notified: "2026-10-23", learned: "2026-10-01", statement: "", tx: [["2026-10-02", 40], ["2026-10-15", 900]] },
  "statement ignored for 60+ days": { type: "unauthorized_eft", notified: "2026-12-15", learned: "", statement: "2026-09-30", tx: [["2026-09-20", 120], ["2026-12-05", 300]] },
  "credit card billing error": { type: "credit_card_billing_error", notified: "2026-10-01", learned: "", statement: "2026-09-15", tx: [], unauth: true },
};
function addTransfer(d = "", a = "") {
  const row = document.createElement("div"); row.className = "row"; row.style.marginTop = "6px";
  row.innerHTML = `<input type="date" value="${esc(d)}"><input type="number" min="0" step="0.01" placeholder="amount" value="${esc(a)}" style="width:130px"><button class="btn" type="button">Remove</button>`;
  $("button", row).addEventListener("click", () => row.remove());
  $("#dl_transfers").appendChild(row);
}
function loadPreset(name) {
  const p = DL_PRESETS[name];
  $("#dl_type").value = p.type; $("#dl_notified").value = p.notified; $("#dl_learned").value = p.learned; $("#dl_statement").value = p.statement;
  $("#dl_transfers").innerHTML = ""; p.tx.forEach(([d, a]) => addTransfer(d, a));
  $("#dl_new").checked = false; $("#dl_pos").checked = false; $("#dl_unauth").checked = !!p.unauth;
  runDeadlines();
}
$("#dl_presets").innerHTML = Object.keys(DL_PRESETS).map((n) => `<button data-v="${esc(n)}">${esc(n)}</button>`).join("");
$("#dl_presets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) loadPreset(b.dataset.v); });
$("#dl_add").addEventListener("click", () => addTransfer());
async function runDeadlines() {
  const body = {
    case_type: $("#dl_type").value, notified: $("#dl_notified").value, learned_of_loss: $("#dl_learned").value || null,
    statement_sent: $("#dl_statement").value || null, new_account: $("#dl_new").checked, pos_or_foreign: $("#dl_pos").checked, unauthorized_use: $("#dl_unauth").checked,
    transfers: $$("#dl_transfers .row").map((r) => { const [d, a] = $$("input", r); return { date: d.value, amount: Number(a.value) }; }).filter((t) => t.date && t.amount > 0),
  };
  try { $("#dl_out").innerHTML = timelineView(await api("/api/timeline", body), body); }
  catch (e) { $("#dl_out").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#dl_run").addEventListener("click", runDeadlines);

// ---------------------------------------------------------------- checker
const CK_PASSAGES = ["12 CFR 1005.11(c)(1)", "12 CFR 1005.6(b)(1)", "12 CFR 1026.13(c)(2)"];
const slug = (c) => c.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
const CK = {};
async function loadChecker() {
  for (const cit of CK_PASSAGES) {
    const c = await api(`/api/chunk/${slug(cit)}`);
    CK[c.id] = (c.context ? c.context + " " : "") + c.text;
    $("#ck_passage").insertAdjacentHTML("beforeend", `<option value="${esc(c.id)}">${esc(c.citation)}</option>`);
  }
  $("#ck_passage").addEventListener("change", onPassage); onPassage();
}
let PRESETS = [];
function onPassage() {
  const src = CK[$("#ck_passage").value];
  $("#ck_source").textContent = src;
  const sentences = src.split(/(?<=\.)\s+/).map((s) => s.trim()).filter((s) => s.length > 40);
  const exact = sentences.find((s) => /\b\d{2,}\b|\$\d/.test(s)) || sentences[0];
  PRESETS = [
    ["exact quote", exact, true], ["change a number", exact.replace(/\$?\d+/, (n) => (n.startsWith("$") ? "$" + Number(n.slice(1)) * 3 : String(Number(n) * 3))), true],
    ["paraphrase", "The credit union has to look into errors quickly and fix them fast.", true], ["real quote, passage not retrieved", exact, false],
    ["too short", exact.split(" ").slice(0, 3).join(" "), true],
  ];
  $("#ck_presets").innerHTML = PRESETS.map((p, i) => `<button data-i="${i}">${esc(p[0])}</button>`).join("");
  usePreset(1);
}
function usePreset(i) { const [, q, r] = PRESETS[i]; $("#ck_quote").value = q; $("#ck_retrieved").checked = r; runCheck(); }
$("#ck_presets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) usePreset(Number(b.dataset.i)); });
const WHY = {
  ok: "Verbatim match in a retrieved passage: this claim is shown to the member.",
  quote_not_in_source: "Not in the passage. The claim is removed before the member sees it.",
  not_retrieved: "The text exists, but this passage was not retrieved for the question, so the model could not have read it. Removed.",
  quote_too_short: "Under 25 characters: too short to carry meaning. Removed.", unknown_chunk: "No such passage. Removed.",
};
async function runCheck() {
  try {
    const r = await api("/api/verify", { chunk_id: $("#ck_passage").value, quote: $("#ck_quote").value, retrieved: $("#ck_retrieved").checked });
    const nq = r.normalized_quote, m = r.matched_prefix;
    const diff = r.reason === "quote_not_in_source" && nq
      ? `<h3>Where it diverges</h3><div class="diff"><span class="m">${esc(nq.slice(0, m))}</span><span class="x">${esc(nq.slice(m))}</span></div>
         <p class="muted">Green: the longest opening stretch found verbatim in the passage (${m} of ${nq.length} characters, normalized). Red: from the first character that is not.</p>` : "";
    $("#ck_out").innerHTML = `<div class="stamp ${r.verified ? "ok" : "bad"}">${r.verified ? "ACCEPTED" : "REJECTED"}</div>
      <div><span class="pill ${r.verified ? "p-emerald" : "p-rose"} mono">${esc(r.reason)}</span></div><p>${esc(WHY[r.reason])}</p>${diff}
      ${r.verified ? `<h3>Matched in source</h3><div class="src">${highlight(r.source, r.span)}</div>` : ""}`;
  } catch (e) { $("#ck_out").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#ck_run").addEventListener("click", runCheck);

// ---------------------------------------------------------------- search + citation graph
async function doSearch(openFirst = true) {
  const q = $("#q").value.trim(); if (!q) return;
  $("#hits").innerHTML = `<span class="spinner"></span>`;
  try {
    const r = await api(`/api/search?q=${encodeURIComponent(q)}&retriever_name=${$("#ret").value}`);
    $("#hits").innerHTML = (Object.keys(r.redactions).length ? `<div class="banner sky">Masked: ${esc(Object.keys(r.redactions).join(", "))}</div>` : "") +
      `<div class="muted">${r.hits.length} results · ${r.latency_ms} ms · ${esc(r.retriever)}</div>` +
      r.hits.map((h) => `<div class="hit" data-id="${esc(h.id)}"><div class="cit">${h.rank}. ${esc(h.citation)}</div><div class="meta">${esc(h.heading)} · ${esc(h.kind)} · ${h.score}</div><div>${esc(h.text)}${h.text.length >= 400 ? "…" : ""}</div></div>`).join("");
    const hits = $$(".hit", $("#hits")); if (hits.length && openFirst) openGraph(hits[0].dataset.id);
  } catch (e) { $("#hits").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#searchBtn").addEventListener("click", () => doSearch());
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
$("#hits").addEventListener("click", (e) => { const h = e.target.closest(".hit"); if (h) openGraph(h.dataset.id); });

const GROUP = (n) => (n.relation === "INTERPRETS" ? (n.direction === "in" ? ["official interpretation", "var(--violet)"] : ["rule it explains", "var(--indigo)"])
  : n.direction === "out" ? ["cites", "var(--sky)"] : ["cited by", "var(--faint)"]);
async function openGraph(id) {
  $$(".hit").forEach((x) => x.classList.toggle("sel", x.dataset.id === id));
  const g = await api(`/api/graph/${encodeURIComponent(id)}`);
  const ns = g.neighbors.slice(0, 16), W = 560, H = 360, cx = W / 2, cy = H / 2, R = 125;
  const nodes = ns.map((n, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(ns.length, 1); return { ...n, x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a }; });
  const svg = `<svg class="graphsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="citation graph">
    ${nodes.map((n) => `<line x1="${cx}" y1="${cy}" x2="${n.x}" y2="${n.y}" stroke="${GROUP(n)[1]}" stroke-width="1.6" opacity=".6"/>`).join("")}
    <circle cx="${cx}" cy="${cy}" r="16" fill="var(--violet)" stroke="var(--violet-soft)" stroke-width="6"/>
    <text x="${cx}" y="${cy + 38}" font-size="12.5" text-anchor="middle" fill="var(--ink)" font-weight="700">${esc(shortCit(g.citation))}</text>
    ${nodes.map((n) => { const c = Math.cos(n.a), anchor = c > 0.3 ? "start" : c < -0.3 ? "end" : "middle", lx = n.x + c * 14, ly = n.y + Math.sin(n.a) * 14 + 4;
      return `<g class="node" data-id="${esc(n.id)}"><title>${esc(n.citation)}\n${esc(n.text)}</title><circle cx="${n.x}" cy="${n.y}" r="9" fill="${GROUP(n)[1]}" stroke="var(--panel)" stroke-width="2"/>
      <text x="${lx}" y="${ly}" font-size="10.5" text-anchor="${anchor}" fill="var(--muted)">${esc(shortCit(n.citation).slice(0, 26))}</text></g>`; }).join("")}
  </svg>`;
  const more = g.neighbors.length > ns.length ? `<p class="muted">Showing 16 of ${g.neighbors.length} neighbors.</p>` : "";
  const groups = {}; g.neighbors.forEach((n) => (groups[GROUP(n)[0]] = (groups[GROUP(n)[0]] || 0) + 1));
  $("#graphOut").innerHTML = `<div class="row" style="margin:0 0 6px"><span class="pill p-violet mono">${esc(g.citation)}</span>
    ${Object.entries(groups).map(([k, v]) => `<span class="pill p-indigo">${v} ${esc(v === 1 || k.startsWith("cite") ? k : k.replace("interpretation", "interpretations").replace("rule ", "rules "))}</span>`).join("")}</div>
    ${g.neighbors.length ? svg + more : `<p class="muted">No cross-references or interpretations link to this paragraph.</p>`}
    <p class="muted">Click a node to walk the graph.</p>`;
}
$("#graphOut").addEventListener("click", (e) => { const n = e.target.closest(".node"); if (n) openGraph(n.dataset.id); });

// ---------------------------------------------------------------- Q&A pipeline
async function doAsk() {
  const q = $("#askq").value.trim(); if (!q) return;
  const out = $("#askOut"); out.classList.remove("hidden"); out.innerHTML = `<span class="spinner"></span>`;
  try {
    const r = await api("/api/ask", { question: q }), f = r.final, u = r.usage;
    const tone = f.status === "answered" ? "p-emerald" : f.status === "refused" ? "p-rose" : "p-amber";
    out.innerHTML = `<h2>Answer</h2><span class="pill ${tone}">${esc(f.status.replaceAll("_", " "))}</span>
      ${f.needs_human_review ? `<div class="banner amber"><b>Routed to a specialist.</b> ${esc(f.review_reason)}</div>` : ""}<div class="reply">${esc(f.answer)}</div>` +
      f.claims.map((c) => `<div class="claim"><div>${esc(c.statement)}</div><div class="cit">${esc(c.citation)}</div><blockquote>“${esc(c.quote)}”</blockquote><details><summary>Source passage</summary><div class="src">${highlight(c.source, c.span)}</div></details></div>`).join("");
    $("#askTrace").innerHTML = `<div class="tablewrap"><table><tr><th>Stage</th><th class="num">ms</th></tr>${r.stages.map((s) => `<tr><td>${esc(s.node)}</td><td class="num">${s.ms}</td></tr>`).join("")}</table></div>
      <div class="row"><span class="pill p-sky">${u.calls} calls</span><span class="pill p-sky">${u.input_tokens} in / ${u.output_tokens} out tok</span><span class="pill p-sky">${u.usd == null ? "–" : "$" + u.usd.toFixed(5)}</span></div>
      <h3>Verifier</h3><ul class="trace">${r.checks.map((c) => `<li class="${c.verified ? "t-ok" : "t-bad"}"><span class="mono">${esc(shortCit(c.citation || c.chunk_id))}</span> → ${esc(c.reason)}</li>`).join("") || "<li>no claims</li>"}</ul>
      ${r.events.length ? `<h3>Events</h3><ul>${r.events.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}`;
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#askBtn").addEventListener("click", doAsk);

// ---------------------------------------------------------------- results
let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return; resultsLoaded = true;
  const e = await api("/api/evals");
  const all = e.retrieval_all || {}, names = Object.keys(all).sort((a, b) => all[b]["recall@5"].mean - all[a]["recall@5"].mean);
  $("#bars").innerHTML = names.map((n, i) => { const m = all[n]["recall@5"]; const cls = i === 0 ? "best" : n.startsWith("graph") ? "graph" : "";
    return `<div class="bar ${cls}"><span class="lbl" title="${esc(n)}">${esc(n)}</span><div class="track"><div class="fill" style="width:${100 * m.mean}%"></div>
      <div class="ci" style="left:${100 * m.lo}%;width:${100 * (m.hi - m.lo)}%"></div></div><span class="mono">${m.mean.toFixed(2)}</span></div>`; }).join("");
  $("#barsNote").textContent = "Dense bge beats BM25 by +0.52 and naive RRF hybrid by +0.30 (paired, CIs exclude 0). Neither a general-domain cross-encoder nor citation-graph expansion improves first-stage recall, so the graph serves the agent instead of ranking.";
  if (e.answers) {
    const ci = (m) => (m && m.n ? `${m.mean.toFixed(2)} <span class="muted">(${m.ci95[0].toFixed(2)}–${m.ci95[1].toFixed(2)})</span>` : "–");
    const pct = (v) => (v == null ? "–" : (100 * v).toFixed(1) + "%");
    $("#answers").innerHTML = `<div class="tablewrap"><table><tr><th>Configuration</th><th class="num">Answer accuracy</th><th class="num">Unsupported / fabricated shown</th></tr>
      ${Object.entries(e.answers.configs).map(([n, s]) => `<tr><td>${esc(n)}</td><td class="num">${ci(s.answer_accuracy)}</td><td class="num">${pct(s.unsupported_claim_rate_shown ?? s.fabricated_citation_rate)}</td></tr>`).join("")}</table></div>
      <p class="muted">${esc(e.answers.run)} · ${esc(e.answers.provenance?.chat_deployment || "")}</p>`;
  } else {
    $("#answers").innerHTML = `<p>Closed book vs RAG vs RAG + verifier vs the copilot, on 43 questions (33 answerable, 4 out of corpus, 6 refusal including 5 adversarial).</p>
      <div class="banner amber"><b>Not run yet.</b> It needs a model; no answer-quality number is shown until a run exists.</div>`;
  }
  if (e.stores) {
    const q = e.stores.qdrant, n = e.stores.neo4j;
    $("#stores").innerHTML = `<div class="stats"><div class="stat emerald"><b>${esc(q.exact_top_k_identical)}</b><span>Qdrant exact top-8 identical to numpy</span></div>
      <div class="stat sky"><b>${Number(q.hnsw_recall_at_k_vs_exact).toFixed(2)}</b><span>HNSW recall@8 vs exact</span></div>
      <div class="stat emerald"><b>${n.neighbor_mismatches} / ${n.nodes_checked.toLocaleString()}</b><span>Neo4j neighbor mismatches</span></div>
      <div class="stat violet"><b>${n.edges.toLocaleString()}</b><span>edges: ${n.graph_stats.by_type.INTERPRETS} interprets, ${n.graph_stats.by_type.CITES} cites</span></div></div>
      <p class="muted">${esc(e.stores.run)} · commit ${esc(e.stores.provenance.git_commit)}</p>`;
  }
}

// ---------------------------------------------------------------- boot
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { const b = $("[data-panel]:not(.hidden) .btn.primary:not(:disabled), [data-panel]:not(.hidden) .btn.amber"); if (b) b.click(); }
});
addTransfer("2026-10-20", "400");
show(location.hash.slice(1) || "copilot");
loadMeta().catch((e) => { $("#status span").textContent = "Offline: " + e.message; });
loadChecker().catch(() => {});
loadRecorded().catch(() => {});
runDeadlines();
