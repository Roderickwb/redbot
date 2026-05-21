from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.analysis.operator_decisions import record_operator_decision
from src.operator_app.backend.auth import require_operator_token
from src.operator_app.backend.data import REPORTS, mobile_bundle, recent_trades, report
from src.operator_app.backend.schemas import DecisionRequest


load_dotenv()

APP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = APP_ROOT / "frontend" / "dist"

app = FastAPI(
    title="Red Bot Operator App",
    version="0.1.0",
    description="Mobile operator control API. V1 write actions are append-only and have no live trading effect.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("OPERATOR_APP_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "OK",
        "app": "redbot-operator-app",
        "live_effect": False,
        "reports": REPORTS,
    }


@app.get("/api/snapshot")
def get_snapshot() -> dict:
    return report("snapshot")


@app.get("/api/mobile")
def get_mobile_bundle(limit: int = Query(default=40, ge=1, le=200)) -> dict:
    return mobile_bundle(trade_limit=limit)


@app.get("/api/cockpit")
def get_cockpit() -> dict:
    return report("cockpit")


@app.get("/api/recommendations")
def get_recommendations() -> dict:
    return report("recommendations")


@app.get("/api/recommendation-quality")
def get_recommendation_quality() -> dict:
    return report("recommendation_quality")


@app.get("/api/operator-decisions")
def get_operator_decisions() -> dict:
    return report("operator_decisions")


@app.get("/api/safety")
def get_safety() -> dict:
    return report("safety")


@app.get("/api/positions")
def get_positions() -> dict:
    return report("positions")


@app.get("/api/exits")
def get_exits() -> dict:
    return report("exits")


@app.get("/api/trades")
def get_trades(limit: int = Query(default=100, ge=1, le=500), symbol: Optional[str] = "") -> dict:
    return recent_trades(limit=limit, symbol=symbol or None)


@app.post("/api/decisions", dependencies=[Depends(require_operator_token)])
def post_decision(payload: DecisionRequest) -> dict:
    if payload.action in {"enable_live_enforcement", "risk_up_live", "entry_rule_live", "ml_live", "clear_kill_switch"}:
        raise HTTPException(status_code=403, detail="Live-effect commands are forbidden in app v1")
    item = record_operator_decision(
        source_id=payload.source_id,
        source_type=payload.source_type,
        action=payload.action,
        operator=payload.operator,
        reason=payload.reason,
        scope=payload.scope,
        source_path=payload.source_path,
        expires_utc=payload.expires_utc,
    )
    return {
        "status": "OK",
        "decision": item,
        "live_effect": False,
    }


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


FALLBACK_HTML = """
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Red Bot Operator</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --bg: #06080c;
      --panel: #0d1219;
      --panel-2: #111821;
      --panel-3: #151f2a;
      --line: #253141;
      --line-hot: #4c1722;
      --text: #f5f7fb;
      --muted: #8c99aa;
      --red: #e22d3f;
      --red-2: #ff4056;
      --blue: #3b82f6;
      --green: #14b8a6;
      --amber: #f59e0b;
      --shadow: 0 24px 70px rgba(0,0,0,.42);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 18% -10%, rgba(226,45,63,.22), transparent 32%),
        linear-gradient(180deg, #090c12 0%, var(--bg) 42%, #030407 100%);
      color: var(--text);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 16px 16px 12px;
      background: linear-gradient(180deg, rgba(9,12,18,.98), rgba(8,10,15,.92));
      border-bottom: 1px solid rgba(226,45,63,.24);
      backdrop-filter: blur(18px);
    }
    h1 { margin: 3px 0 0; font-size: 23px; line-height: 1.05; letter-spacing: 0; }
    main { display: grid; gap: 12px; padding: 12px 10px 84px; max-width: 980px; margin: 0 auto; }
    section {
      background: linear-gradient(180deg, rgba(17,24,33,.98), rgba(10,15,22,.98));
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      box-shadow: var(--shadow);
    }
    h2 { margin: 0 0 10px; font-size: 13px; color: #b7c4d6; text-transform: uppercase; letter-spacing: .04em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }
    .metric { padding: 11px; background: rgba(10,15,22,.92); border: 1px solid var(--line); border-radius: 8px; }
    .label { font-size: 12px; color: var(--muted); }
    .value { margin-top: 4px; font-size: 18px; font-weight: 700; }
    .ok { color: #63e6b1; } .review { color: #ffc857; } .bad { color: #ff7180; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 6px; border-bottom: 1px solid #25303a; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; }
    button {
      border: 1px solid #334155;
      background: linear-gradient(180deg, #202a36, #151d27);
      color: var(--text);
      border-radius: 8px;
      padding: 9px 11px;
      font-weight: 750;
      min-height: 36px;
    }
    button:active { transform: translateY(1px); }
    .primary { background: linear-gradient(180deg, #f7f9ff, #dce6f5); color: #0c1118; border-color: #f7f9ff; box-shadow: 0 10px 24px rgba(255,255,255,.08); }
    .btn-approve { background: linear-gradient(180deg, #19c6a7, #0f8d7f); border-color: #25d3b2; color: #03110f; }
    .btn-wait { background: linear-gradient(180deg, #3b82f6, #225ec4); border-color: #60a5fa; color: #f8fbff; }
    .btn-reject { background: linear-gradient(180deg, #ef4054, #ad1f31); border-color: #ff5d70; color: #fff7f8; }
    .btn-freeze { background: linear-gradient(180deg, #f5a623, #ad6b08); border-color: #ffc55c; color: #1a1002; }
    .btn-note { background: #1d2733; color: #d7e1ee; }
    button + button { margin-left: 6px; }
    .muted { color: var(--muted); }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
    .pill { border: 1px solid #344250; border-radius: 999px; padding: 4px 8px; color: #c7d3e2; font-size: 11px; background: rgba(255,255,255,.02); }
    .pill.hot { border-color: rgba(226,45,63,.55); color: #ffd1d7; background: rgba(226,45,63,.08); }
    input { width: min(100%, 420px); box-sizing: border-box; background: #090e15; color: var(--text); border: 1px solid #344250; border-radius: 8px; padding: 10px; }
    .topline { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .subline { margin-top: 6px; color: #91a4b8; font-size: 13px; }
    .page { display: none; }
    .page.active { display: grid; gap: 12px; }
    .bottom-nav { position: fixed; left: 0; right: 0; bottom: 0; z-index: 3; display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 8px 10px calc(8px + env(safe-area-inset-bottom)); background: rgba(7,9,13,.96); border-top: 1px solid rgba(226,45,63,.24); backdrop-filter: blur(18px); }
    .bottom-nav button { padding: 10px 4px; font-size: 12px; }
    .bottom-nav button.active { background: linear-gradient(180deg, #f7f9ff, #dce6f5); color: #10161c; border-color: #f7f9ff; }
    .trade-card { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; padding: 10px 0; border-bottom: 1px solid #25303a; }
    .trade-card:last-child { border-bottom: 0; }
    .card-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .card-actions button { margin-left: 0; }
    .evidence { margin-top: 9px; padding: 9px; background: #080d13; border: 1px solid #26313b; border-radius: 8px; color: #b8c7d6; font-size: 12px; line-height: 1.45; }
    .decision-card { position: relative; overflow: hidden; border-color: rgba(226,45,63,.5); background: linear-gradient(180deg, #151d27, #0b1118); }
    .decision-card:before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: linear-gradient(180deg, var(--red-2), var(--red)); }
    .decision-card h3 { margin: 0 0 8px; font-size: 18px; line-height: 1.12; }
    .decision-meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
    .decision-meta .pill { background: #0d1319; }
    .section-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 8px; }
    .mini-list { display: grid; gap: 8px; }
    .mini-item { padding: 10px; background: #090e15; border: 1px solid #26313b; border-radius: 8px; }
    .summary-strip { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
    .summary-strip .metric strong { font-size: 20px; }
    .decision-question { margin: 8px 0; padding: 9px 10px; border: 1px solid var(--line-hot); border-radius: 8px; background: rgba(226,45,63,.08); color: #ffd5da; font-weight: 800; }
    .card-copy { margin-top: 8px; line-height: 1.42; color: #e8eef7; }
    #toast { position: fixed; left: 12px; right: 12px; bottom: 66px; z-index: 4; display: none; padding: 12px; border-radius: 10px; background: #f7f9ff; color: #10161c; font-weight: 750; box-shadow: 0 18px 45px rgba(0,0,0,.35); }
    #toast.bad { background: #ffb4b4; }
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <div class="muted">Red Bot Operator</div>
        <h1 id="headline">Loading cockpit</h1>
      </div>
      <button class="primary" onclick="loadAll()">Refresh</button>
    </div>
    <div class="subline" id="subline">Mobiele cockpit | beslissen met bewijs</div>
  </header>
  <main>
    <div id="page-cockpit" class="page active">
    <section>
      <h2>Cockpit</h2>
      <div id="metrics" class="grid"></div>
    </section>
    <section>
      <h2>Next</h2>
      <div id="next"></div>
    </section>
    </div>
    <div id="page-recommendations" class="page">
    <section>
      <h2>Recommendations</h2>
      <div class="muted" style="margin-bottom:8px">V1-acties worden append-only opgeslagen en hebben geen live effect.</div>
      <input id="token" placeholder="Optional OPERATOR_APP_TOKEN for write actions" />
      <div id="recommendations"></div>
    </section>
    </div>
    <div id="page-positions" class="page">
    <section>
      <h2>Positions</h2>
      <div id="positions"></div>
    </section>
    <section>
      <h2>Recent Trades</h2>
      <div id="trades"></div>
    </section>
    </div>
    <div id="page-safety" class="page">
    <section>
      <h2>Safety</h2>
      <div id="safety"></div>
    </section>
    </div>
  </main>
  <nav class="bottom-nav">
    <button id="nav-cockpit" class="active" onclick="showTab('cockpit')">Cockpit</button>
    <button id="nav-recommendations" onclick="showTab('recommendations')">Review</button>
    <button id="nav-positions" onclick="showTab('positions')">Trades</button>
    <button id="nav-safety" onclick="showTab('safety')">Safety</button>
  </nav>
  <div id="toast"></div>
  <script>
    let recommendationItems = [];
    const fmt = (v) => v === undefined || v === null ? "-" : String(v);
    const cls = (v) => /OK|WATCH|NO ACTION/i.test(String(v)) ? "ok" : /STOP|ACTION|ERROR/i.test(String(v)) ? "bad" : "review";
    async function getJson(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(path + " failed: " + res.status);
      return await res.json();
    }
    function metric(label, value) {
      return `<div class="metric"><div class="label">${label}</div><div class="value ${cls(value)}">${fmt(value)}</div></div>`;
    }
    function showTab(name) {
      for (const page of document.querySelectorAll(".page")) page.classList.remove("active");
      for (const nav of document.querySelectorAll(".bottom-nav button")) nav.classList.remove("active");
      document.getElementById("page-" + name).classList.add("active");
      document.getElementById("nav-" + name).classList.add("active");
    }
    function list(items) {
      if (!items || !items.length) return '<div class="muted">Geen data.</div>';
      return items.map((item) => `<div class="trade-card"><strong>${item[0]}</strong><span>${item[1]}</span><span class="muted">${item[2] || ""}</span><span class="muted">${item[3] || ""}</span></div>`).join("");
    }
    function toast(message, bad = false) {
      const el = document.getElementById("toast");
      el.textContent = message;
      el.className = bad ? "bad" : "";
      el.style.display = "block";
      window.setTimeout(() => { el.style.display = "none"; }, 2600);
    }
    async function decideByIndex(index, action) {
      const item = recommendationItems[index];
      if (!item) {
        toast("Recommendation not found", true);
        return;
      }
      await decide(item, action);
    }
    async function decide(item, action) {
      const token = document.getElementById("token").value.trim();
      const payload = {
        source_id: String(item.id || item.source_id || item.key || item.title || "unknown"),
        source_type: item.source_type || item.type || "recommendation",
        action,
        operator: "operator_app",
        reason: "mobile fallback action",
        scope: item.scope || "recommendation",
        source_path: item.source_path || "analysis/recommendations/latest_recommendation_aggregator.json",
        expires_utc: ""
      };
      const res = await fetch("/api/decisions", {
        method: "POST",
        headers: { "content-type": "application/json", ...(token ? { "x-operator-token": token } : {}) },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        toast("Action failed: " + res.status, true);
        return;
      }
      toast(action.toUpperCase() + " opgeslagen. Pipeline verwerkt dit bij de volgende run.");
      await loadAll();
    }
    function actionText(item, action) {
      const level = item.effect_level || "shadow_only";
      if (action === "approve" && level === "context_live") return "Akkoord context";
      if (action === "approve" && level === "risk_down_live") return "Akkoord live-gate";
      if (action === "approve" && level === "strategy_live") return "Akkoord strict gate";
      if (action === "approve") return "Akkoord";
      if (action === "wait") return "Wacht op bewijs";
      if (action === "reject") return "Afwijzen";
      if (action === "freeze") return "Parkeren";
      if (action === "snooze") return "Later";
      if (action === "note") return "Notitie";
      return action;
    }
    function actionClass(action) {
      if (action === "approve") return "btn-approve";
      if (action === "wait") return "btn-wait";
      if (action === "reject") return "btn-reject";
      if (action === "freeze" || action === "snooze") return "btn-freeze";
      return "btn-note";
    }
    function decisionQuestion(item) {
      const level = item.effect_level || "";
      if (level === "context_live") return "Mag deze context automatisch mee blijven wegen in GPT/profielen?";
      if (level === "shadow_only") return "Blijft dit als shadow-test doorlopen tot er sterker bewijs is?";
      if (level === "risk_down_live") return "Mag dit door naar de live-gate voor risico omlaag?";
      if (level === "strategy_live") return "Mag dit door naar de strikte gate voor strategie/exit tuning?";
      if (level === "risk_up_live") return "Risk-up blijft geblokkeerd tot aparte goedkeuring.";
      return "Welke vervolgstap wil je voor deze aanbeveling?";
    }
    function levelLabel(level) {
      if (level === "context_live") return "Context automatisch";
      if (level === "shadow_only") return "Shadow test";
      if (level === "risk_down_live") return "Risico omlaag live-gate";
      if (level === "strategy_live") return "Strategie live-gate";
      if (level === "risk_up_live") return "Risk-up geblokkeerd";
      return level || "Onbekend";
    }
    function statusLabel(status) {
      if (status === "needs_operator_review") return "Besluit nodig";
      if (status === "approved_pending_live_gate") return "Wacht op live-gate";
      if (status === "auto_accept_as_context") return "Autonoom verwerkt";
      if (status === "wait_more_evidence") return "Wacht op bewijs";
      if (status === "blocked") return "Geblokkeerd";
      return status || "Open";
    }
    function evidenceText(item) {
      const e = item.evidence || {};
      const parts = [];
      if (e.stable_symbols !== undefined) parts.push(`stable symbols ${e.stable_symbols}`);
      if (e.days !== undefined) parts.push(`days ${e.days}`);
      if (e.usable_rows !== undefined) parts.push(`rows ${e.usable_rows}`);
      if (e.ranked_features !== undefined) parts.push(`ranked ${e.ranked_features}`);
      if (e.positions !== undefined) parts.push(`positions ${e.positions}`);
      if (e.closed !== undefined) parts.push(`closed ${e.closed}`);
      if (e.verdict !== undefined) parts.push(`verdict ${e.verdict}`);
      return parts.length ? parts.join(" | ") : "Evidence details available in report.";
    }
    function nextStepText(item) {
      const steps = item.allowed_next_steps || [];
      if (!steps.length) return "";
      return `Next: ${steps.join(", ")}`;
    }
    function primaryActions(item) {
      const status = item.status || "";
      const level = item.effect_level || "";
      if (status === "blocked") return ["reject", "freeze", "note"];
      if (status === "wait_more_evidence") return ["wait", "freeze"];
      if (status === "auto_accept_as_context") return ["freeze", "note"];
      if (level === "context_live") return ["approve", "wait", "freeze"];
      if (level === "shadow_only") return ["wait", "freeze"];
      if (level === "risk_down_live" || level === "strategy_live") return ["approve", "reject", "wait", "freeze"];
      return (item.allowed_actions_v1 || ["wait", "freeze"]).filter((x) => x !== "snooze" && x !== "note");
    }
    function isDecisionItem(item) {
      const status = item.status || "";
      return status === "needs_operator_review" || status === "approved_pending_live_gate";
    }
    function groupItems(items) {
      const decisions = [];
      const autonomous = [];
      const waiting = [];
      const blocked = [];
      for (const item of items) {
        const status = item.status || "";
        const level = item.effect_level || "";
        if (status === "blocked") {
          blocked.push(item);
        } else if (status === "wait_more_evidence") {
          waiting.push(item);
        } else if (isDecisionItem(item)) {
          decisions.push(item);
        } else if (status === "auto_accept_as_context" || level === "context_live" || level === "shadow_only") {
          autonomous.push(item);
        } else {
          waiting.push(item);
        }
      }
      return {
        decisions,
        autonomous,
        waiting,
        blocked
      };
    }
    function decisionCard(item, index) {
      return `
        <article class="metric decision-card" style="margin-bottom:10px">
          <div class="section-title"><h3>${fmt(item.title || item.id)}</h3><span class="pill hot">${statusLabel(item.status)}</span></div>
          <div class="decision-meta"><span class="pill">${levelLabel(item.effect_level)}</span><span class="pill">${fmt(item.area || "")}</span></div>
          <div class="decision-question">${decisionQuestion(item)}</div>
          <div class="card-copy"><strong>Waarom nu:</strong> ${fmt(item.headline || "")}</div>
          <div class="card-copy"><strong>Onderbouwing:</strong> ${fmt(item.why || "")}</div>
          <div class="evidence">${evidenceText(item)}<br>${nextStepText(item)}</div>
          <div class="card-actions">${primaryActions(item).map((action) => `<button class="${actionClass(action)}" onclick="decideByIndex(${index}, '${action}')">${actionText(item, action)}</button>`).join("")}</div>
        </article>`;
    }
    function compactSection(title, items, emptyText) {
      if (!items.length) return `<section><div class="section-title"><h2>${title}</h2><span class="pill">0</span></div><div class="muted">${emptyText}</div></section>`;
      return `
        <section>
          <div class="section-title"><h2>${title}</h2><span class="pill">${items.length}</span></div>
          <div class="mini-list">
            ${items.slice(0, 8).map((item) => `<div class="mini-item"><strong>${fmt(item.title || item.id)}</strong><br><span class="muted">${levelLabel(item.effect_level)} | ${statusLabel(item.status)}</span></div>`).join("")}
          </div>
        </section>`;
    }
    function recommendationRows(data) {
      const items = data.items || data.recommendations || data.review_items || data.actions || [];
      recommendationItems = items;
      if (!items.length) return '<div class="muted">Geen aanbevelingen gevonden.</div>';
      const grouped = groupItems(items);
      const indexOf = (item) => items.indexOf(item);
      const strip = `
        <div class="summary-strip">
          ${metric("Beslissen", grouped.decisions.length)}
          ${metric("Autonoom", grouped.autonomous.length)}
          ${metric("Wacht", grouped.waiting.length)}
          ${metric("Geblokkeerd", grouped.blocked.length)}
        </div>`;
      const decisions = grouped.decisions.length
        ? `<section><div class="section-title"><h2>Nu beslissen</h2><span class="pill">${grouped.decisions.length}</span></div>${grouped.decisions.map((item) => decisionCard(item, indexOf(item))).join("")}</section>`
        : `<section><div class="section-title"><h2>Nu beslissen</h2><span class="pill">0</span></div><div class="muted">Geen directe operatorbeslissing nodig.</div></section>`;
      return strip + decisions
        + compactSection("Autonoom verwerkt", grouped.autonomous, "Context en shadow learning lopen autonoom.")
        + compactSection("Wacht op bewijs", grouped.waiting, "Geen wachtende items.")
        + compactSection("Geblokkeerd / parkeren", grouped.blocked, "Geen geblokkeerde items.");
    }
    async function loadAll() {
      try {
        const bundle = await getJson("/api/mobile?limit=40");
        const snapshot = bundle.snapshot || {};
        const cockpit = bundle.cockpit || {};
        const recs = bundle.recommendations || {};
        const positions = bundle.positions || {};
        const trades = bundle.trades || {};
        const safety = bundle.safety || {};
        const c = cockpit || {};
        const learning = c.learning || {};
        const risk = c.risk || {};
        const live = c.live_readiness || {};
        const s = c.safety || safety || {};
        document.getElementById("headline").textContent = c.daily_decision?.label || snapshot.summary?.daily_decision || c.status || "Red Bot";
        document.getElementById("subline").textContent = c.daily_decision?.reason || `Updated ${bundle.generated_utc || ""}`;
        document.getElementById("metrics").innerHTML = [
          metric("Daily", c.daily_decision?.label || snapshot.summary?.daily_decision || c.status),
          metric("Safety", `${s.status || "-"} kill=${!!s.kill_switch_active} meltdown=${!!s.meltdown_active}`),
          metric("Live readiness", `review=${live.ready_for_operator_review ?? "-"} eligible=${live.eligible_for_live_wiring ?? "-"}`),
          metric("Learning flow", `pending=${(c.recommendations || {}).pending_live_gate ?? "-"} suppressed=${(c.recommendations || {}).suppressed ?? "-"}`),
          metric("ML", `${learning.ml_status || "-"} rows=${learning.ml_rows || "-"}`),
          metric("GPT hold", `${learning.gpt_hold_rate_pct || "-"}%`),
          metric("Risk", `down=${risk.risk_down ?? "-"} guard=${risk.guard_verdict || "-"}`)
        ].join("");
        document.getElementById("next").innerHTML = (c.next_actions || []).slice(0, 5).map((x) => `<div class="metric" style="margin-bottom:8px">${fmt(x)}</div>`).join("") || '<div class="muted">Geen acties.</div>';
        document.getElementById("recommendations").innerHTML = recommendationRows(recs);
        const lifecycles = positions.lifecycles || positions.positions || positions.items || [];
        document.getElementById("positions").innerHTML = list(lifecycles.filter((p) => p.status !== "closed").slice(0, 10).map((p) => [p.symbol || p.position_id, p.status, `amount ${fmt(p.master_amount || p.amount)}`, `pnl ${fmt(p.realized_pnl_eur || p.pnl)}`]));
        const tradeRows = trades.rows || trades.trades || trades.items || [];
        document.getElementById("trades").innerHTML = tradeRows.length
          ? list(tradeRows.slice(0, 40).map((t) => [t.symbol || `trade ${t.id}`, `${t.side || ""} ${t.status || ""}`, t.datetime_utc || t.timestamp, `pnl ${fmt(t.pnl_eur)} ${t.exit_reason || ""}`]))
          : `<div class="metric bad">Geen trades zichtbaar. row_count=${fmt(trades.row_count)} warning=${fmt(trades.warning)}</div>`;
        document.getElementById("safety").innerHTML = [
          metric("Status", s.status),
          metric("Kill switch", String(!!s.kill_switch_active)),
          metric("Meltdown", String(!!s.meltdown_active)),
          metric("Live entries", String(!!s.live_entry_orders_allowed)),
          metric("Live enforcement", String(!!s.live_enforcement_allowed))
        ].join("");
      } catch (err) {
        document.getElementById("metrics").innerHTML = `<div class="metric bad">Error: ${err.message}</div>`;
      }
    }
    loadAll();
  </script>
</body>
</html>
"""


@app.get("/{path:path}", response_model=None)
def frontend(path: str):
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(FALLBACK_HTML)
