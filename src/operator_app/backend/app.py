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
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #0e1216; color: #eef3f8; }
    header { position: sticky; top: 0; z-index: 2; padding: 14px 16px 10px; background: rgba(18,24,30,.96); border-bottom: 1px solid #2a333c; backdrop-filter: blur(14px); }
    h1 { margin: 2px 0 0; font-size: 22px; letter-spacing: 0; }
    main { display: grid; gap: 12px; padding: 12px 12px 78px; max-width: 980px; margin: 0 auto; }
    section { background: #171e25; border: 1px solid #2a333c; border-radius: 8px; padding: 12px; box-shadow: 0 16px 40px rgba(0,0,0,.18); }
    h2 { margin: 0 0 10px; font-size: 15px; color: #9fb3c8; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }
    .metric { padding: 10px; background: #10161c; border: 1px solid #26313b; border-radius: 6px; }
    .label { font-size: 12px; color: #91a4b8; }
    .value { margin-top: 4px; font-size: 18px; font-weight: 700; }
    .ok { color: #73e6a5; } .review { color: #ffd36a; } .bad { color: #ff8f8f; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 6px; border-bottom: 1px solid #25303a; text-align: left; vertical-align: top; }
    th { color: #91a4b8; font-weight: 600; }
    button { border: 1px solid #3b4855; background: #202a33; color: #eef3f8; border-radius: 6px; padding: 8px 10px; font-weight: 650; }
    button:active { transform: translateY(1px); }
    .primary { background: #e9f1ff; color: #10161c; border-color: #e9f1ff; }
    button + button { margin-left: 6px; }
    .muted { color: #91a4b8; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
    .pill { border: 1px solid #344250; border-radius: 999px; padding: 4px 8px; color: #b8c7d6; font-size: 12px; }
    input { width: min(100%, 420px); box-sizing: border-box; background: #10161c; color: #eef3f8; border: 1px solid #344250; border-radius: 6px; padding: 9px; }
    .topline { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .subline { margin-top: 6px; color: #91a4b8; font-size: 13px; }
    .page { display: none; }
    .page.active { display: grid; gap: 12px; }
    .bottom-nav { position: fixed; left: 0; right: 0; bottom: 0; z-index: 3; display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 8px 10px calc(8px + env(safe-area-inset-bottom)); background: rgba(18,24,30,.96); border-top: 1px solid #2a333c; backdrop-filter: blur(14px); }
    .bottom-nav button { padding: 10px 4px; font-size: 12px; }
    .bottom-nav button.active { background: #e9f1ff; color: #10161c; border-color: #e9f1ff; }
    .trade-card { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; padding: 10px 0; border-bottom: 1px solid #25303a; }
    .trade-card:last-child { border-bottom: 0; }
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
    <div class="subline" id="subline">Mobile control cockpit · append-only actions</div>
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
  <script>
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
      if (!res.ok) alert("Decision failed: " + res.status + " " + await res.text());
      await loadAll();
    }
    function recommendationRows(data) {
      const items = data.items || data.recommendations || data.review_items || data.actions || [];
      if (!items.length) return '<div class="muted">Geen aanbevelingen gevonden.</div>';
      return items.slice(0, 30).map((item) => `
        <div class="metric" style="margin-bottom:8px">
          <div class="row"><strong>${fmt(item.title || item.name || item.source_id || item.id || item.type)}</strong><span class="pill">${fmt(item.status || item.action || item.priority)}</span></div>
          <div class="muted" style="margin:7px 0">${fmt(item.reason || item.finding || item.summary || item.verdict)}</div>
          <button onclick='decide(${JSON.stringify(item).replaceAll("'", "&#39;")}, "approve")'>Approve</button>
          <button onclick='decide(${JSON.stringify(item).replaceAll("'", "&#39;")}, "reject")'>Reject</button>
          <button onclick='decide(${JSON.stringify(item).replaceAll("'", "&#39;")}, "wait")'>Wait</button>
          <button onclick='decide(${JSON.stringify(item).replaceAll("'", "&#39;")}, "freeze")'>Freeze</button>
        </div>`).join('');
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
          metric("ML", `${learning.ml_status || "-"} rows=${learning.ml_rows || "-"}`),
          metric("GPT hold", `${learning.gpt_hold_rate_pct || "-"}%`),
          metric("Risk", `down=${risk.risk_down ?? "-"} guard=${risk.guard_verdict || "-"}`)
        ].join("");
        document.getElementById("next").innerHTML = (c.next_actions || []).slice(0, 5).map((x) => `<div class="metric" style="margin-bottom:8px">${fmt(x)}</div>`).join("") || '<div class="muted">Geen acties.</div>';
        document.getElementById("recommendations").innerHTML = recommendationRows(recs);
        const lifecycles = positions.lifecycles || positions.positions || positions.items || [];
        document.getElementById("positions").innerHTML = list(lifecycles.filter((p) => p.status !== "closed").slice(0, 10).map((p) => [p.symbol || p.position_id, p.status, `amount ${fmt(p.master_amount || p.amount)}`, `pnl ${fmt(p.realized_pnl_eur || p.pnl)}`]));
        document.getElementById("trades").innerHTML = list((trades.rows || trades.trades || trades.items || []).slice(0, 40).map((t) => [t.symbol, `${t.side || ""} ${t.status || ""}`, t.datetime_utc || t.timestamp, `pnl ${fmt(t.pnl_eur)} ${t.exit_reason || ""}`]));
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
