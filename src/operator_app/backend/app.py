from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analysis.operator_decisions import record_operator_decision
from src.operator_app.backend.auth import require_operator_token
from src.operator_app.backend.data import REPORTS, recent_trades, report
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


@app.get("/{path:path}")
def frontend(path: str) -> FileResponse | dict:
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "status": "NO_FRONTEND_BUILD",
        "message": "Run npm install && npm run build in src/operator_app/frontend, or use /api/* endpoints.",
        "live_effect": False,
    }
