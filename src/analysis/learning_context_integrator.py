# ============================================================
# src/analysis/learning_context_integrator.py
# ============================================================
"""
Integrate read-only learning reports into the live GPT coin-profile context.

This module is deliberately context-only:
- it does not alter live strategy/risk settings;
- it does not enable live enforcement;
- it only enriches coin_profiles.profile_json for strategy_name=trend_4h.

The goal is to prevent useful learning reports from becoming isolated
dashboards. Anything written here can be read by GPT through the existing
coin_profile path, while remaining non-authoritative evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


LIVE_STRATEGY_NAME = "trend_4h"
DEFAULT_OUTPUT_DIR = os.path.join("analysis", "learning_context")
DEFAULT_LATEST_FILE = "latest_learning_context_integrator.json"
DEFAULT_INDICATOR_EDGE_PATH = os.path.join("analysis", "indicator_edge", "latest_indicator_edge_report.json")
DEFAULT_ML_EDGE_PATH = os.path.join("analysis", "ml_models", "latest_edge_model_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _compact_feature(row: dict) -> dict:
    return {
        "feature": row.get("feature"),
        "type": row.get("type"),
        "sample_size": _safe_int(row.get("sample_size")),
        "edge_R": round(_safe_float(row.get("edge_r")), 4),
        "best_value": row.get("best_value"),
        "best_avg_R": row.get("best_avg_r"),
        "worst_value": row.get("worst_value"),
        "worst_avg_R": row.get("worst_avg_r"),
        "use_as": "context_only",
    }


def _top_features(items: list[dict], max_items: int = 5, min_abs_edge: float = 0.35) -> list[dict]:
    result = []
    for item in items or []:
        edge = abs(_safe_float(item.get("edge_r")))
        if edge < min_abs_edge:
            continue
        result.append(_compact_feature(item))
        if len(result) >= max_items:
            break
    return result


def _indicator_context_for_symbol(report: dict, symbol: str) -> dict:
    summary = report.get("summary", {}) or {}
    overall = report.get("overall", {}) or {}
    by_symbol = report.get("by_symbol", []) or []

    symbol_row = None
    for row in by_symbol:
        if row.get("symbol") == symbol:
            symbol_row = row
            break

    return {
        "status": report.get("status", "UNKNOWN"),
        "usable_rows": _safe_int(summary.get("usable_rows")),
        "ranked_features": _safe_int(summary.get("ranked_features")),
        "global_top_features": _top_features(overall.get("top_features", []), max_items=5),
        "symbol_top_features": _top_features((symbol_row or {}).get("top_features", []), max_items=5),
        "symbol_sample_size": _safe_int((symbol_row or {}).get("sample_size")),
        "live_effect": False,
    }


def _ml_context(report: dict) -> dict:
    readiness = report.get("readiness", {}) or {}
    model = report.get("model", {}) or {}
    return {
        "status": readiness.get("status") or report.get("status") or "UNKNOWN",
        "model_status": model.get("status"),
        "feature_version": model.get("feature_version"),
        "rows": _safe_int(readiness.get("rows")),
        "positive": _safe_int(readiness.get("positive")),
        "non_positive": _safe_int(readiness.get("non_positive")),
        "metrics": model.get("metrics", {}) or {},
        "prediction_summary": model.get("prediction_summary", {}) or {},
        "use_as": "shadow_context_only",
        "live_effect": False,
    }


def _load_live_profiles(db: DatabaseManager, strategy_name: str) -> list[dict]:
    db.cursor.execute(
        """
        SELECT symbol, source, updated_ts, profile_json
        FROM coin_profiles
        WHERE strategy_name=?
        ORDER BY symbol ASC
        """,
        (strategy_name,),
    )
    rows = []
    for symbol, source, updated_ts, raw in db.cursor.fetchall():
        try:
            profile = json.loads(raw or "{}")
        except Exception:
            profile = {}
        rows.append({
            "symbol": symbol,
            "source": source,
            "updated_ts": updated_ts,
            "profile": profile,
        })
    return rows


def run_learning_context_integrator(
    db: Optional[DatabaseManager] = None,
    strategy_name: str = LIVE_STRATEGY_NAME,
    indicator_edge_path: str = DEFAULT_INDICATOR_EDGE_PATH,
    ml_edge_path: str = DEFAULT_ML_EDGE_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    apply: bool = True,
) -> dict:
    local_db = db is None
    if db is None:
        db = DatabaseManager(db_path=DB_FILE)

    indicator_report = _load_json(indicator_edge_path)
    ml_report = _load_json(ml_edge_path)
    profiles = _load_live_profiles(db, strategy_name=strategy_name)

    updated = 0
    now = _utc_now()
    updated_ts = int(time.time() * 1000)
    ml_context = _ml_context(ml_report)

    try:
        for row in profiles:
            symbol = row["symbol"]
            profile = dict(row.get("profile") or {})
            existing_context = dict(profile.get("learned_context") or {})
            integrated = {
                "version": "learning_context_v1",
                "updated_at_utc": now,
                "strategy_name": strategy_name,
                "live_effect": False,
                "indicator_edges": _indicator_context_for_symbol(indicator_report, symbol),
                "ml_edge_model": ml_context,
                "notes": [
                    "Context-only evidence for GPT/operator review.",
                    "This does not alter sizing, entries, exits, guards or live enforcement.",
                ],
            }
            existing_context.update(integrated)
            profile["learned_context"] = existing_context

            if apply:
                db.upsert_coin_profile(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    profile=profile,
                    updated_ts=updated_ts,
                    source=row.get("source") or profile.get("source") or "learning_context_integrator",
                )
            updated += 1
    finally:
        if local_db:
            db.close_connection()

    summary = {
        "status": "OK",
        "strategy_name": strategy_name,
        "profiles_seen": len(profiles),
        "profiles_updated": updated if apply else 0,
        "apply": apply,
        "indicator_status": indicator_report.get("status"),
        "indicator_top_feature": ((indicator_report.get("summary") or {}).get("top_feature") or {}).get("feature"),
        "ml_status": ml_context.get("status"),
        "ml_model_status": ml_context.get("model_status"),
        "live_effect": False,
    }
    payload = {
        "created_utc": now,
        "summary": summary,
        "output_path": os.path.join(output_dir, DEFAULT_LATEST_FILE),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(payload["output_path"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Integrate learning reports into GPT coin-profile context.")
    parser.add_argument("--dry-run", action="store_true", help="Build report without updating coin_profiles.")
    parser.add_argument("--strategy-name", default=LIVE_STRATEGY_NAME)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_learning_context_integrator(strategy_name=args.strategy_name, apply=not args.dry_run)
    print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
