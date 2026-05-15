# ============================================================
# src/analysis/risk_bridge_outcome_evaluator.py
# ============================================================
"""
Outcome evaluator for the risk strategy bridge.

Compares shadow risk sizing decisions with labeled outcomes. This is read-only:
it estimates whether risk-down would have saved or missed R, but it does not
change live sizing or trading rules.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_LATEST_FILE = "latest_risk_bridge_outcome_report.json"
DEFAULT_BRIDGE_REPORT = os.path.join("analysis", "risk", "latest_risk_strategy_bridge_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e), "_path": path}


def _parse_json(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class RiskBridgeOutcomeEvaluator:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        bridge_path: str = DEFAULT_BRIDGE_REPORT,
    ):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.bridge_path = bridge_path

    def build_report(self) -> dict:
        bridge = _load_json(self.bridge_path, {"decisions": []})
        decisions = bridge.get("decisions", []) or []
        event_ids = [
            _safe_int(item.get("event_id"))
            for item in decisions
            if _safe_int(item.get("event_id"))
        ]
        outcomes = self._load_outcomes(event_ids)
        evaluated = [
            self._evaluate_decision(item, outcomes.get(_safe_int(item.get("event_id")), {}))
            for item in decisions
        ]

        return {
            "meta": {
                "created_utc": _utc_now(),
                "bridge_path": self.bridge_path,
                "bridge_decisions": len(decisions),
                "loaded_outcomes": len(outcomes),
                "read_only": True,
                "live_enforcement": False,
            },
            "summary": self._summary(evaluated),
            "evaluated_decisions": evaluated[:200],
        }

    def _load_outcomes(self, event_ids: list[int]) -> dict[int, dict]:
        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.db.execute_query(
            f"""
            SELECT id, outcome_status, outcome_json
            FROM strategy_events
            WHERE id IN ({placeholders})
            """,
            tuple(event_ids),
        )
        result = {}
        for event_id, outcome_status, outcome_json in rows or []:
            result[_safe_int(event_id)] = {
                "outcome_status": outcome_status,
                "outcome": _parse_json(outcome_json),
            }
        return result

    def _evaluate_decision(self, decision: dict, outcome_row: dict) -> dict:
        outcome = outcome_row.get("outcome") or {}
        counterfactual = outcome.get("counterfactual_trade") or {}
        realized_trade = outcome.get("realized_trade") or {}
        cf_r = _safe_float(counterfactual.get("r_multiple"))
        original_mult = _safe_float(decision.get("original_size_multiplier"))
        adjusted_mult = _safe_float(decision.get("adjusted_size_multiplier"))
        multiplier_delta = max(0.0, original_mult - adjusted_mult)
        opened = bool(decision.get("opened_trade"))
        adjusted = opened and decision.get("risk_shadow_action") != "would_allow_full_size"

        saved_r = 0.0
        missed_r = 0.0
        if adjusted and cf_r < 0:
            saved_r = abs(cf_r) * multiplier_delta
        elif adjusted and cf_r > 0:
            missed_r = cf_r * multiplier_delta

        return {
            "event_id": decision.get("event_id"),
            "symbol": decision.get("symbol"),
            "direction": decision.get("direction"),
            "gpt_action": decision.get("gpt_action"),
            "opened_trade": opened,
            "risk_shadow_action": decision.get("risk_shadow_action"),
            "original_size_multiplier": original_mult,
            "adjusted_size_multiplier": adjusted_mult,
            "multiplier_delta": round(multiplier_delta, 4),
            "outcome_status": outcome_row.get("outcome_status"),
            "outcome_label": outcome.get("label"),
            "counterfactual_label": counterfactual.get("label"),
            "cf_r": round(cf_r, 6),
            "realized_pnl_eur": round(_safe_float(realized_trade.get("pnl_eur")), 6),
            "estimated_saved_r": round(saved_r, 6),
            "estimated_missed_r": round(missed_r, 6),
            "policy_mode": decision.get("policy_mode"),
            "policy_reasons": decision.get("policy_reasons", []),
        }

    def _summary(self, rows: list[dict]) -> dict:
        opened = [row for row in rows if row.get("opened_trade")]
        adjusted = [
            row for row in opened
            if row.get("risk_shadow_action") != "would_allow_full_size"
        ]
        adjusted_with_outcomes = [row for row in adjusted if row.get("outcome_status") == "labeled"]
        losses = [row for row in adjusted_with_outcomes if _safe_float(row.get("cf_r")) < 0]
        winners = [row for row in adjusted_with_outcomes if _safe_float(row.get("cf_r")) > 0]
        saved_r = sum(_safe_float(row.get("estimated_saved_r")) for row in adjusted_with_outcomes)
        missed_r = sum(_safe_float(row.get("estimated_missed_r")) for row in adjusted_with_outcomes)
        net_saved_r = saved_r - missed_r
        return {
            "evaluated_decisions": len(rows),
            "opened_trades": len(opened),
            "adjusted_open_trades": len(adjusted),
            "adjusted_with_labeled_outcomes": len(adjusted_with_outcomes),
            "adjusted_loss_trades": len(losses),
            "adjusted_winner_trades": len(winners),
            "adjusted_avg_cf_r": self._avg([_safe_float(row.get("cf_r")) for row in adjusted_with_outcomes]),
            "estimated_saved_r": round(saved_r, 6),
            "estimated_missed_r": round(missed_r, 6),
            "estimated_net_saved_r": round(net_saved_r, 6),
            "verdict": self._verdict(len(adjusted_with_outcomes), net_saved_r, missed_r),
            "by_risk_shadow_action": dict(Counter(row.get("risk_shadow_action") for row in rows)),
            "top_saved_symbols": self._top_symbols(adjusted_with_outcomes, "estimated_saved_r"),
            "top_missed_symbols": self._top_symbols(adjusted_with_outcomes, "estimated_missed_r"),
        }

    def _avg(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    def _top_symbols(self, rows: list[dict], metric: str) -> list[dict]:
        buckets: dict[str, dict] = {}
        for row in rows:
            symbol = row.get("symbol") or "UNKNOWN"
            bucket = buckets.setdefault(symbol, {"symbol": symbol, "events": 0, metric: 0.0})
            bucket["events"] += 1
            bucket[metric] += _safe_float(row.get(metric))
        result = [
            {"symbol": item["symbol"], "events": item["events"], metric: round(item[metric], 6)}
            for item in buckets.values()
        ]
        result.sort(key=lambda item: item.get(metric, 0.0), reverse=True)
        return result[:10]

    def _verdict(self, sample: int, net_saved_r: float, missed_r: float) -> str:
        if sample < 5:
            return "insufficient_labeled_outcomes"
        if net_saved_r > 1.0 and missed_r <= net_saved_r:
            return "risk_down_helpful"
        if missed_r > net_saved_r:
            return "risk_down_too_strict"
        return "mixed_or_small_edge"


def run_risk_bridge_outcome_evaluator(
    bridge_path: str = DEFAULT_BRIDGE_REPORT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = RiskBridgeOutcomeEvaluator(db=db, bridge_path=bridge_path).build_report()
    finally:
        db.close_connection()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate risk bridge sizing decisions against labeled outcomes.")
    parser.add_argument("--bridge", type=str, default=DEFAULT_BRIDGE_REPORT, help="Risk strategy bridge report JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_bridge_outcome_evaluator(
        bridge_path=args.bridge,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
