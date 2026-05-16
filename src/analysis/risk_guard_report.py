# ============================================================
# src/analysis/risk_guard_report.py
# ============================================================
"""
Read-only risk guard report.

Replays labeled GPT open decisions and checks whether conservative guardrails
would have paused or capped trading during bad clusters. This module does not
change live trading behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_LATEST_FILE = "latest_risk_guard_report.json"

DEFAULT_DAILY_LOSS_LIMIT_R = -2.0
DEFAULT_WEEKLY_LOSS_LIMIT_R = -5.0
DEFAULT_LOSS_STREAK_LIMIT = 3
DEFAULT_MAX_DAILY_OPENS = 5
DEFAULT_MAX_DAILY_SYMBOL_OPENS = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


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


class RiskGuardReport:
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        daily_loss_limit_r: float = DEFAULT_DAILY_LOSS_LIMIT_R,
        weekly_loss_limit_r: float = DEFAULT_WEEKLY_LOSS_LIMIT_R,
        loss_streak_limit: int = DEFAULT_LOSS_STREAK_LIMIT,
        max_daily_opens: int = DEFAULT_MAX_DAILY_OPENS,
        max_daily_symbol_opens: int = DEFAULT_MAX_DAILY_SYMBOL_OPENS,
    ):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.daily_loss_limit_r = daily_loss_limit_r
        self.weekly_loss_limit_r = weekly_loss_limit_r
        self.loss_streak_limit = loss_streak_limit
        self.max_daily_opens = max_daily_opens
        self.max_daily_symbol_opens = max_daily_symbol_opens

    def build_report(self, limit: int = 5000) -> dict:
        trades = self._load_labeled_opens(limit=limit)
        guard_events = self._replay_guards(trades)
        summary = self._summary(trades, guard_events)
        return {
            "meta": {
                "created_utc": _utc_now(),
                "limit": limit,
                "read_only": True,
                "live_enforcement": False,
                "thresholds": {
                    "daily_loss_limit_r": self.daily_loss_limit_r,
                    "weekly_loss_limit_r": self.weekly_loss_limit_r,
                    "loss_streak_limit": self.loss_streak_limit,
                    "max_daily_opens": self.max_daily_opens,
                    "max_daily_symbol_opens": self.max_daily_symbol_opens,
                },
            },
            "summary": summary,
            "guard_events": guard_events[:100],
            "recent_trades": trades[-25:],
        }

    def _load_labeled_opens(self, limit: int) -> list[dict]:
        rows = self.db.execute_query(
            """
            SELECT id, timestamp, symbol, gpt_action, outcome_json
            FROM strategy_events
            WHERE event_type='gpt_decision'
              AND outcome_status='labeled'
              AND gpt_action IN ('OPEN_LONG', 'OPEN_SHORT')
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (int(limit),),
        )
        result = []
        for event_id, timestamp, symbol, action, outcome_json in rows or []:
            outcome = _parse_json(outcome_json)
            counterfactual = outcome.get("counterfactual_trade") or {}
            realized = outcome.get("realized_trade") or {}
            r_multiple = _safe_float(counterfactual.get("r_multiple"))
            result.append({
                "event_id": _safe_int(event_id),
                "timestamp": _safe_int(timestamp),
                "timestamp_utc": self._ts_to_utc(timestamp),
                "day": self._day_key(timestamp),
                "week": self._week_key(timestamp),
                "symbol": symbol,
                "direction": "long" if action == "OPEN_LONG" else "short",
                "gpt_action": action,
                "r_multiple": round(r_multiple, 6),
                "pnl_eur": round(_safe_float(realized.get("pnl_eur")), 6),
                "outcome_label": outcome.get("label"),
                "counterfactual_label": counterfactual.get("label"),
            })
        return result

    def _replay_guards(self, trades: list[dict]) -> list[dict]:
        guard_events = []
        day_r = defaultdict(float)
        week_r = defaultdict(float)
        day_opens = defaultdict(int)
        day_symbol_opens = defaultdict(int)
        loss_streak = 0

        for trade in trades:
            day = trade.get("day")
            week = trade.get("week")
            symbol = trade.get("symbol")
            r_multiple = _safe_float(trade.get("r_multiple"))

            if day_r[day] <= self.daily_loss_limit_r:
                guard_events.append(self._guard_event("daily_loss_limit", trade, day_r[day]))
            if week_r[week] <= self.weekly_loss_limit_r:
                guard_events.append(self._guard_event("weekly_drawdown_guard", trade, week_r[week]))
            if loss_streak >= self.loss_streak_limit:
                guard_events.append(self._guard_event("loss_streak_cooldown", trade, loss_streak))
            if day_opens[day] >= self.max_daily_opens:
                guard_events.append(self._guard_event("max_daily_opens", trade, day_opens[day]))
            if day_symbol_opens[(day, symbol)] >= self.max_daily_symbol_opens:
                guard_events.append(self._guard_event("max_daily_symbol_opens", trade, day_symbol_opens[(day, symbol)]))

            day_r[day] += r_multiple
            week_r[week] += r_multiple
            day_opens[day] += 1
            day_symbol_opens[(day, symbol)] += 1
            loss_streak = loss_streak + 1 if r_multiple < 0 else 0

        return guard_events

    def _guard_event(self, guard: str, trade: dict, trigger_value: Any) -> dict:
        r_multiple = _safe_float(trade.get("r_multiple"))
        saved_r = abs(r_multiple) if r_multiple < 0 else 0.0
        missed_r = r_multiple if r_multiple > 0 else 0.0
        return {
            "guard": guard,
            "shadow_action": "would_pause_or_block_new_trade",
            "event_id": trade.get("event_id"),
            "timestamp_utc": trade.get("timestamp_utc"),
            "day": trade.get("day"),
            "week": trade.get("week"),
            "symbol": trade.get("symbol"),
            "direction": trade.get("direction"),
            "r_multiple": r_multiple,
            "trigger_value": trigger_value,
            "estimated_saved_r": round(saved_r, 6),
            "estimated_missed_r": round(missed_r, 6),
            "outcome_label": trade.get("outcome_label"),
        }

    def _summary(self, trades: list[dict], guard_events: list[dict]) -> dict:
        total_r = sum(_safe_float(row.get("r_multiple")) for row in trades)
        saved_r = sum(_safe_float(row.get("estimated_saved_r")) for row in guard_events)
        missed_r = sum(_safe_float(row.get("estimated_missed_r")) for row in guard_events)
        by_guard = Counter(row.get("guard") for row in guard_events)
        days = {row.get("day") for row in trades if row.get("day")}
        weeks = {row.get("week") for row in trades if row.get("week")}
        guard_breakdown = self._guard_breakdown(guard_events)
        return {
            "loaded_open_trades": len(trades),
            "days_observed": len(days),
            "weeks_observed": len(weeks),
            "total_r": round(total_r, 6),
            "guard_triggers": len(guard_events),
            "by_guard": dict(by_guard),
            "estimated_saved_r": round(saved_r, 6),
            "estimated_missed_r": round(missed_r, 6),
            "estimated_net_saved_r": round(saved_r - missed_r, 6),
            "verdict": self._verdict(len(trades), len(guard_events), saved_r - missed_r, missed_r),
            "primary_issue": self._primary_issue(guard_breakdown),
            "guard_breakdown": guard_breakdown,
            "calibration_advice": self._calibration_advice(guard_breakdown),
            "top_saved_symbols": self._top_symbols(guard_events, "estimated_saved_r"),
            "top_missed_symbols": self._top_symbols(guard_events, "estimated_missed_r"),
        }

    def _verdict(self, trades: int, triggers: int, net_saved_r: float, missed_r: float) -> str:
        if trades < 20:
            return "collect_more_trade_history"
        if triggers == 0:
            return "no_guard_pressure"
        if net_saved_r >= 2.0 and missed_r <= net_saved_r:
            return "guards_look_helpful"
        if missed_r > max(1.0, net_saved_r):
            return "guards_too_strict"
        return "mixed_guard_impact"

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

    def _guard_breakdown(self, guard_events: list[dict]) -> list[dict]:
        buckets: dict[str, dict] = {}
        for row in guard_events:
            guard = row.get("guard") or "unknown"
            bucket = buckets.setdefault(guard, {
                "guard": guard,
                "triggers": 0,
                "estimated_saved_r": 0.0,
                "estimated_missed_r": 0.0,
                "estimated_net_saved_r": 0.0,
                "winner_triggers": 0,
                "loss_triggers": 0,
            })
            saved = _safe_float(row.get("estimated_saved_r"))
            missed = _safe_float(row.get("estimated_missed_r"))
            bucket["triggers"] += 1
            bucket["estimated_saved_r"] += saved
            bucket["estimated_missed_r"] += missed
            bucket["estimated_net_saved_r"] += saved - missed
            if missed > 0:
                bucket["winner_triggers"] += 1
            elif saved > 0:
                bucket["loss_triggers"] += 1

        result = []
        for bucket in buckets.values():
            triggers = _safe_int(bucket.get("triggers"))
            missed = _safe_float(bucket.get("estimated_missed_r"))
            saved = _safe_float(bucket.get("estimated_saved_r"))
            net = _safe_float(bucket.get("estimated_net_saved_r"))
            if triggers == 0:
                verdict = "no_pressure"
            elif missed > max(1.0, saved):
                verdict = "too_strict"
            elif net >= 1.0 and missed <= saved:
                verdict = "helpful"
            else:
                verdict = "mixed"
            result.append({
                "guard": bucket["guard"],
                "triggers": triggers,
                "winner_triggers": _safe_int(bucket.get("winner_triggers")),
                "loss_triggers": _safe_int(bucket.get("loss_triggers")),
                "estimated_saved_r": round(saved, 6),
                "estimated_missed_r": round(missed, 6),
                "estimated_net_saved_r": round(net, 6),
                "verdict": verdict,
            })
        result.sort(
            key=lambda item: (
                item.get("verdict") == "too_strict",
                item.get("estimated_missed_r", 0.0),
                item.get("triggers", 0),
            ),
            reverse=True,
        )
        return result

    def _primary_issue(self, guard_breakdown: list[dict]) -> Optional[dict]:
        for item in guard_breakdown:
            if item.get("verdict") == "too_strict":
                return item
        return guard_breakdown[0] if guard_breakdown else None

    def _calibration_advice(self, guard_breakdown: list[dict]) -> list[dict]:
        advice = []
        for item in guard_breakdown:
            guard = item.get("guard")
            verdict = item.get("verdict")
            if verdict == "too_strict":
                recommendation = self._too_strict_recommendation(guard)
            elif verdict == "helpful":
                recommendation = "Keep this guard candidate in shadow; evidence is helpful but still read-only."
            elif verdict == "mixed":
                recommendation = "Keep collecting evidence; do not wire this guard live yet."
            else:
                recommendation = "No action; this guard has no meaningful pressure."
            advice.append({
                "guard": guard,
                "verdict": verdict,
                "recommendation": recommendation,
                "triggers": item.get("triggers", 0),
                "estimated_net_saved_r": item.get("estimated_net_saved_r", 0.0),
            })
        return advice

    def _too_strict_recommendation(self, guard: Optional[str]) -> str:
        if guard == "max_daily_opens":
            return "Raise max_daily_opens in shadow or add regime/symbol filters before live wiring."
        if guard == "max_daily_symbol_opens":
            return "Raise max_daily_symbol_opens in shadow or require a recent loss before blocking same-symbol opens."
        if guard == "loss_streak_cooldown":
            return "Increase loss_streak_limit in shadow or require negative daily R before cooldown."
        if guard == "daily_loss_limit":
            return "Lower the daily loss trigger only after checking it did not block recovery trades."
        if guard == "weekly_drawdown_guard":
            return "Keep weekly drawdown as shadow-only until multi-week evidence is available."
        return "This guard is too strict in replay; tune threshold before any live wiring."

    def _ts_to_utc(self, timestamp_ms: Any) -> Optional[str]:
        ts = _safe_int(timestamp_ms)
        if not ts:
            return None
        return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _day_key(self, timestamp_ms: Any) -> Optional[str]:
        ts = _safe_int(timestamp_ms)
        if not ts:
            return None
        return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d")

    def _week_key(self, timestamp_ms: Any) -> Optional[str]:
        ts = _safe_int(timestamp_ms)
        if not ts:
            return None
        dt = datetime.fromtimestamp(ts / 1000, timezone.utc)
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"


def run_risk_guard_report(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int = 5000,
) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = RiskGuardReport(db=db).build_report(limit=limit)
    finally:
        db.close_connection()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only risk guard report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled open trades.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_guard_report(output_dir=args.output_dir, limit=args.limit)
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
