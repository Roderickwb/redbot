# ============================================================
# src/analysis/risk_bridge_history.py
# ============================================================
"""
Persistent history for risk bridge outcomes.

The risk bridge outcome evaluator describes the latest run. This module keeps a
deduplicated, multi-day history of adjusted open trades with labeled outcomes,
so repeated daily-analysis runs do not count the same event as new evidence.
It is read-only and never changes live sizing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_HISTORY_FILE = "risk_bridge_outcome_history.json"
DEFAULT_LATEST_FILE = "latest_risk_bridge_history_report.json"
DEFAULT_OUTCOME_REPORT = os.path.join("analysis", "risk", "latest_risk_bridge_outcome_report.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utc_day(value: Optional[str] = None) -> str:
    raw = value or _utc_now()
    return raw[:10]


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


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class RiskBridgeHistory:
    def __init__(
        self,
        outcome_path: str = DEFAULT_OUTCOME_REPORT,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.outcome_path = outcome_path
        self.output_dir = output_dir
        self.history_path = os.path.join(output_dir, DEFAULT_HISTORY_FILE)

    def build_report(self) -> dict:
        now = _utc_now()
        outcome = _load_json(self.outcome_path, {"evaluated_decisions": []})
        history = self._load_history()
        candidates = self._history_candidates(outcome.get("evaluated_decisions", []) or [])
        added = self._merge_events(history, candidates, now)
        self._append_snapshot(history, added, now)
        history["meta"]["updated_utc"] = now
        history["meta"]["outcome_path"] = self.outcome_path
        write_json(self.history_path, history)

        summary = self._summary(history)
        report = {
            "meta": {
                "created_utc": now,
                "outcome_path": self.outcome_path,
                "history_path": self.history_path,
                "read_only": True,
                "live_enforcement": False,
            },
            "summary": summary,
            "recent_added": added[:20],
            "recent_events": self._recent_events(history),
            "snapshots": history.get("daily_snapshots", [])[-14:],
        }
        return report

    def _load_history(self) -> dict:
        history = _load_json(self.history_path, {})
        if not isinstance(history, dict) or history.get("_error"):
            history = {}
        history.setdefault("meta", {})
        history.setdefault("events", {})
        history.setdefault("daily_snapshots", [])
        return history

    def _history_candidates(self, decisions: list[dict]) -> list[dict]:
        rows = []
        for item in decisions:
            if not item.get("opened_trade"):
                continue
            if item.get("risk_shadow_action") == "would_allow_full_size":
                continue
            if item.get("outcome_status") != "labeled":
                continue
            event_id = _safe_int(item.get("event_id"))
            if not event_id:
                continue
            rows.append({
                "event_id": event_id,
                "symbol": item.get("symbol"),
                "direction": item.get("direction"),
                "gpt_action": item.get("gpt_action"),
                "risk_shadow_action": item.get("risk_shadow_action"),
                "original_size_multiplier": _safe_float(item.get("original_size_multiplier")),
                "adjusted_size_multiplier": _safe_float(item.get("adjusted_size_multiplier")),
                "multiplier_delta": _safe_float(item.get("multiplier_delta")),
                "outcome_label": item.get("outcome_label"),
                "counterfactual_label": item.get("counterfactual_label"),
                "cf_r": _safe_float(item.get("cf_r")),
                "estimated_saved_r": _safe_float(item.get("estimated_saved_r")),
                "estimated_missed_r": _safe_float(item.get("estimated_missed_r")),
                "policy_mode": item.get("policy_mode"),
                "policy_reasons": item.get("policy_reasons", []),
            })
        return rows

    def _merge_events(self, history: dict, candidates: list[dict], now: str) -> list[dict]:
        events = history.setdefault("events", {})
        added = []
        for item in candidates:
            key = str(item["event_id"])
            existing = events.get(key)
            if existing:
                existing.update(item)
                existing["last_seen_utc"] = now
                existing["seen_count"] = _safe_int(existing.get("seen_count")) + 1
                continue
            row = dict(item)
            row["first_seen_utc"] = now
            row["last_seen_utc"] = now
            row["seen_count"] = 1
            events[key] = row
            added.append(row)
        return added

    def _append_snapshot(self, history: dict, added: list[dict], now: str) -> None:
        snapshots = history.setdefault("daily_snapshots", [])
        summary = self._summary(history)
        today = _utc_day(now)
        existing_added = 0
        if snapshots and snapshots[-1].get("date_utc") == today:
            existing_added = _safe_int(snapshots[-1].get("added_events"))
        snapshot = {
            "date_utc": today,
            "created_utc": now,
            "added_events": existing_added + len(added),
            "total_unique_events": summary.get("unique_adjusted_labeled_events", 0),
            "estimated_saved_r": summary.get("estimated_saved_r", 0.0),
            "estimated_missed_r": summary.get("estimated_missed_r", 0.0),
            "estimated_net_saved_r": summary.get("estimated_net_saved_r", 0.0),
            "verdict": summary.get("verdict"),
        }
        if snapshots and snapshots[-1].get("date_utc") == today:
            snapshots[-1] = snapshot
        else:
            snapshots.append(snapshot)
        del snapshots[:-60]

    def _summary(self, history: dict) -> dict:
        events = list((history.get("events") or {}).values())
        losses = [row for row in events if _safe_float(row.get("cf_r")) < 0]
        winners = [row for row in events if _safe_float(row.get("cf_r")) > 0]
        saved_r = sum(_safe_float(row.get("estimated_saved_r")) for row in events)
        missed_r = sum(_safe_float(row.get("estimated_missed_r")) for row in events)
        net_saved_r = saved_r - missed_r
        days = self._days_observed(events)
        by_mode = Counter(row.get("policy_mode") or "unknown" for row in events)
        by_action = Counter(row.get("risk_shadow_action") or "unknown" for row in events)
        return {
            "unique_adjusted_labeled_events": len(events),
            "days_observed": days,
            "adjusted_loss_trades": len(losses),
            "adjusted_winner_trades": len(winners),
            "adjusted_avg_cf_r": self._avg([_safe_float(row.get("cf_r")) for row in events]),
            "estimated_saved_r": round(saved_r, 6),
            "estimated_missed_r": round(missed_r, 6),
            "estimated_net_saved_r": round(net_saved_r, 6),
            "verdict": self._verdict(len(events), days, net_saved_r, missed_r),
            "by_policy_mode": dict(by_mode),
            "by_risk_shadow_action": dict(by_action),
            "top_saved_symbols": self._top_symbols(events, "estimated_saved_r"),
            "top_missed_symbols": self._top_symbols(events, "estimated_missed_r"),
        }

    def _days_observed(self, events: list[dict]) -> int:
        days = {
            _utc_day(row.get("first_seen_utc"))
            for row in events
            if row.get("first_seen_utc")
        }
        return len(days)

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

    def _recent_events(self, history: dict) -> list[dict]:
        rows = list((history.get("events") or {}).values())
        rows.sort(key=lambda item: (_safe_int(item.get("event_id")), item.get("last_seen_utc") or ""), reverse=True)
        return rows[:25]

    def _verdict(self, sample: int, days: int, net_saved_r: float, missed_r: float) -> str:
        if sample < 20 or days < 3:
            return "collect_more_evidence"
        if net_saved_r >= 3.0 and missed_r <= net_saved_r:
            return "stable_risk_down_helpful"
        if missed_r > max(1.0, net_saved_r):
            return "risk_down_too_strict"
        return "mixed_or_small_edge"


def run_risk_bridge_history(
    outcome_path: str = DEFAULT_OUTCOME_REPORT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = RiskBridgeHistory(outcome_path=outcome_path, output_dir=output_dir).build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Accumulate risk bridge outcomes without double-counting events.")
    parser.add_argument("--outcome-report", type=str, default=DEFAULT_OUTCOME_REPORT, help="Latest risk bridge outcome report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_bridge_history(
        outcome_path=args.outcome_report,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
        "history_path": report.get("meta", {}).get("history_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
