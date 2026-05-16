# ============================================================
# src/analysis/risk_advice_history.py
# ============================================================
"""
Persistent history for risk policy advice.

The risk policy report describes the latest read-only advice. This module keeps
a deduplicated daily history per symbol, so repeated smoke checks on the same
day do not look like stronger evidence. It never changes live sizing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "risk")
DEFAULT_HISTORY_FILE = "risk_advice_history.json"
DEFAULT_LATEST_FILE = "latest_risk_advice_history_report.json"
DEFAULT_RISK_POLICY = os.path.join("analysis", "risk", "latest_risk_policy_report.json")
STABLE_DAYS = 3


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


class RiskAdviceHistory:
    def __init__(
        self,
        risk_policy_path: str = DEFAULT_RISK_POLICY,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.risk_policy_path = risk_policy_path
        self.output_dir = output_dir
        self.history_path = os.path.join(output_dir, DEFAULT_HISTORY_FILE)

    def build_report(self) -> dict:
        now = _utc_now()
        today = _utc_day(now)
        risk_policy = _load_json(self.risk_policy_path, {"policies": []})
        history = self._load_history()
        observations = self._observations(risk_policy.get("policies", []) or [])
        self._merge_day(history, observations, today, now)
        history["meta"]["updated_utc"] = now
        history["meta"]["risk_policy_path"] = self.risk_policy_path
        write_json(self.history_path, history)

        summary = self._summary(history)
        return {
            "meta": {
                "created_utc": now,
                "risk_policy_path": self.risk_policy_path,
                "history_path": self.history_path,
                "read_only": True,
                "live_enforcement": False,
                "stable_days_required": STABLE_DAYS,
            },
            "summary": summary,
            "today": {
                "date_utc": today,
                "observed_symbols": len(observations),
                "data_driven_risk_down": sum(1 for item in observations if item.get("risk_down")),
                "market_context_only": sum(1 for item in observations if item.get("market_context_only")),
                "review_only": sum(1 for item in observations if item.get("review_only")),
                "risk_up": sum(1 for item in observations if item.get("risk_up")),
            },
            "symbols": self._symbol_rows(history)[:50],
            "snapshots": history.get("daily_snapshots", [])[-30:],
        }

    def _load_history(self) -> dict:
        history = _load_json(self.history_path, {})
        if not isinstance(history, dict) or history.get("_error"):
            history = {}
        history.setdefault("meta", {})
        history.setdefault("symbols", {})
        history.setdefault("daily_snapshots", [])
        return history

    def _observations(self, policies: list[dict]) -> list[dict]:
        rows = []
        for policy in policies:
            symbol = policy.get("symbol")
            if not symbol:
                continue
            advice = policy.get("advice") or {}
            rows.append({
                "symbol": str(symbol),
                "policy_mode": policy.get("policy_mode"),
                "risk_multiplier": _safe_float(policy.get("risk_multiplier"), 1.0),
                "long_multiplier": _safe_float(advice.get("long_multiplier"), 1.0),
                "short_multiplier": _safe_float(advice.get("short_multiplier"), 1.0),
                "risk_down": bool(advice.get("risk_down")),
                "risk_up": bool(advice.get("risk_up")),
                "review_only": bool(advice.get("review_only")),
                "market_context_only": bool(advice.get("market_context_only")),
                "data_reasons": list(advice.get("data_reasons") or []),
                "market_reasons": list(advice.get("market_reasons") or []),
                "review_reasons": list(advice.get("review_reasons") or []),
            })
        rows.sort(key=lambda item: item["symbol"])
        return rows

    def _merge_day(self, history: dict, observations: list[dict], today: str, now: str) -> None:
        symbols = history.setdefault("symbols", {})
        seen_today = set()
        for item in observations:
            symbol = item["symbol"]
            seen_today.add(symbol)
            row = symbols.setdefault(symbol, {
                "symbol": symbol,
                "first_seen_utc": now,
                "last_seen_utc": now,
                "days": {},
            })
            row["last_seen_utc"] = now
            row.setdefault("days", {})[today] = item
            row["last_advice"] = item

        snapshot = {
            "date_utc": today,
            "created_utc": now,
            "observed_symbols": len(observations),
            "data_driven_risk_down": sum(1 for item in observations if item.get("risk_down")),
            "market_context_only": sum(1 for item in observations if item.get("market_context_only")),
            "review_only": sum(1 for item in observations if item.get("review_only")),
            "risk_up": sum(1 for item in observations if item.get("risk_up")),
            "symbols": sorted(seen_today),
        }
        snapshots = history.setdefault("daily_snapshots", [])
        if snapshots and snapshots[-1].get("date_utc") == today:
            snapshots[-1] = snapshot
        else:
            snapshots.append(snapshot)
        del snapshots[:-90]

    def _summary(self, history: dict) -> dict:
        rows = self._symbol_rows(history)
        dates = {
            date
            for symbol in (history.get("symbols") or {}).values()
            for date in (symbol.get("days") or {}).keys()
        }
        data_down = [row for row in rows if row.get("current_risk_down")]
        market_only = [row for row in rows if row.get("current_market_context_only")]
        review_only = [row for row in rows if row.get("current_review_only")]
        risk_up = [row for row in rows if row.get("current_risk_up")]
        stable_data_down = [row for row in rows if _safe_int(row.get("data_down_days")) >= STABLE_DAYS]
        by_mode = Counter(row.get("current_policy_mode") or "unknown" for row in rows)
        return {
            "tracked_symbols": len(rows),
            "days_observed": len(dates),
            "stable_days_required": STABLE_DAYS,
            "data_driven_risk_down_symbols": len(data_down),
            "stable_data_down_symbols": len(stable_data_down),
            "market_context_only_symbols": len(market_only),
            "review_only_symbols": len(review_only),
            "risk_up_symbols": len(risk_up),
            "by_policy_mode": dict(by_mode),
            "verdict": self._verdict(len(dates), stable_data_down),
            "top_stable_data_down": stable_data_down[:10],
            "top_current_data_down": data_down[:10],
            "read_only": True,
            "live_enforcement": False,
        }

    def _symbol_rows(self, history: dict) -> list[dict]:
        rows = []
        for symbol, state in (history.get("symbols") or {}).items():
            days = state.get("days") or {}
            latest_date = max(days.keys()) if days else None
            latest = days.get(latest_date, {}) if latest_date else {}
            data_days = [day for day, item in days.items() if item.get("risk_down")]
            market_days = [day for day, item in days.items() if item.get("market_context_only")]
            review_days = [day for day, item in days.items() if item.get("review_only")]
            risk_up_days = [day for day, item in days.items() if item.get("risk_up")]
            rows.append({
                "symbol": symbol,
                "days_seen": len(days),
                "first_seen_utc": state.get("first_seen_utc"),
                "last_seen_utc": state.get("last_seen_utc"),
                "latest_date_utc": latest_date,
                "current_policy_mode": latest.get("policy_mode"),
                "current_risk_down": bool(latest.get("risk_down")),
                "current_market_context_only": bool(latest.get("market_context_only")),
                "current_review_only": bool(latest.get("review_only")),
                "current_risk_up": bool(latest.get("risk_up")),
                "current_long_multiplier": latest.get("long_multiplier"),
                "current_short_multiplier": latest.get("short_multiplier"),
                "data_down_days": len(data_days),
                "market_context_only_days": len(market_days),
                "review_only_days": len(review_days),
                "risk_up_days": len(risk_up_days),
                "data_reasons": latest.get("data_reasons", []),
                "market_reasons": latest.get("market_reasons", []),
                "review_reasons": latest.get("review_reasons", []),
            })
        rows.sort(
            key=lambda item: (
                _safe_int(item.get("data_down_days")),
                bool(item.get("current_risk_down")),
                item.get("symbol") or "",
            ),
            reverse=True,
        )
        return rows

    def _verdict(self, days_observed: int, stable_data_down: list[dict]) -> str:
        if days_observed < STABLE_DAYS:
            return "collect_more_days"
        if stable_data_down:
            return "stable_data_down_candidates"
        return "collect_more_evidence"


def run_risk_advice_history(
    risk_policy_path: str = DEFAULT_RISK_POLICY,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = RiskAdviceHistory(risk_policy_path=risk_policy_path, output_dir=output_dir).build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Accumulate risk policy advice without double-counting daily runs.")
    parser.add_argument("--risk-policy", type=str, default=DEFAULT_RISK_POLICY, help="Latest risk policy report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_risk_advice_history(
        risk_policy_path=args.risk_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "today": report.get("today", {}),
        "output_path": report.get("output_path"),
        "history_path": report.get("meta", {}).get("history_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
