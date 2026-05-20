"""Track recommendation quality and stability over time.

This module keeps a daily history of bundled recommendation items from
recommendation_aggregator. It is read-only and has no live effect. The goal is
to give the future operator app memory: which recommendations are stable,
stale, changing, blocked repeatedly, or already acted on by the operator.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "recommendations")
DEFAULT_HISTORY_FILE = "recommendation_quality_history.json"
DEFAULT_LATEST_FILE = "latest_recommendation_quality_report.json"
DEFAULT_RECOMMENDATION_AGGREGATOR = os.path.join("analysis", "recommendations", "latest_recommendation_aggregator.json")
DEFAULT_OPERATOR_DECISIONS = os.path.join("analysis", "operator_decisions", "latest_operator_decisions.json")
STALE_DAYS = 5
STABLE_DAYS = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_day(value: Optional[str] = None) -> str:
    return (value or _utc_now())[:10]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
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


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class RecommendationQualityTracker:
    def __init__(
        self,
        recommendation_path: str = DEFAULT_RECOMMENDATION_AGGREGATOR,
        operator_decisions_path: str = DEFAULT_OPERATOR_DECISIONS,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.recommendation_path = recommendation_path
        self.operator_decisions_path = operator_decisions_path
        self.output_dir = output_dir
        self.history_path = os.path.join(output_dir, DEFAULT_HISTORY_FILE)

    def build_report(self) -> dict:
        now = _utc_now()
        today = _utc_day(now)
        recommendations = _load_json(self.recommendation_path, {"items": []})
        decisions = _load_json(self.operator_decisions_path, {})
        history = self._load_history()
        latest_decisions = self._latest_decisions(decisions)
        observations = self._observations(recommendations.get("items", []) or [], latest_decisions)
        self._merge_day(history, observations, today, now)
        history["meta"]["updated_utc"] = now
        history["meta"]["recommendation_path"] = self.recommendation_path
        history["meta"]["operator_decisions_path"] = self.operator_decisions_path
        _write_json(self.history_path, history)

        rows = self._quality_rows(history)
        summary = self._summary(history, rows, observations)
        report = {
            "created_utc": now,
            "status": summary.get("status"),
            "meta": {
                "recommendation_path": self.recommendation_path,
                "operator_decisions_path": self.operator_decisions_path,
                "history_path": self.history_path,
                "read_only": True,
                "live_effect": False,
                "stable_days": STABLE_DAYS,
                "stale_days": STALE_DAYS,
            },
            "summary": summary,
            "items": rows[:100],
            "today": {
                "date_utc": today,
                "observed_items": len(observations),
                "by_status": dict(Counter(item.get("status") or "unknown" for item in observations)),
            },
            "snapshots": history.get("daily_snapshots", [])[-30:],
        }
        output_path = os.path.join(self.output_dir, DEFAULT_LATEST_FILE)
        report["output_path"] = output_path
        _write_json(output_path, report)
        return report

    def _load_history(self) -> dict:
        history = _load_json(self.history_path, {})
        if not isinstance(history, dict) or history.get("_error"):
            history = {}
        history.setdefault("meta", {})
        history.setdefault("items", {})
        history.setdefault("daily_snapshots", [])
        return history

    def _latest_decisions(self, report: dict) -> dict[str, dict]:
        latest = {}
        for item in report.get("latest_by_source", []) or []:
            source_id = str(item.get("source_id") or "")
            if source_id:
                latest[source_id] = item
        return latest

    def _observations(self, items: list[dict], latest_decisions: dict[str, dict]) -> list[dict]:
        rows = []
        for item in items:
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            decision = latest_decisions.get(item_id, {})
            rows.append({
                "id": item_id,
                "area": item.get("area"),
                "candidate_type": item.get("candidate_type"),
                "subject": item.get("subject"),
                "status": item.get("status"),
                "title": item.get("title"),
                "headline": item.get("headline"),
                "default_action": item.get("default_action"),
                "operator_action": decision.get("action"),
                "operator_decision_id": decision.get("decision_id"),
                "operator_decision_utc": decision.get("created_utc"),
            })
        rows.sort(key=lambda row: (str(row.get("area")), str(row.get("candidate_type")), str(row.get("id"))))
        return rows

    def _merge_day(self, history: dict, observations: list[dict], today: str, now: str) -> None:
        state = history.setdefault("items", {})
        observed_ids = set()
        for obs in observations:
            item_id = obs["id"]
            observed_ids.add(item_id)
            row = state.setdefault(item_id, {
                "id": item_id,
                "first_seen_utc": now,
                "last_seen_utc": now,
                "days": {},
            })
            row["last_seen_utc"] = now
            row["area"] = obs.get("area")
            row["candidate_type"] = obs.get("candidate_type")
            row["subject"] = obs.get("subject")
            row["title"] = obs.get("title")
            row["headline"] = obs.get("headline")
            row["days"][today] = obs
            row["latest"] = obs

        snapshot = {
            "date_utc": today,
            "created_utc": now,
            "observed_items": len(observations),
            "by_status": dict(Counter(item.get("status") or "unknown" for item in observations)),
            "ids": sorted(observed_ids),
        }
        snapshots = history.setdefault("daily_snapshots", [])
        if snapshots and snapshots[-1].get("date_utc") == today:
            snapshots[-1] = snapshot
        else:
            snapshots.append(snapshot)
        del snapshots[:-120]

    def _quality_rows(self, history: dict) -> list[dict]:
        rows = []
        all_dates = sorted({
            day
            for item in (history.get("items") or {}).values()
            for day in (item.get("days") or {}).keys()
        })
        latest_day = all_dates[-1] if all_dates else None
        for item_id, state in (history.get("items") or {}).items():
            days = state.get("days") or {}
            ordered_days = sorted(days.keys())
            latest_date = ordered_days[-1] if ordered_days else None
            latest = days.get(latest_date, {}) if latest_date else {}
            statuses = [days[day].get("status") for day in ordered_days if days.get(day)]
            unique_statuses = sorted({str(status or "unknown") for status in statuses})
            current_streak = self._current_status_streak(ordered_days, days)
            missing_days = 0
            if latest_day and latest_date:
                missing_days = max(0, len([day for day in all_dates if latest_date < day <= latest_day]))
            operator_action = latest.get("operator_action")
            quality = self._quality_label(latest, unique_statuses, current_streak, missing_days, operator_action)
            rows.append({
                "id": item_id,
                "area": state.get("area"),
                "candidate_type": state.get("candidate_type"),
                "subject": state.get("subject"),
                "title": state.get("title"),
                "headline": state.get("headline"),
                "days_seen": len(ordered_days),
                "first_seen_utc": state.get("first_seen_utc"),
                "last_seen_utc": state.get("last_seen_utc"),
                "latest_date_utc": latest_date,
                "current_status": latest.get("status"),
                "current_status_streak_days": current_streak,
                "status_changes": max(0, len(unique_statuses) - 1),
                "unique_statuses": unique_statuses,
                "missing_days_since_seen": missing_days,
                "operator_action": operator_action,
                "operator_decision_id": latest.get("operator_decision_id"),
                "quality_label": quality,
                "live_effect": False,
            })
        rows.sort(key=lambda row: (
            {"needs_review": 0, "unstable": 1, "stale_unresolved": 2, "stable_unresolved": 3, "operator_handled": 4, "stable_context": 5}.get(str(row.get("quality_label")), 9),
            -_safe_int(row.get("days_seen")),
            str(row.get("area")),
        ))
        return rows

    @staticmethod
    def _current_status_streak(ordered_days: list[str], days: dict) -> int:
        if not ordered_days:
            return 0
        latest_status = days[ordered_days[-1]].get("status")
        streak = 0
        for day in reversed(ordered_days):
            if days[day].get("status") != latest_status:
                break
            streak += 1
        return streak

    @staticmethod
    def _quality_label(latest: dict, unique_statuses: list[str], streak: int, missing_days: int, operator_action: Optional[str]) -> str:
        if operator_action in {"approve", "reject", "freeze"}:
            return "operator_handled"
        if len(unique_statuses) >= 3:
            return "unstable"
        status = latest.get("status")
        if status == "needs_operator_review":
            return "needs_review"
        if missing_days >= STALE_DAYS:
            return "stale_unresolved"
        if streak >= STABLE_DAYS and status in {"wait_more_evidence", "blocked"}:
            return "stable_unresolved"
        if streak >= STABLE_DAYS and status == "auto_accept_as_context":
            return "stable_context"
        return "collecting"

    def _summary(self, history: dict, rows: list[dict], observations: list[dict]) -> dict:
        dates = {
            day
            for item in (history.get("items") or {}).values()
            for day in (item.get("days") or {}).keys()
        }
        by_quality = Counter(row.get("quality_label") or "unknown" for row in rows)
        by_status = Counter(row.get("current_status") or "unknown" for row in rows)
        by_area = Counter(row.get("area") or "unknown" for row in rows)
        needs_attention = by_quality.get("needs_review", 0) + by_quality.get("unstable", 0)
        status = "REVIEW" if needs_attention else "WATCH"
        return {
            "status": status,
            "tracked_items": len(rows),
            "observed_today": len(observations),
            "days_observed": len(dates),
            "needs_attention": needs_attention,
            "needs_review": by_quality.get("needs_review", 0),
            "unstable": by_quality.get("unstable", 0),
            "stale_unresolved": by_quality.get("stale_unresolved", 0),
            "stable_unresolved": by_quality.get("stable_unresolved", 0),
            "stable_context": by_quality.get("stable_context", 0),
            "operator_handled": by_quality.get("operator_handled", 0),
            "by_quality": dict(by_quality),
            "by_status": dict(by_status),
            "by_area": dict(by_area),
            "top_attention": rows[:5],
            "live_effect": False,
        }


def run_recommendation_quality_tracker(
    recommendation_path: str = DEFAULT_RECOMMENDATION_AGGREGATOR,
    operator_decisions_path: str = DEFAULT_OPERATOR_DECISIONS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    return RecommendationQualityTracker(
        recommendation_path=recommendation_path,
        operator_decisions_path=operator_decisions_path,
        output_dir=output_dir,
    ).build_report()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Track recommendation quality/stability over time.")
    parser.add_argument("--recommendation-path", type=str, default=DEFAULT_RECOMMENDATION_AGGREGATOR)
    parser.add_argument("--operator-decisions-path", type=str, default=DEFAULT_OPERATOR_DECISIONS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_recommendation_quality_tracker(
        recommendation_path=args.recommendation_path,
        operator_decisions_path=args.operator_decisions_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
