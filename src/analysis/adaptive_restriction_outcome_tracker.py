# ============================================================
# src/analysis/adaptive_restriction_outcome_tracker.py
# ============================================================
"""Measure whether approved adaptive restrictions actually affect paper flow.

This closes the first behavior-learning loop:
approved recommendation -> adaptive restriction -> strategy event -> outcome.

It is read-only. It does not create or enforce restrictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE


DEFAULT_RESTRICTIONS_PATH = os.path.join(
    "analysis",
    "adaptive_restrictions",
    "latest_adaptive_restrictions.json",
)
DEFAULT_OUTPUT_DIR = os.path.join("analysis", "adaptive_restrictions")
DEFAULT_LATEST_FILE = "latest_adaptive_restriction_outcomes.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
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


def _nested(data: dict, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _cf_r(outcome: dict) -> Optional[float]:
    value = _nested(outcome, "counterfactual_trade", "r_multiple")
    if value is None:
        return None
    return _safe_float(value)


def _restriction_ids_in_features(features: dict) -> set[str]:
    ids: set[str] = set()
    single = features.get("adaptive_restriction") or {}
    if isinstance(single, dict) and single.get("restriction_id"):
        ids.add(str(single.get("restriction_id")))

    sizing = features.get("adaptive_restriction_sizing") or {}
    if isinstance(sizing, dict):
        for restriction in sizing.get("restrictions", []) or []:
            if isinstance(restriction, dict) and restriction.get("restriction_id"):
                ids.add(str(restriction.get("restriction_id")))
    return ids


def _applied_kind(features: dict) -> str:
    if features.get("adaptive_restriction_sizing"):
        return "sizing"
    if features.get("adaptive_restriction_applied"):
        return "pre_gpt_skip"
    return "unknown"


class AdaptiveRestrictionOutcomeTracker:
    def __init__(
        self,
        db_path: str = DB_FILE,
        restrictions_path: str = DEFAULT_RESTRICTIONS_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.db_path = db_path
        self.restrictions_path = restrictions_path
        self.output_dir = output_dir

    def build(self, limit: int = 10000) -> dict:
        restriction_report = _load_json(self.restrictions_path, {})
        restrictions = [
            row for row in restriction_report.get("restrictions", []) or []
            if isinstance(row, dict) and row.get("restriction_id")
        ]
        events, event_error = self._load_events(limit=limit)
        rows = [
            self._restriction_row(restriction, events)
            for restriction in restrictions
        ]
        rows.sort(key=lambda row: (row.get("status"), row.get("restriction_id")))
        applied = [row for row in rows if row.get("applied_events", 0) > 0]
        labeled = [row for row in rows if row.get("labeled_events", 0) > 0]
        ready = [row for row in rows if row.get("status") == "READY_FOR_REVIEW"]
        collecting = [row for row in rows if row.get("status") == "COLLECTING"]

        summary = {
            "active_restrictions": len(restrictions),
            "restrictions_with_events": len(applied),
            "restrictions_with_labeled_outcomes": len(labeled),
            "ready_for_review": len(ready),
            "collecting": len(collecting),
            "applied_events": sum(row.get("applied_events", 0) for row in rows),
            "pre_gpt_skips": sum(row.get("pre_gpt_skips", 0) for row in rows),
            "sizing_adjustments": sum(row.get("sizing_adjustments", 0) for row in rows),
            "observed_r": round(sum(_safe_float(row.get("observed_R")) for row in rows), 6),
            "event_error": event_error,
            "live_effect": False,
        }
        payload = {
            "created_utc": _utc_now(),
            "status": "REVIEW" if ready else "WATCH",
            "summary": summary,
            "restrictions": rows,
            "source_path": self.restrictions_path,
            "output_path": os.path.join(self.output_dir, DEFAULT_LATEST_FILE),
            "live_effect": False,
        }
        _write_json(payload["output_path"], payload)
        return payload

    def _load_events(self, limit: int) -> tuple[list[dict], str]:
        if not os.path.exists(self.db_path):
            return [], f"db_not_found:{self.db_path}"
        sql = """
            SELECT id, timestamp, symbol, event_type, decision_stage, skip_reason,
                   gpt_action, trade_id, features_json, outcome_status, outcome_json
              FROM strategy_events
             WHERE features_json IS NOT NULL
             ORDER BY timestamp DESC
             LIMIT ?
        """
        try:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            try:
                rows = [dict(row) for row in con.execute(sql, (int(limit),)).fetchall()]
            finally:
                con.close()
            return rows, ""
        except Exception as e:
            return [], str(e)

    def _restriction_row(self, restriction: dict, events: list[dict]) -> dict:
        rid = str(restriction.get("restriction_id"))
        matched = []
        for event in events:
            raw_features = event.get("features_json")
            if rid not in str(raw_features):
                continue
            features = _parse_json(raw_features)
            if rid in _restriction_ids_in_features(features):
                matched.append((event, features))

        labeled = []
        pending = 0
        observed_r = 0.0
        pre_gpt_skips = 0
        sizing_adjustments = 0
        latest_ts = None
        for event, features in matched:
            latest_ts = max(latest_ts or 0, int(event.get("timestamp") or 0))
            kind = _applied_kind(features)
            if kind == "pre_gpt_skip":
                pre_gpt_skips += 1
            if kind == "sizing":
                sizing_adjustments += 1
            outcome = _parse_json(event.get("outcome_json"))
            r_value = _cf_r(outcome)
            if str(event.get("outcome_status") or "") == "labeled" and r_value is not None:
                labeled.append(event)
                observed_r += r_value
            else:
                pending += 1

        applied_count = len(matched)
        labeled_count = len(labeled)
        status = "NO_EVENTS"
        if applied_count:
            status = "COLLECTING"
        if labeled_count >= 10 or (applied_count >= 10 and pending == 0):
            status = "READY_FOR_REVIEW"

        evidence = restriction.get("evidence") or {}
        best = evidence.get("best_candidate") or evidence.get("best_coin_rule_candidate") or {}
        return {
            "restriction_id": rid,
            "source_item_id": restriction.get("source_item_id"),
            "scope": restriction.get("scope"),
            "symbol": restriction.get("symbol"),
            "state": restriction.get("state"),
            "rule_id": restriction.get("rule_id"),
            "status": status,
            "applied_events": applied_count,
            "labeled_events": labeled_count,
            "pending_events": pending,
            "pre_gpt_skips": pre_gpt_skips,
            "sizing_adjustments": sizing_adjustments,
            "observed_R": round(observed_r, 6),
            "avg_observed_R": round(observed_r / labeled_count, 6) if labeled_count else 0.0,
            "expected_net_R": best.get("estimated_net_R"),
            "candidate_baseline_R": best.get("baseline_R"),
            "candidate_after_R": best.get("candidate_R") or best.get("estimated_after_R"),
            "review_after": restriction.get("review_after"),
            "reopen_criteria": restriction.get("reopen_criteria") or [],
            "latest_event_ts": latest_ts,
            "paper_effect": True,
            "live_effect": False,
        }


def run_adaptive_restriction_outcome_tracker(
    restrictions_path: str = DEFAULT_RESTRICTIONS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int = 10000,
) -> dict:
    return AdaptiveRestrictionOutcomeTracker(
        restrictions_path=restrictions_path,
        output_dir=output_dir,
    ).build(limit=limit)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure adaptive paper restriction outcomes.")
    parser.add_argument("--restrictions-path", type=str, default=DEFAULT_RESTRICTIONS_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_adaptive_restriction_outcome_tracker(
        restrictions_path=args.restrictions_path,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
