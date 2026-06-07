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
MIN_LABELED_FOR_CONCLUSION = 10
MIN_AVG_DELTA_R = 0.10


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


def candidate_effect_r(kind: str, baseline_r: float, restriction: dict, features: dict) -> tuple[float, float]:
    """Return candidate R and delta versus the unrestricted baseline."""
    if kind == "pre_gpt_skip":
        candidate_r = 0.0
    elif kind == "sizing":
        multiplier = _safe_float(restriction.get("risk_multiplier"), 1.0)
        sizing = features.get("adaptive_restriction_sizing") or {}
        for applied in sizing.get("restrictions", []) or []:
            if str(applied.get("restriction_id")) != str(restriction.get("restriction_id")):
                continue
            before = _safe_float(applied.get("before"), 0.0)
            after = _safe_float(applied.get("after"), 0.0)
            if before > 0:
                multiplier = after / before
            break
        candidate_r = baseline_r * min(max(multiplier, 0.0), 1.0)
    else:
        candidate_r = baseline_r
    return candidate_r, candidate_r - baseline_r


def outcome_conclusion(labeled_count: int, avg_delta_r: float) -> str:
    if labeled_count < MIN_LABELED_FOR_CONCLUSION:
        return "COLLECT_MORE"
    if avg_delta_r >= MIN_AVG_DELTA_R:
        return "READY_FOR_PROMOTION_REVIEW"
    if avg_delta_r <= -MIN_AVG_DELTA_R:
        return "STOP_PAPER"
    return "INCONCLUSIVE"


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
        active_restrictions = [
            row for row in restriction_report.get("restrictions", []) or []
            if isinstance(row, dict) and row.get("restriction_id")
        ]
        suspended_restrictions = [
            row for row in restriction_report.get("suspended_restrictions", []) or []
            if isinstance(row, dict) and row.get("restriction_id")
        ]
        restrictions = active_restrictions + suspended_restrictions
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
        positive = [row for row in rows if row.get("conclusion") == "READY_FOR_PROMOTION_REVIEW"]
        harmful = [row for row in rows if row.get("conclusion") == "STOP_PAPER"]
        inconclusive = [row for row in rows if row.get("conclusion") == "INCONCLUSIVE"]

        summary = {
            "active_restrictions": len(active_restrictions),
            "suspended_restrictions": len(suspended_restrictions),
            "restrictions_with_events": len(applied),
            "restrictions_with_labeled_outcomes": len(labeled),
            "ready_for_review": len(ready),
            "collecting": len(collecting),
            "positive_conclusions": len(positive),
            "harmful_conclusions": len(harmful),
            "inconclusive_conclusions": len(inconclusive),
            "applied_events": sum(row.get("applied_events", 0) for row in rows),
            "labeled_events": sum(row.get("labeled_events", 0) for row in rows),
            "pending_events": sum(row.get("pending_events", 0) for row in rows),
            "pre_gpt_skips": sum(row.get("pre_gpt_skips", 0) for row in rows),
            "sizing_adjustments": sum(row.get("sizing_adjustments", 0) for row in rows),
            "baseline_r": round(sum(_safe_float(row.get("baseline_R")) for row in rows), 6),
            "candidate_r": round(sum(_safe_float(row.get("candidate_R")) for row in rows), 6),
            "delta_r": round(sum(_safe_float(row.get("delta_R")) for row in rows), 6),
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
        baseline_r = 0.0
        candidate_r = 0.0
        delta_r = 0.0
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
                event_candidate_r, event_delta_r = candidate_effect_r(kind, r_value, restriction, features)
                baseline_r += r_value
                candidate_r += event_candidate_r
                delta_r += event_delta_r
            else:
                pending += 1

        applied_count = len(matched)
        labeled_count = len(labeled)
        avg_delta_r = delta_r / labeled_count if labeled_count else 0.0
        conclusion = outcome_conclusion(labeled_count, avg_delta_r)
        prior_conclusion = restriction.get("outcome_conclusion") or {}
        prior_labeled = int(prior_conclusion.get("labeled_events") or 0)
        if (
            restriction.get("auto_suspended")
            and prior_conclusion.get("conclusion") == "STOP_PAPER"
            and prior_labeled > labeled_count
        ):
            # A stopped test creates no new events, so its original evidence
            # eventually falls outside the rolling event window. Preserve the
            # measured stop instead of silently reactivating the same rule.
            labeled_count = prior_labeled
            applied_count = max(applied_count, int(prior_conclusion.get("applied_events") or 0))
            baseline_r = _safe_float(prior_conclusion.get("baseline_R"))
            candidate_r = _safe_float(prior_conclusion.get("candidate_R"))
            delta_r = _safe_float(prior_conclusion.get("delta_R"))
            avg_delta_r = _safe_float(prior_conclusion.get("avg_delta_R"))
            conclusion = "STOP_PAPER"
        status = "NO_EVENTS"
        if applied_count:
            status = "COLLECTING"
        if conclusion in {"READY_FOR_PROMOTION_REVIEW", "STOP_PAPER", "INCONCLUSIVE"}:
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
            "auto_suspended": bool(restriction.get("auto_suspended")),
            "status": status,
            "conclusion": conclusion,
            "applied_events": applied_count,
            "labeled_events": labeled_count,
            "pending_events": pending,
            "pre_gpt_skips": pre_gpt_skips,
            "sizing_adjustments": sizing_adjustments,
            "baseline_R": round(baseline_r, 6),
            "candidate_R": round(candidate_r, 6),
            "delta_R": round(delta_r, 6),
            "avg_delta_R": round(avg_delta_r, 6),
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
