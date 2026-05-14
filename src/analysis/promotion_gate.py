# ============================================================
# src/analysis/promotion_gate.py
# ============================================================
"""
Promotion gate for experiments.

Turns shadow experiment measurements into promotion decisions. This is a
read-only safety layer: it does not approve, reject, or apply any trading rule.
It only states whether an experiment is blocked, waiting, or ready for human
review based on replay and forward-shadow evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "promotion_gate")
DEFAULT_LATEST_FILE = "latest_promotion_gate_report.json"
DEFAULT_SHADOW_RESULTS = os.path.join("analysis", "experiments", "latest_shadow_experiment_results.json")

MIN_FORWARD_MATCHES_RELAX = 25
MIN_REPLAY_MATCHES_RELAX = 20
MIN_FORWARD_MATCHES_PROTECTION = 25
MIN_REPLAY_MATCHES_PROTECTION = 25


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


class PromotionGate:
    def __init__(self, shadow_results_path: str = DEFAULT_SHADOW_RESULTS):
        self.shadow_results_path = shadow_results_path

    def build_report(self) -> dict:
        shadow = _load_json(self.shadow_results_path, {"results": [], "summary": {}})
        if shadow.get("_error"):
            decisions = [{
                "status": "blocked",
                "reason": "shadow_results_unreadable",
                "details": shadow.get("_error"),
            }]
        else:
            decisions = [
                self._decision_for_result(result)
                for result in (shadow.get("results") or [])
            ]

        return {
            "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.shadow_results_path,
            "summary": self._summary(decisions),
            "decisions": decisions,
        }

    def _decision_for_result(self, result: dict) -> dict:
        exp_type = result.get("experiment_type")
        replay = result.get("replay_results") or {}
        forward = result.get("forward_shadow_results") or {}
        verdict = result.get("verdict") or {}

        if exp_type == "shadow_relax_entry_rule":
            return self._relax_decision(result, replay, forward, verdict)
        if exp_type == "shadow_protection_rule":
            return self._protection_decision(result, replay, forward, verdict)
        return self._base_decision(
            result=result,
            status="waiting",
            reason="unsupported_experiment_type",
            next_action="Keep observing; no promotion path is defined for this experiment type.",
        )

    def _relax_decision(self, result: dict, replay: dict, forward: dict, verdict: dict) -> dict:
        replay_matches = _safe_int(replay.get("matches"))
        forward_matches = _safe_int(forward.get("matches"))
        replay_ok = (
            replay_matches >= MIN_REPLAY_MATCHES_RELAX
            and _safe_float(replay.get("cf_avg_r")) >= 0.35
            and _safe_float(replay.get("cf_positive_rate_pct")) >= 55.0
            and _safe_float(replay.get("cf_loss_rate_pct")) <= 35.0
        )
        forward_ok = (
            forward_matches >= MIN_FORWARD_MATCHES_RELAX
            and _safe_float(forward.get("cf_avg_r")) >= 0.35
            and _safe_float(forward.get("cf_positive_rate_pct")) >= 55.0
            and _safe_float(forward.get("cf_loss_rate_pct")) <= 35.0
        )
        forward_bad = (
            forward_matches >= 10
            and (
                _safe_float(forward.get("cf_avg_r")) <= 0.0
                or _safe_float(forward.get("cf_loss_rate_pct")) >= 45.0
            )
        )

        if forward_bad:
            return self._base_decision(
                result=result,
                status="blocked",
                reason="forward_shadow_rejects_relax",
                next_action="Do not approve; forward evidence says relaxing this pattern is unsafe.",
            )
        if replay_ok and forward_ok:
            return self._base_decision(
                result=result,
                status="ready_for_human_review",
                reason="replay_and_forward_support_relax",
                next_action="Human may review this as a controlled shadow/live experiment candidate.",
            )
        if replay_ok:
            return self._base_decision(
                result=result,
                status="waiting_for_forward",
                reason="replay_supports_relax_but_forward_sample_is_too_small",
                next_action="Keep collecting forward-shadow matches before approval.",
            )
        return self._base_decision(
            result=result,
            status="blocked",
            reason="replay_does_not_support_relax",
            next_action="Do not approve relaxation from this evidence.",
        )

    def _protection_decision(self, result: dict, replay: dict, forward: dict, verdict: dict) -> dict:
        replay_matches = _safe_int(replay.get("matches"))
        forward_matches = _safe_int(forward.get("matches"))
        replay_confirms = (
            replay_matches >= MIN_REPLAY_MATCHES_PROTECTION
            and _safe_float(replay.get("cf_avg_r")) <= -0.1
            and _safe_float(replay.get("cf_loss_rate_pct")) >= 45.0
        )
        forward_confirms = (
            forward_matches >= MIN_FORWARD_MATCHES_PROTECTION
            and _safe_float(forward.get("cf_avg_r")) <= -0.1
            and _safe_float(forward.get("cf_loss_rate_pct")) >= 45.0
        )
        forward_overprotects = (
            forward_matches >= 10
            and _safe_float(forward.get("cf_avg_r")) >= 0.5
            and _safe_float(forward.get("cf_positive_rate_pct")) >= 60.0
        )

        if forward_overprotects:
            return self._base_decision(
                result=result,
                status="needs_review",
                reason="forward_shadow_suggests_possible_overprotection",
                next_action="Review before strengthening this protection; it may be blocking good trades.",
            )
        if replay_confirms and forward_confirms:
            return self._base_decision(
                result=result,
                status="confirmed_protection",
                reason="replay_and_forward_confirm_protection",
                next_action="Keep this protection; avoid broad loosening that includes this pattern.",
            )
        if replay_confirms:
            return self._base_decision(
                result=result,
                status="waiting_for_forward",
                reason="replay_confirms_protection_but_forward_sample_is_too_small",
                next_action="Keep observing until forward-shadow sample is large enough.",
            )
        return self._base_decision(
            result=result,
            status="waiting",
            reason="protection_evidence_is_mixed_or_small",
            next_action="Keep collecting evidence; do not create a new live rule from this yet.",
        )

    def _base_decision(self, result: dict, status: str, reason: str, next_action: str) -> dict:
        replay = result.get("replay_results") or {}
        forward = result.get("forward_shadow_results") or {}
        return {
            "experiment_id": result.get("experiment_id"),
            "experiment_type": result.get("experiment_type"),
            "pattern": result.get("pattern"),
            "status": status,
            "reason": reason,
            "next_action": next_action,
            "source_status": result.get("status"),
            "source_verdict": result.get("verdict"),
            "replay": self._compact_metrics(replay),
            "forward": self._compact_metrics(forward),
        }

    def _compact_metrics(self, metrics: dict) -> dict:
        return {
            "matches": _safe_int(metrics.get("matches")),
            "cf_avg_r": _safe_float(metrics.get("cf_avg_r")),
            "cf_positive_rate_pct": _safe_float(metrics.get("cf_positive_rate_pct")),
            "cf_loss_rate_pct": _safe_float(metrics.get("cf_loss_rate_pct")),
            "cf_large_positive": _safe_int(metrics.get("cf_large_positive")),
        }

    def _summary(self, decisions: list[dict]) -> dict:
        by_status = {}
        by_type = {}
        for decision in decisions:
            by_status[decision.get("status", "unknown")] = by_status.get(decision.get("status", "unknown"), 0) + 1
            by_type[decision.get("experiment_type", "unknown")] = by_type.get(decision.get("experiment_type", "unknown"), 0) + 1
        return {
            "total": len(decisions),
            "by_status": by_status,
            "by_type": by_type,
            "ready_for_human_review": by_status.get("ready_for_human_review", 0),
            "confirmed_protection": by_status.get("confirmed_protection", 0),
            "blocked": by_status.get("blocked", 0),
            "waiting": by_status.get("waiting", 0) + by_status.get("waiting_for_forward", 0),
        }


def run_promotion_gate(
    shadow_results_path: str = DEFAULT_SHADOW_RESULTS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = PromotionGate(shadow_results_path=shadow_results_path).build_report()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build promotion gate report from shadow experiment results.")
    parser.add_argument("--shadow-results", type=str, default=DEFAULT_SHADOW_RESULTS, help="Shadow experiment results JSON.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_promotion_gate(
        shadow_results_path=args.shadow_results,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
