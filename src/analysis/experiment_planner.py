# ============================================================
# src/analysis/experiment_planner.py
# ============================================================
"""
Experiment planner.

Turns tracked recommendations and hypotheses into a controlled experiment
queue. This module does not change live trading behavior. It only prepares
shadow/prompt/risk experiments that can later be approved explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "experiments")
DEFAULT_LATEST_FILE = "latest_experiment_plan.json"
DEFAULT_REGISTRY_PATH = os.path.join("analysis", "recommendations", "recommendation_registry.json")
ACTIVE_RECOMMENDATION_STATUSES = {"proposed", "approved"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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


def _stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class ExperimentPlanner:
    def __init__(
        self,
        registry_path: str = DEFAULT_REGISTRY_PATH,
    ):
        self.registry_path = registry_path

    def build_plan(self) -> dict:
        registry = _load_json(self.registry_path, {"meta": {}, "items": {}})
        if registry.get("_error"):
            return {
                "meta": {
                    "created_utc": _utc_now(),
                    "registry_path": self.registry_path,
                    "status": "ERROR",
                },
                "error": registry.get("_error"),
                "experiments": [],
                "summary": self._summary([]),
            }

        experiments = []
        for item in (registry.get("items") or {}).values():
            if item.get("status") not in ACTIVE_RECOMMENDATION_STATUSES:
                continue
            experiments.extend(self._experiments_for_item(item))

        experiments.sort(
            key=lambda exp: (
                self._status_rank(exp.get("status")),
                self._priority_rank(exp.get("priority")),
                exp.get("evidence", {}).get("cf_avg_r", 0.0) or 0.0,
                exp.get("evidence", {}).get("seen_count", 0) or 0,
            ),
            reverse=True,
        )

        return {
            "meta": {
                "created_utc": _utc_now(),
                "registry_path": self.registry_path,
                "status": "OK",
            },
            "summary": self._summary(experiments),
            "experiments": experiments,
        }

    def _experiments_for_item(self, item: dict) -> list[dict]:
        hypotheses = list((item.get("hypotheses") or {}).values())
        if not hypotheses:
            return []

        experiments = []
        for hypothesis in hypotheses:
            experiments.append(self._experiment_for_hypothesis(item, hypothesis))
        return experiments

    def _experiment_for_hypothesis(self, item: dict, hypothesis: dict) -> dict:
        status = self._experiment_status(item, hypothesis)
        experiment_type = self._experiment_type(hypothesis)
        exp_id = _stable_id({
            "recommendation_id": item.get("id"),
            "hypothesis_id": hypothesis.get("id"),
            "experiment_type": experiment_type,
        })

        return {
            "id": exp_id,
            "status": status,
            "experiment_type": experiment_type,
            "priority": item.get("priority"),
            "requires_human_approval": True,
            "source": {
                "recommendation_id": item.get("id"),
                "recommendation_status": item.get("status"),
                "area": item.get("area"),
                "finding": item.get("finding"),
                "hypothesis_id": hypothesis.get("id"),
            },
            "proposal": self._proposal_text(hypothesis),
            "evidence": {
                "hypothesis_type": hypothesis.get("hypothesis_type"),
                "pattern": hypothesis.get("pattern"),
                "confidence": hypothesis.get("confidence"),
                "stability": hypothesis.get("stability"),
                "promotable": hypothesis.get("promotable"),
                "seen_count": hypothesis.get("seen_count"),
                "age_days": hypothesis.get("age_days"),
                "matches": hypothesis.get("matches"),
                "cf_avg_r": hypothesis.get("cf_avg_r"),
                "cf_positive_rate_pct": hypothesis.get("cf_positive_rate_pct"),
                "cf_loss_rate_pct": hypothesis.get("cf_loss_rate_pct"),
            },
            "guardrails": self._guardrails(hypothesis),
            "next_action": self._next_action(status),
        }

    def _experiment_status(self, item: dict, hypothesis: dict) -> str:
        if item.get("status") == "approved":
            return "approved_for_shadow"
        if hypothesis.get("promotable"):
            return "ready_for_approval"
        if hypothesis.get("stability") in {"repeat", "stable_same_day"}:
            return "waiting_for_more_days"
        return "waiting_for_more_data"

    def _experiment_type(self, hypothesis: dict) -> str:
        hypothesis_type = hypothesis.get("hypothesis_type")
        if hypothesis_type == "allow_or_relax_hold":
            return "shadow_relax_entry_rule"
        if hypothesis_type == "protect_or_block":
            return "shadow_protection_rule"
        return "shadow_observation"

    def _proposal_text(self, hypothesis: dict) -> str:
        pattern = hypothesis.get("pattern")
        hypothesis_type = hypothesis.get("hypothesis_type")
        if hypothesis_type == "allow_or_relax_hold":
            return f"Shadow-test whether this HOLD pattern can be relaxed: {pattern}"
        if hypothesis_type == "protect_or_block":
            return f"Keep or strengthen protection for risky pattern: {pattern}"
        return f"Continue observing pattern: {pattern}"

    def _guardrails(self, hypothesis: dict) -> list[str]:
        guardrails = list(hypothesis.get("guardrails") or [])
        defaults = [
            "shadow_only",
            "no_live_trade_behavior_change",
            "requires_human_approval_before_live_use",
        ]
        for item in defaults:
            if item not in guardrails:
                guardrails.append(item)
        return guardrails

    def _next_action(self, status: str) -> str:
        mapping = {
            "approved_for_shadow": "Run as shadow experiment and compare against future outcomes.",
            "ready_for_approval": "Human approval required before starting a formal shadow experiment.",
            "waiting_for_more_days": "Keep tracking until the signal repeats across multiple days.",
            "waiting_for_more_data": "Keep collecting structured labeled events.",
        }
        return mapping.get(status, "Review manually.")

    def _summary(self, experiments: list[dict]) -> dict:
        by_status = {}
        by_type = {}
        for experiment in experiments:
            by_status[experiment.get("status", "unknown")] = by_status.get(experiment.get("status", "unknown"), 0) + 1
            by_type[experiment.get("experiment_type", "unknown")] = by_type.get(experiment.get("experiment_type", "unknown"), 0) + 1

        return {
            "total": len(experiments),
            "by_status": by_status,
            "by_type": by_type,
            "ready_for_approval": by_status.get("ready_for_approval", 0),
            "approved_for_shadow": by_status.get("approved_for_shadow", 0),
        }

    def _priority_rank(self, priority: Any) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(str(priority), 0)

    def _status_rank(self, status: Any) -> int:
        return {
            "approved_for_shadow": 4,
            "ready_for_approval": 3,
            "waiting_for_more_days": 2,
            "waiting_for_more_data": 1,
        }.get(str(status), 0)


def run_experiment_planner(
    registry_path: str = DEFAULT_REGISTRY_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    report = ExperimentPlanner(registry_path=registry_path).build_plan()
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build experiment plan from recommendation registry.")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY_PATH, help="Recommendation registry path.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_experiment_planner(
        registry_path=args.registry,
        output_dir=args.output_dir,
    )
    result = {
        "status": report.get("meta", {}).get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
