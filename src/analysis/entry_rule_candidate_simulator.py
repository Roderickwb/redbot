# ============================================================
# src/analysis/entry_rule_candidate_simulator.py
# ============================================================
"""Simulate concrete entry-rule candidates from loss diagnosis.

This is read-only. It turns a diagnosed losing entry cluster into candidate
rules and estimates baseline-vs-candidate R from labeled opened trades.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.analysis.loss_diagnosis_report import DEFAULT_DATASET_PATH, _load_jsonl, _nested, _safe_float, _safe_int


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "entry_rules")
DEFAULT_LATEST_FILE = "latest_entry_rule_candidate_simulator.json"
DEFAULT_LOSS_DIAGNOSIS = os.path.join("analysis", "loss_diagnosis", "latest_loss_diagnosis_report.json")
DEFAULT_MIN_CLUSTER_ROWS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _target_r(row: dict) -> float:
    return _safe_float(_nested(row, "targets", "cf_r"))


def _opened(row: dict) -> bool:
    return bool(_nested(row, "targets", "opened_trade"))


def _feature_value(row: dict, dimension: str) -> str:
    features = row.get("features") or {}
    if dimension == "symbol":
        return str(row.get("symbol") or "missing")
    if dimension == "direction":
        return str(row.get("direction") or "missing")
    if dimension == "gpt_action":
        return str(row.get("gpt_action") or "missing")
    if dimension == "market_regime":
        return str((features.get("market_regime") or {}).get("regime") or "missing")
    if dimension == "risk_mode":
        return str((features.get("market_regime") or {}).get("risk_mode") or "missing")
    if dimension == "directional_bias":
        return str((features.get("market_regime") or {}).get("directional_bias") or "missing")
    if dimension.startswith("chart_1h."):
        return str((features.get("chart_1h") or {}).get(dimension.split(".", 1)[1]) or "missing")
    if dimension.startswith("chart_4h."):
        return str((features.get("chart_4h") or {}).get(dimension.split(".", 1)[1]) or "missing")
    return "missing"


def _confidence(row: dict) -> float:
    return _safe_float(row.get("gpt_confidence"))


def _risk_mode(row: dict) -> str:
    return str(_nested(row, "features", "market_regime", "risk_mode") or "missing")


def _direction(row: dict) -> str:
    return str(row.get("direction") or "missing")


class EntryRuleCandidateSimulator:
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        loss_diagnosis_path: str = DEFAULT_LOSS_DIAGNOSIS,
    ):
        self.dataset_path = dataset_path
        self.loss_diagnosis_path = loss_diagnosis_path

    def build_report(self, limit: Optional[int] = None, min_cluster_rows: int = DEFAULT_MIN_CLUSTER_ROWS) -> dict:
        diagnosis = _load_json(self.loss_diagnosis_path)
        top_loss = ((diagnosis.get("summary") or {}).get("top_loss") or {})
        rows = [row for row in _load_jsonl(self.dataset_path, limit=limit) if _opened(row)]
        if not top_loss:
            return self._empty("NO_LOSS_CLUSTER", rows, top_loss, min_cluster_rows)

        dimension = str(top_loss.get("dimension") or "")
        value = str(top_loss.get("value") or "")
        cluster_rows = [row for row in rows if _feature_value(row, dimension) == value]
        if len(cluster_rows) < min_cluster_rows:
            return self._empty("LOW_SAMPLE", rows, top_loss, min_cluster_rows)

        candidates = self._simulate_candidates(cluster_rows)
        candidates.sort(key=self._candidate_sort_key, reverse=True)
        best = candidates[0] if candidates else None
        status = "REVIEW" if best and best.get("estimated_net_R", 0.0) > 0 else "WATCH"
        return {
            "created_utc": _utc_now(),
            "status": status,
            "meta": {
                "dataset_path": self.dataset_path,
                "loss_diagnosis_path": self.loss_diagnosis_path,
                "opened_rows": len(rows),
                "cluster_rows": len(cluster_rows),
                "min_cluster_rows": min_cluster_rows,
                "read_only": True,
                "live_effect": False,
            },
            "source_cluster": top_loss,
            "summary": {
                "dimension": dimension,
                "value": value,
                "cluster_rows": len(cluster_rows),
                "cluster_net_R": round(sum(_target_r(row) for row in cluster_rows), 6),
                "candidate_count": len(candidates),
                "best_rule": (best or {}).get("rule_id"),
                "best_estimated_net_R": (best or {}).get("estimated_net_R"),
                "recommendation": "approve_for_paper_test" if status == "REVIEW" else "collect_more_evidence",
            },
            "candidates": candidates,
            "best_candidate": best,
        }

    def _empty(self, status: str, rows: list[dict], top_loss: dict, min_cluster_rows: int) -> dict:
        return {
            "created_utc": _utc_now(),
            "status": status,
            "meta": {
                "opened_rows": len(rows),
                "min_cluster_rows": min_cluster_rows,
                "read_only": True,
                "live_effect": False,
            },
            "source_cluster": top_loss,
            "summary": {
                "candidate_count": 0,
                "recommendation": "collect_more_evidence",
            },
            "candidates": [],
            "best_candidate": None,
        }

    def _simulate_candidates(self, cluster_rows: list[dict]) -> list[dict]:
        return [
            self._risk_multiplier_rule(cluster_rows, 0.5),
            self._confidence_rule(cluster_rows, 70),
            self._confidence_rule(cluster_rows, 75),
            self._risk_off_block_rule(cluster_rows),
            self._long_only_block_rule(cluster_rows),
            self._cooldown_rule(cluster_rows),
        ]

    @staticmethod
    def _candidate_sort_key(item: dict) -> tuple[int, float, int]:
        action_type = str(item.get("action_type") or "")
        soft_priority = {
            "reduced_risk": 3,
            "strict_confirmation": 2,
            "conditional_cooldown": 1,
            "cooldown": 0,
        }.get(action_type, 0)
        return (
            soft_priority,
            _safe_float(item.get("estimated_net_R")),
            -_safe_int(item.get("affected_trades")),
        )

    def _cooldown_rule(self, rows: list[dict]) -> dict:
        return self._rule_summary(
            "cooldown_cluster",
            "Zet dit cluster tijdelijk op cooldown met heropencriteria",
            rows,
            blocked_rows=rows,
            action_type="cooldown",
        )

    def _risk_multiplier_rule(self, rows: list[dict], multiplier: float) -> dict:
        return self._rule_summary(
            f"risk_multiplier_{multiplier}",
            f"Verlaag risk naar {multiplier:.0%} voor dit cluster",
            rows,
            blocked_rows=[],
            adjusted_multiplier=multiplier,
            action_type="reduced_risk",
        )

    def _confidence_rule(self, rows: list[dict], threshold: int) -> dict:
        blocked = [row for row in rows if _confidence(row) < threshold]
        return self._rule_summary(
            f"require_confidence_{threshold}",
            f"Sta dit cluster alleen toe bij GPT confidence >= {threshold}",
            rows,
            blocked_rows=blocked,
            action_type="strict_confirmation",
        )

    def _risk_off_block_rule(self, rows: list[dict]) -> dict:
        blocked = [row for row in rows if _risk_mode(row) == "risk_off"]
        return self._rule_summary(
            "cooldown_cluster_in_risk_off",
            "Zet dit cluster alleen in risk_off tijdelijk op cooldown",
            rows,
            blocked_rows=blocked,
            action_type="conditional_cooldown",
        )

    def _long_only_block_rule(self, rows: list[dict]) -> dict:
        blocked = [row for row in rows if _direction(row) == "long"]
        return self._rule_summary(
            "cooldown_cluster_longs",
            "Zet long entries in dit cluster tijdelijk op cooldown",
            rows,
            blocked_rows=blocked,
            action_type="conditional_cooldown",
        )

    def _rule_summary(
        self,
        rule_id: str,
        title: str,
        rows: list[dict],
        blocked_rows: list[dict],
        adjusted_multiplier: Optional[float] = None,
        action_type: str = "cooldown",
    ) -> dict:
        baseline_r = sum(_target_r(row) for row in rows)
        if adjusted_multiplier is not None:
            candidate_r = sum(_target_r(row) * adjusted_multiplier for row in rows)
            affected = rows
        else:
            blocked_ids = {id(row) for row in blocked_rows}
            candidate_r = sum(_target_r(row) for row in rows if id(row) not in blocked_ids)
            affected = blocked_rows

        blocked_losers = sum(1 for row in affected if _target_r(row) <= 0)
        missed_winners = sum(1 for row in affected if _target_r(row) > 0)
        saved_loss_r = sum(abs(_target_r(row)) for row in affected if _target_r(row) < 0)
        missed_win_r = sum(_target_r(row) for row in affected if _target_r(row) > 0)
        if adjusted_multiplier is not None:
            saved_loss_r *= (1.0 - adjusted_multiplier)
            missed_win_r *= (1.0 - adjusted_multiplier)

        return {
            "rule_id": rule_id,
            "title": title,
            "action_type": action_type,
            "reopen_criteria": self._reopen_criteria(action_type),
            "review_after": "na 10 nieuwe paper/shadow signalen of 48 uur, wat eerder komt",
            "baseline_R": round(baseline_r, 6),
            "candidate_R": round(candidate_r, 6),
            "estimated_net_R": round(candidate_r - baseline_r, 6),
            "affected_trades": len(affected),
            "blocked_or_adjusted_losers": blocked_losers,
            "missed_or_adjusted_winners": missed_winners,
            "estimated_saved_loss_R": round(saved_loss_r, 6),
            "estimated_missed_win_R": round(missed_win_r, 6),
            "live_effect": False,
        }

    @staticmethod
    def _reopen_criteria(action_type: str) -> list[str]:
        base = [
            "cluster blijft elk uur geanalyseerd",
            "heropen als nieuwe paper/shadow signalen voor dit cluster positief worden",
            "heropen als marktregime of 1h/4h structuur duidelijk verbetert",
        ]
        if action_type == "reduced_risk":
            return base + ["normaliseer sizing pas na positieve baseline-vergelijking"]
        if action_type == "strict_confirmation":
            return base + ["versoepel bevestiging pas als confidence-filter geen goede entries mist"]
        return base + ["cooldown is tijdelijk en mag niet als permanente ban worden behandeld"]


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_entry_rule_candidate_simulator(
    dataset_path: str = DEFAULT_DATASET_PATH,
    loss_diagnosis_path: str = DEFAULT_LOSS_DIAGNOSIS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    min_cluster_rows: int = DEFAULT_MIN_CLUSTER_ROWS,
) -> dict:
    report = EntryRuleCandidateSimulator(
        dataset_path=dataset_path,
        loss_diagnosis_path=loss_diagnosis_path,
    ).build_report(limit=limit, min_cluster_rows=min_cluster_rows)
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate entry-rule candidates from loss diagnosis.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--loss-diagnosis-path", type=str, default=DEFAULT_LOSS_DIAGNOSIS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-cluster-rows", type=int, default=DEFAULT_MIN_CLUSTER_ROWS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_entry_rule_candidate_simulator(
        dataset_path=args.dataset_path,
        loss_diagnosis_path=args.loss_diagnosis_path,
        output_dir=args.output_dir,
        limit=args.limit,
        min_cluster_rows=args.min_cluster_rows,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "best_candidate": report.get("best_candidate"),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
