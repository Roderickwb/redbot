# ============================================================
# src/analysis/loss_diagnosis_report.py
# ============================================================
"""Diagnose where the bot is losing or finding opportunity.

This is the first step of the autonomy loop:
observe outcomes -> diagnose loss/opportunity clusters -> propose testable
improvement candidates. It is read-only and has no live effect.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.analysis.ml_training_dataset import DEFAULT_LATEST_JSONL, DEFAULT_OUTPUT_DIR as ML_DATASET_OUTPUT_DIR


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "loss_diagnosis")
DEFAULT_LATEST_FILE = "latest_loss_diagnosis_report.json"
DEFAULT_DATASET_PATH = os.path.join(ML_DATASET_OUTPUT_DIR, DEFAULT_LATEST_JSONL)
DEFAULT_MIN_CLUSTER_ROWS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _load_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda row: _safe_int(row.get("timestamp")))
    return rows


def _nested(row: dict, *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _target_r(row: dict) -> float:
    return _safe_float(_nested(row, "targets", "cf_r"))


def _actual_pnl(row: dict) -> float:
    return _safe_float(_nested(row, "targets", "realized_pnl_eur"))


def _confidence_bucket(value: Any) -> str:
    confidence = _safe_float(value)
    if confidence <= 0:
        return "missing"
    low = int(confidence // 10) * 10
    high = low + 9
    return f"{low}-{high}"


def _score_bucket(value: Any) -> str:
    score = _safe_int(value)
    if score <= 0:
        return "missing"
    low = (score // 20) * 20
    high = min(100, low + 19)
    return f"{low}-{high}"


def _cluster_keys(row: dict) -> list[tuple[str, str]]:
    features = row.get("features") or {}
    scores = features.get("scores") or {}
    regime = features.get("market_regime") or {}
    c1 = features.get("chart_1h") or {}
    c4 = features.get("chart_4h") or {}
    profile = features.get("coin_profile") or {}
    return [
        ("symbol", str(row.get("symbol") or "missing")),
        ("direction", str(row.get("direction") or "missing")),
        ("gpt_action", str(row.get("gpt_action") or "missing")),
        ("gpt_confidence", _confidence_bucket(row.get("gpt_confidence"))),
        ("market_regime", str(regime.get("regime") or "missing")),
        ("risk_mode", str(regime.get("risk_mode") or "missing")),
        ("directional_bias", str(regime.get("directional_bias") or "missing")),
        ("coin_profile_bias", str(profile.get("bias") or "missing")),
        ("coin_profile_risk", _score_bucket(_safe_float(profile.get("risk_multiplier"), 1.0) * 100.0)),
        ("score_entry", _score_bucket(scores.get("entry"))),
        ("score_risk", _score_bucket(scores.get("risk"))),
        ("score_learning", _score_bucket(scores.get("learning"))),
        ("chart_1h.structure_label", str(c1.get("structure_label") or "missing")),
        ("chart_1h.chop_subtype", str(c1.get("chop_subtype") or "missing")),
        ("chart_1h.entry_timing", str(c1.get("entry_timing") or "missing")),
        ("chart_1h.last_candle_quality", str(c1.get("last_candle_quality") or "missing")),
        ("chart_4h.structure_label", str(c4.get("structure_label") or "missing")),
        ("chart_4h.chop_subtype", str(c4.get("chop_subtype") or "missing")),
        ("chart_4h.entry_timing", str(c4.get("entry_timing") or "missing")),
    ]


class LossDiagnosisReport:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path

    def build_report(self, limit: Optional[int] = None, min_cluster_rows: int = DEFAULT_MIN_CLUSTER_ROWS) -> dict:
        rows = _load_jsonl(self.dataset_path, limit=limit)
        usable = [row for row in rows if _nested(row, "targets", "cf_r") is not None]
        opened = [row for row in usable if _nested(row, "targets", "opened_trade")]
        clusters = self._clusters(opened, min_cluster_rows=min_cluster_rows)
        loss_clusters = sorted(
            [row for row in clusters if row["net_R"] < 0],
            key=lambda row: (row["net_R"], -row["count"]),
        )[:12]
        opportunity_clusters = sorted(
            [row for row in clusters if row["avg_R"] > 0 and row["count"] >= min_cluster_rows],
            key=lambda row: (-row["net_R"], -row["avg_R"], -row["count"]),
        )[:12]
        candidates = self._candidates(loss_clusters, opportunity_clusters)
        total_r = sum(_target_r(row) for row in opened)
        return {
            "created_utc": _utc_now(),
            "status": "OK" if opened else "NO_DATA",
            "meta": {
                "dataset_path": self.dataset_path,
                "loaded_rows": len(rows),
                "usable_rows": len(usable),
                "opened_rows": len(opened),
                "min_cluster_rows": min_cluster_rows,
                "read_only": True,
                "live_effect": False,
            },
            "summary": {
                "opened_rows": len(opened),
                "total_R": round(total_r, 6),
                "avg_R": round(total_r / len(opened), 6) if opened else 0.0,
                "loss_clusters": len(loss_clusters),
                "opportunity_clusters": len(opportunity_clusters),
                "candidate_count": len(candidates),
                "top_loss": loss_clusters[0] if loss_clusters else None,
                "top_opportunity": opportunity_clusters[0] if opportunity_clusters else None,
            },
            "top_loss_clusters": loss_clusters,
            "top_opportunity_clusters": opportunity_clusters,
            "improvement_candidates": candidates,
        }

    def _clusters(self, rows: list[dict], min_cluster_rows: int) -> list[dict]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            for key in _cluster_keys(row):
                if key[1] and key[1] != "missing":
                    grouped[key].append(row)

        result = []
        for (dimension, value), group_rows in grouped.items():
            if len(group_rows) < min_cluster_rows:
                continue
            net_r = sum(_target_r(row) for row in group_rows)
            pnl = sum(_actual_pnl(row) for row in group_rows)
            wins = sum(1 for row in group_rows if _target_r(row) > 0)
            losses = sum(1 for row in group_rows if _target_r(row) <= 0)
            result.append({
                "dimension": dimension,
                "value": value,
                "count": len(group_rows),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round((wins / len(group_rows)) * 100.0, 2),
                "net_R": round(net_r, 6),
                "avg_R": round(net_r / len(group_rows), 6),
                "realized_pnl_eur": round(pnl, 6),
            })
        return result

    def _candidates(self, losses: list[dict], opportunities: list[dict]) -> list[dict]:
        candidates = []
        for row in losses[:3]:
            candidates.append({
                "kind": "problem",
                "area": self._area_for_dimension(row["dimension"]),
                "title": f"Loss cluster: {row['dimension']}={row['value']}",
                "problem_or_opportunity": f"This cluster lost {row['net_R']} R over {row['count']} opened trades.",
                "proposed_change": self._proposal_for_loss(row),
                "test_plan": "Shadow/paper compare baseline versus candidate filter or weighting for the same cluster.",
                "evidence": row,
                "live_effect": False,
            })
        for row in opportunities[:3]:
            candidates.append({
                "kind": "opportunity",
                "area": self._area_for_dimension(row["dimension"]),
                "title": f"Opportunity cluster: {row['dimension']}={row['value']}",
                "problem_or_opportunity": f"This cluster earned {row['net_R']} R over {row['count']} opened trades.",
                "proposed_change": self._proposal_for_opportunity(row),
                "test_plan": "Shadow/paper compare whether favoring this cluster improves net R without overtrading.",
                "evidence": row,
                "live_effect": False,
            })
        return candidates

    @staticmethod
    def _area_for_dimension(dimension: str) -> str:
        if dimension.startswith("chart_") or dimension.startswith("score_") or dimension in {"gpt_confidence", "direction", "gpt_action"}:
            return "entry"
        if dimension in {"market_regime", "risk_mode", "directional_bias", "coin_profile_risk"}:
            return "risk"
        if dimension in {"symbol", "coin_profile_bias"}:
            return "coin_selection"
        return "entry"

    @staticmethod
    def _proposal_for_loss(row: dict) -> str:
        area = LossDiagnosisReport._area_for_dimension(row["dimension"])
        if area == "entry":
            return "Test a stricter entry filter or lower GPT confidence/score weight for this cluster."
        if area == "risk":
            return "Test reduced sizing or no-new-entry conditions for this regime/risk cluster."
        if area == "coin_selection":
            return "Test lower priority or reduced sizing for this coin/profile cluster."
        return "Test a conservative rule for this losing cluster."

    @staticmethod
    def _proposal_for_opportunity(row: dict) -> str:
        area = LossDiagnosisReport._area_for_dimension(row["dimension"])
        if area == "entry":
            return "Test whether this entry context should get higher priority or softer filters."
        if area == "risk":
            return "Test whether risk can remain normal in this favorable regime while other regimes stay capped."
        if area == "coin_selection":
            return "Test whether this coin/profile cluster deserves higher priority."
        return "Test whether this positive cluster should be favored."


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_loss_diagnosis_report(
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    min_cluster_rows: int = DEFAULT_MIN_CLUSTER_ROWS,
) -> dict:
    report = LossDiagnosisReport(dataset_path=dataset_path).build_report(
        limit=limit,
        min_cluster_rows=min_cluster_rows,
    )
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build loss/opportunity diagnosis from labeled strategy events.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-cluster-rows", type=int, default=DEFAULT_MIN_CLUSTER_ROWS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_loss_diagnosis_report(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        limit=args.limit,
        min_cluster_rows=args.min_cluster_rows,
    )
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
