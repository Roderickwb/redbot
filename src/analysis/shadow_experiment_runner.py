# ============================================================
# src/analysis/shadow_experiment_runner.py
# ============================================================
"""
Shadow experiment runner.

Evaluates experiment-planner hypotheses against the ML training dataset.
It reports historical replay and recent forward-shadow metrics separately.

This module is read-only for trading behavior: it does not open, block or
modify live trades.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "experiments")
DEFAULT_LATEST_FILE = "latest_shadow_experiment_results.json"
DEFAULT_EXPERIMENT_PLAN = os.path.join("analysis", "experiments", "latest_experiment_plan.json")
DEFAULT_DATASET = os.path.join("analysis", "ml_training", "latest_strategy_event_dataset.jsonl")


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


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


class ShadowExperimentRunner:
    def __init__(
        self,
        experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
        dataset_path: str = DEFAULT_DATASET,
    ):
        self.experiment_plan_path = experiment_plan_path
        self.dataset_path = dataset_path

    def build_report(self, forward_hours: int = 24, include_waiting: bool = True) -> dict:
        plan = _load_json(self.experiment_plan_path, {"experiments": []})
        rows = _load_jsonl(self.dataset_path)
        now_ms = int(time.time() * 1000)
        forward_cutoff_ms = now_ms - int(forward_hours) * 3600 * 1000

        experiments = [
            exp for exp in (plan.get("experiments") or [])
            if include_waiting or exp.get("status") in {"ready_for_approval", "approved_for_shadow"}
        ]

        results = []
        for exp in experiments:
            replay_rows = self._matching_rows(exp, rows)
            forward_rows = [
                row for row in replay_rows
                if _safe_int(row.get("timestamp")) >= forward_cutoff_ms
            ]
            replay_results = self._metrics(replay_rows, exp)
            forward_results = self._metrics(forward_rows, exp)
            results.append({
                "experiment_id": exp.get("id"),
                "status": exp.get("status"),
                "experiment_type": exp.get("experiment_type"),
                "proposal": exp.get("proposal"),
                "pattern": (exp.get("evidence") or {}).get("pattern"),
                "source": exp.get("source"),
                "guardrails": exp.get("guardrails", []),
                "verdict": self._verdict(exp, replay_results, forward_results),
                "replay_signature": self._match_signature(replay_rows),
                "forward_signature": self._match_signature(forward_rows),
                "replay_results": replay_results,
                "forward_shadow_results": forward_results,
            })

        return {
            "meta": {
                "created_utc": _utc_now(),
                "experiment_plan_path": self.experiment_plan_path,
                "dataset_path": self.dataset_path,
                "loaded_experiments": len(experiments),
                "loaded_rows": len(rows),
                "forward_hours": forward_hours,
                "include_waiting": include_waiting,
            },
            "summary": self._summary(results),
            "overlap_groups": self._overlap_groups(results),
            "results": results,
        }

    def _matching_rows(self, experiment: dict, rows: list[dict]) -> list[dict]:
        pattern = ((experiment.get("evidence") or {}).get("pattern") or "").strip()
        if not pattern:
            return []
        return [
            row for row in rows
            if pattern in self._row_patterns(row)
        ]

    def _row_patterns(self, row: dict) -> set[str]:
        features = row.get("features") or {}
        regime = (features.get("market_regime") or {}).get("regime") or "missing"
        chart_1h = features.get("chart_1h") or {}
        profile = features.get("coin_profile") or {}
        direction = row.get("direction") or "missing"
        veto = row.get("primary_veto") or "missing"
        structure = chart_1h.get("structure_label") or "missing"
        chop_subtype = chart_1h.get("chop_subtype") or "missing"
        entry_timing = chart_1h.get("entry_timing") or "missing"
        pressure = self._pressure_bucket(row)
        risk_mult = _safe_float(profile.get("risk_multiplier"), 1.0)
        flags = set(profile.get("flags") or [])

        drawdown_flag = "drawdown_flag" if "DRAWDOWN_RISK" in flags else "no_drawdown_flag"
        cf_flag = "cf_negative_flag" if "COUNTERFACTUAL_EDGE_NEGATIVE" in flags else "no_cf_negative_flag"

        return {
            f"{direction}|{regime}|{veto}",
            f"{direction}|{regime}|{structure}|{chop_subtype}|{entry_timing}",
            f"{direction}|{regime}|{veto}|{pressure}",
            f"{direction}|{veto}|risk_mult_{risk_mult:.2f}|{drawdown_flag}|{cf_flag}",
        }

    def _pressure_bucket(self, row: dict) -> str:
        chart_1h = ((row.get("features") or {}).get("chart_1h") or {})
        direction = row.get("direction")
        if direction == "short":
            value = _safe_int(chart_1h.get("breakdown_pressure"))
        elif direction == "long":
            value = _safe_int(chart_1h.get("breakout_pressure"))
        else:
            value = max(
                _safe_int(chart_1h.get("breakdown_pressure")),
                _safe_int(chart_1h.get("breakout_pressure")),
            )
        if value >= 75:
            return "p75"
        if value >= 50:
            return "p50"
        if value >= 25:
            return "p25"
        return "p0"

    def _metrics(self, rows: list[dict], experiment: dict) -> dict:
        action = Counter(row.get("gpt_action") or "missing" for row in rows)
        outcome = Counter(row.get("outcome_label") or "missing" for row in rows)
        symbols = Counter(row.get("symbol") or "UNKNOWN" for row in rows)
        cf_values = [
            _safe_float((row.get("targets") or {}).get("cf_r"))
            for row in rows
        ]
        positive = [value for value in cf_values if value > 0]
        losses = [value for value in cf_values if value <= -0.5]
        large_positive = [value for value in cf_values if value >= 1.0]
        holds = sum(1 for row in rows if row.get("gpt_action") == "HOLD")
        opens = sum(1 for row in rows if row.get("gpt_action") in {"OPEN_LONG", "OPEN_SHORT"})
        simulated_action = self._simulated_action(experiment)

        return {
            "matches": len(rows),
            "simulated_action": simulated_action,
            "current_holds": holds,
            "current_opens": opens,
            "cf_avg_r": round(sum(cf_values) / len(cf_values), 4) if cf_values else 0.0,
            "cf_positive_rate_pct": round(100.0 * len(positive) / len(cf_values), 2) if cf_values else 0.0,
            "cf_loss_rate_pct": round(100.0 * len(losses) / len(cf_values), 2) if cf_values else 0.0,
            "cf_large_positive": len(large_positive),
            "by_action": dict(action),
            "by_outcome": dict(outcome),
            "top_symbols": [
                {"symbol": symbol, "events": count}
                for symbol, count in symbols.most_common(8)
            ],
            "sample_event_ids": [row.get("id") for row in rows[:10]],
        }

    def _match_signature(self, rows: list[dict]) -> str:
        ids = sorted(str(row.get("id")) for row in rows if row.get("id") is not None)
        if not ids:
            return "empty"
        raw = "|".join(ids)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _verdict(self, experiment: dict, replay: dict, forward: dict) -> dict:
        exp_type = experiment.get("experiment_type")
        replay_verdict = self._window_verdict(exp_type, replay, min_matches=10)
        forward_verdict = self._window_verdict(exp_type, forward, min_matches=10)
        if forward.get("matches", 0) >= 10:
            primary = forward_verdict
            basis = "forward_shadow"
        else:
            primary = self._replay_only_verdict(replay_verdict)
            basis = "replay_only"
        return {
            "primary": primary,
            "basis": basis,
            "replay": replay_verdict,
            "forward": forward_verdict,
        }

    def _window_verdict(self, exp_type: Any, metrics: dict, min_matches: int) -> str:
        matches = _safe_int(metrics.get("matches"))
        avg_r = _safe_float(metrics.get("cf_avg_r"))
        loss_rate = _safe_float(metrics.get("cf_loss_rate_pct"))
        positive_rate = _safe_float(metrics.get("cf_positive_rate_pct"))
        if matches < min_matches:
            return "insufficient_sample"
        if exp_type == "shadow_relax_entry_rule":
            if avg_r >= 0.5 and positive_rate >= 60.0 and loss_rate <= 25.0:
                return "promising_relax"
            if avg_r <= 0.0 or loss_rate >= 45.0:
                return "reject_relax"
            return "mixed_relax"
        if exp_type == "shadow_protection_rule":
            if avg_r <= -0.15 and loss_rate >= 50.0:
                return "protection_confirmed"
            if avg_r >= 0.5 and positive_rate >= 60.0:
                return "possibly_too_protective"
            return "mixed_protection"
        return "observed"

    def _replay_only_verdict(self, replay_verdict: str) -> str:
        mapping = {
            "promising_relax": "promising_replay_needs_forward",
            "protection_confirmed": "protective_replay_needs_forward",
            "possibly_too_protective": "possible_overprotection_needs_forward",
        }
        return mapping.get(replay_verdict, replay_verdict)

    def _simulated_action(self, experiment: dict) -> str:
        exp_type = experiment.get("experiment_type")
        if exp_type == "shadow_relax_entry_rule":
            return "would_allow_candidate"
        if exp_type == "shadow_protection_rule":
            return "would_block_or_keep_block"
        return "observe_only"

    def _summary(self, results: list[dict]) -> dict:
        by_status = Counter(result.get("status") or "unknown" for result in results)
        by_type = Counter(result.get("experiment_type") or "unknown" for result in results)
        by_verdict = Counter((result.get("verdict") or {}).get("primary") or "unknown" for result in results)
        replay_matches = sum(_safe_int((result.get("replay_results") or {}).get("matches")) for result in results)
        forward_matches = sum(_safe_int((result.get("forward_shadow_results") or {}).get("matches")) for result in results)
        return {
            "experiments": len(results),
            "by_status": dict(by_status),
            "by_type": dict(by_type),
            "by_verdict": dict(by_verdict),
            "replay_matches": replay_matches,
            "forward_matches": forward_matches,
        }

    def _overlap_groups(self, results: list[dict]) -> list[dict]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for result in results:
            key = (
                result.get("replay_signature") or "empty",
                result.get("forward_signature") or "empty",
            )
            grouped.setdefault(key, []).append(result)

        groups = []
        for (replay_sig, forward_sig), items in grouped.items():
            if len(items) <= 1:
                continue
            groups.append({
                "replay_signature": replay_sig,
                "forward_signature": forward_sig,
                "experiments": len(items),
                "experiment_ids": [item.get("experiment_id") for item in items],
                "patterns": [item.get("pattern") for item in items],
                "replay_matches": (items[0].get("replay_results") or {}).get("matches", 0),
                "forward_matches": (items[0].get("forward_shadow_results") or {}).get("matches", 0),
                "verdicts": [
                    (item.get("verdict") or {}).get("primary")
                    for item in items
                ],
                "interpretation": "same_event_set_do_not_count_independently",
            })
        groups.sort(key=lambda item: (item.get("experiments", 0), item.get("replay_matches", 0)), reverse=True)
        return groups


def run_shadow_experiment_runner(
    experiment_plan_path: str = DEFAULT_EXPERIMENT_PLAN,
    dataset_path: str = DEFAULT_DATASET,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    forward_hours: int = 24,
    include_waiting: bool = True,
) -> dict:
    report = ShadowExperimentRunner(
        experiment_plan_path=experiment_plan_path,
        dataset_path=dataset_path,
    ).build_report(
        forward_hours=forward_hours,
        include_waiting=include_waiting,
    )
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run shadow experiment replay/forward analysis.")
    parser.add_argument("--experiment-plan", type=str, default=DEFAULT_EXPERIMENT_PLAN)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forward-hours", type=int, default=24)
    parser.add_argument("--ready-only", action="store_true", help="Only evaluate ready/approved experiments.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_shadow_experiment_runner(
        experiment_plan_path=args.experiment_plan,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        forward_hours=args.forward_hours,
        include_waiting=not args.ready_only,
    )
    result = {
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
