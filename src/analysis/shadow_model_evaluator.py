# ============================================================
# src/analysis/shadow_model_evaluator.py
# ============================================================
"""
Shadow evaluator for candidate rule changes.

This module tests hypothetical decision rules against the ML training dataset.
It never changes live trading behavior. Its job is to answer:
"If this rule had opened/allowed a trade, what did the later counterfactual say?"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Iterable, Optional

from src.analysis.ml_training_dataset import (
    DEFAULT_LATEST_JSONL as ML_DATASET_LATEST_JSONL,
    DEFAULT_OUTPUT_DIR as ML_DATASET_OUTPUT_DIR,
    _safe_float,
    _safe_int,
)


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "shadow_models")
DEFAULT_LATEST_FILE = "latest_shadow_model_report.json"
DEFAULT_DATASET_PATH = os.path.join(ML_DATASET_OUTPUT_DIR, ML_DATASET_LATEST_JSONL)


RuleFn = Callable[[dict], bool]


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
    return rows


def _chart(row: dict, timeframe: str = "1h") -> dict:
    return ((row.get("features") or {}).get(f"chart_{timeframe}") or {})


def _scores(row: dict) -> dict:
    return ((row.get("features") or {}).get("scores") or {})


def _profile(row: dict) -> dict:
    return ((row.get("features") or {}).get("coin_profile") or {})


def _regime(row: dict) -> dict:
    return ((row.get("features") or {}).get("market_regime") or {})


def _flags(row: dict) -> list[str]:
    flags = (_profile(row).get("flags") or [])
    return [str(flag).lower() for flag in flags]


def _has_flag(row: dict, text: str) -> bool:
    needle = text.lower()
    return any(needle in flag for flag in _flags(row))


def _cf_r(row: dict) -> float:
    return _safe_float((row.get("targets") or {}).get("cf_r"))


def _action(row: dict) -> str:
    return str(row.get("gpt_action") or "")


def _is_hold(row: dict) -> bool:
    return _action(row) == "HOLD"


def _entry(row: dict) -> int:
    return _safe_int(_scores(row).get("entry"))


def _risk(row: dict) -> int:
    return _safe_int(_scores(row).get("risk"))


def _learning(row: dict) -> int:
    return _safe_int(_scores(row).get("learning"))


def _risk_multiplier(row: dict) -> float:
    return _safe_float(_profile(row).get("risk_multiplier"), 1.0)


def _rule_short_breakdown_v1(row: dict) -> bool:
    c1 = _chart(row, "1h")
    return (
        _is_hold(row)
        and row.get("direction") == "short"
        and c1.get("breakdown_pressure", 0) >= 80
        and c1.get("chop_subtype") in ("bearish_continuation_chop", "not_chop", None)
        and _entry(row) >= 45
        and _risk(row) >= 35
    )


def _rule_short_breakdown_ignore_drawdown_if_clean(row: dict) -> bool:
    c1 = _chart(row, "1h")
    return (
        _is_hold(row)
        and row.get("direction") == "short"
        and _has_flag(row, "drawdown")
        and c1.get("breakdown_pressure", 0) >= 90
        and c1.get("entry_timing") in ("clean", "continuation", "noisy")
        and _entry(row) >= 50
        and _risk(row) >= 35
    )


def _rule_long_breakout_v1(row: dict) -> bool:
    c1 = _chart(row, "1h")
    return (
        _is_hold(row)
        and row.get("direction") == "long"
        and c1.get("breakout_pressure", 0) >= 80
        and c1.get("chop_subtype") in ("bullish_continuation_chop", "not_chop", None)
        and _entry(row) >= 50
        and _risk(row) >= 40
        and (_regime(row).get("regime") or "") != "risk_off"
    )


def _rule_relax_local_chop_when_pressure_high(row: dict) -> bool:
    c1 = _chart(row, "1h")
    pressure = max(
        _safe_int(c1.get("breakout_pressure")),
        _safe_int(c1.get("breakdown_pressure")),
        _safe_int(c1.get("continuation_pressure")),
    )
    return (
        _is_hold(row)
        and row.get("primary_veto") == "local_chop"
        and pressure >= 80
        and c1.get("chop_subtype") not in ("true_range_chop", "messy_chop", "late_extension_chop")
        and _entry(row) >= 45
    )


def _rule_block_low_quality_opens(row: dict) -> bool:
    # This rule evaluates whether historical OPENs with weak entry should have been blocked.
    return (
        _action(row) in ("OPEN_LONG", "OPEN_SHORT")
        and (_entry(row) < 55 or _risk(row) < 40)
    )


RULES: dict[str, RuleFn] = {
    "allow_short_breakdown_v1": _rule_short_breakdown_v1,
    "allow_short_breakdown_despite_drawdown_if_clean": _rule_short_breakdown_ignore_drawdown_if_clean,
    "allow_long_breakout_v1": _rule_long_breakout_v1,
    "relax_local_chop_when_pressure_high": _rule_relax_local_chop_when_pressure_high,
    "block_low_quality_opens": _rule_block_low_quality_opens,
}


class ShadowModelEvaluator:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path

    def build_report(self, limit: Optional[int] = None, min_matches: int = 3) -> dict:
        rows = _load_jsonl(self.dataset_path, limit=limit)
        evaluations = {
            name: self._evaluate_rule(name, fn, rows)
            for name, fn in RULES.items()
        }
        recommendations = self._recommendations(evaluations, min_matches=min_matches)
        return {
            "meta": {
                "dataset_path": self.dataset_path,
                "loaded_rows": len(rows),
                "limit": limit,
                "min_matches": min_matches,
            },
            "rules": evaluations,
            "recommendations": recommendations,
        }

    def _evaluate_rule(self, name: str, fn: RuleFn, rows: list[dict]) -> dict:
        matched = [row for row in rows if fn(row)]
        total_r = sum(_cf_r(row) for row in matched)
        positives = [row for row in matched if _cf_r(row) > 0]
        large_positive = [row for row in matched if _cf_r(row) >= 1.0]
        losses = [row for row in matched if _cf_r(row) <= -0.5]
        worst = min((_cf_r(row) for row in matched), default=0.0)
        best = max((_cf_r(row) for row in matched), default=0.0)

        return {
            "rule": name,
            "matches": len(matched),
            "cf_total_r": round(total_r, 4),
            "cf_avg_r": round(total_r / len(matched), 4) if matched else 0.0,
            "cf_positive": len(positives),
            "cf_large_positive": len(large_positive),
            "cf_loss": len(losses),
            "cf_positive_rate_pct": round(len(positives) / len(matched) * 100.0, 2) if matched else 0.0,
            "cf_loss_rate_pct": round(len(losses) / len(matched) * 100.0, 2) if matched else 0.0,
            "best_r": round(best, 4),
            "worst_r": round(worst, 4),
            "sample_cases": [self._sample_case(row) for row in matched[:20]],
        }

    def _sample_case(self, row: dict) -> dict:
        c1 = _chart(row, "1h")
        return {
            "id": row.get("id"),
            "symbol": row.get("symbol"),
            "direction": row.get("direction"),
            "action": row.get("gpt_action"),
            "primary_veto": row.get("primary_veto"),
            "regime": _regime(row).get("regime"),
            "structure": c1.get("structure_label"),
            "chop_subtype": c1.get("chop_subtype"),
            "entry_timing": c1.get("entry_timing"),
            "breakout_pressure": c1.get("breakout_pressure"),
            "breakdown_pressure": c1.get("breakdown_pressure"),
            "entry_score": _entry(row),
            "risk_score": _risk(row),
            "learning_score": _learning(row),
            "risk_multiplier": _risk_multiplier(row),
            "cf_r": _cf_r(row),
            "outcome_label": row.get("outcome_label"),
            "counterfactual_label": row.get("counterfactual_label"),
        }

    def _recommendations(self, evaluations: dict, min_matches: int) -> list[dict]:
        recs = []
        for name, result in evaluations.items():
            matches = _safe_int(result.get("matches"))
            avg_r = _safe_float(result.get("cf_avg_r"))
            loss_rate = _safe_float(result.get("cf_loss_rate_pct"))

            if matches < min_matches:
                verdict = "insufficient_sample"
                priority = "low"
                advice = "Keep collecting data before considering this rule."
            elif avg_r >= 0.35 and loss_rate <= 45:
                verdict = "promising_shadow_rule"
                priority = "medium"
                advice = "Candidate for deeper shadow testing; do not enable live yet."
            elif avg_r <= -0.15 or loss_rate >= 55:
                verdict = "reject_or_tighten"
                priority = "medium"
                advice = "This rule would likely worsen outcomes; keep current protection or tighten it."
            else:
                verdict = "mixed"
                priority = "low"
                advice = "Mixed result. Split by symbol/regime/chop subtype before acting."

            recs.append({
                "rule": name,
                "priority": priority,
                "verdict": verdict,
                "advice": advice,
                "matches": matches,
                "cf_avg_r": avg_r,
                "cf_loss_rate_pct": loss_rate,
                "requires_human_approval": True,
            })

        recs.sort(
            key=lambda item: (
                {"medium": 2, "low": 1}.get(item["priority"], 0),
                item["cf_avg_r"],
                item["matches"],
            ),
            reverse=True,
        )
        return recs


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate shadow rules on the ML training dataset.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Input JSONL dataset path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum matches before a rule is considered.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = ShadowModelEvaluator(dataset_path=args.dataset).build_report(
        limit=args.limit,
        min_matches=args.min_matches,
    )
    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)

    result = {
        "loaded_rows": report.get("meta", {}).get("loaded_rows", 0),
        "output_path": output_path,
        "rules": {
            name: {
                "matches": values.get("matches", 0),
                "cf_avg_r": values.get("cf_avg_r", 0),
                "cf_loss_rate_pct": values.get("cf_loss_rate_pct", 0),
            }
            for name, values in (report.get("rules") or {}).items()
        },
        "recommendations": report.get("recommendations", [])[:5],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
