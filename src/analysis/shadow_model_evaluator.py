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
from collections import defaultdict
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


def _pressure(row: dict) -> int:
    c1 = _chart(row, "1h")
    return max(
        _safe_int(c1.get("breakout_pressure")),
        _safe_int(c1.get("breakdown_pressure")),
        _safe_int(c1.get("continuation_pressure")),
    )


def _pressure_band(value: int) -> str:
    if value >= 80:
        return "p80_plus"
    if value >= 60:
        return "p60_79"
    if value >= 40:
        return "p40_59"
    if value > 0:
        return "p1_39"
    return "p0"


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
        discovered = self._discover_patterns(rows, min_matches=min_matches)
        hypotheses = self._generate_hypotheses(discovered, min_matches=min_matches)
        recommendations = self._recommendations(evaluations, min_matches=min_matches)
        return {
            "meta": {
                "dataset_path": self.dataset_path,
                "loaded_rows": len(rows),
                "limit": limit,
                "min_matches": min_matches,
            },
            "rules": evaluations,
            "discovered_patterns": discovered,
            "generated_hypotheses": hypotheses,
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

    def _discover_patterns(self, rows: list[dict], min_matches: int) -> dict:
        held_rows = [row for row in rows if _is_hold(row)]
        pattern_sets = {
            "direction_regime_veto": lambda row: [
                row.get("direction") or "missing",
                _regime(row).get("regime") or "missing",
                row.get("primary_veto") or "missing",
            ],
            "chart_context": lambda row: [
                row.get("direction") or "missing",
                _regime(row).get("regime") or "missing",
                _chart(row).get("structure_label") or "missing",
                _chart(row).get("chop_subtype") or "missing",
                _chart(row).get("entry_timing") or "missing",
            ],
            "pressure_context": lambda row: [
                row.get("direction") or "missing",
                _regime(row).get("regime") or "missing",
                row.get("primary_veto") or "missing",
                _pressure_band(_pressure(row)),
            ],
            "profile_risk_context": lambda row: [
                row.get("direction") or "missing",
                row.get("primary_veto") or "missing",
                f"risk_mult_{_risk_multiplier(row):.2f}",
                "drawdown" if _has_flag(row, "drawdown") else "no_drawdown_flag",
                "cf_negative" if _has_flag(row, "counterfactual_edge_negative") else "no_cf_negative_flag",
            ],
        }

        return {
            name: self._discover_pattern_set(held_rows, builder, min_matches)
            for name, builder in pattern_sets.items()
        }

    def _discover_pattern_set(
        self,
        rows: list[dict],
        key_builder: Callable[[dict], list[str]],
        min_matches: int,
    ) -> list[dict]:
        buckets = defaultdict(list)
        for row in rows:
            key = "|".join(str(part) for part in key_builder(row))
            buckets[key].append(row)

        patterns = []
        for key, bucket_rows in buckets.items():
            if len(bucket_rows) < min_matches:
                continue
            total_r = sum(_cf_r(row) for row in bucket_rows)
            positives = [row for row in bucket_rows if _cf_r(row) > 0]
            large_positive = [row for row in bucket_rows if _cf_r(row) >= 1.0]
            losses = [row for row in bucket_rows if _cf_r(row) <= -0.5]
            avg_r = total_r / len(bucket_rows)
            loss_rate = len(losses) / len(bucket_rows) * 100.0
            positive_rate = len(positives) / len(bucket_rows) * 100.0
            patterns.append({
                "pattern": key,
                "matches": len(bucket_rows),
                "cf_total_r": round(total_r, 4),
                "cf_avg_r": round(avg_r, 4),
                "cf_positive_rate_pct": round(positive_rate, 2),
                "cf_loss_rate_pct": round(loss_rate, 2),
                "cf_large_positive": len(large_positive),
                "interpretation": self._interpret_discovered(avg_r, loss_rate, len(bucket_rows)),
                "sample_cases": [self._sample_case(row) for row in bucket_rows[:8]],
            })

        patterns.sort(
            key=lambda item: (
                item["cf_avg_r"],
                item["matches"],
                -item["cf_loss_rate_pct"],
            ),
            reverse=True,
        )
        return patterns[:15]

    def _interpret_discovered(self, avg_r: float, loss_rate: float, matches: int) -> str:
        if matches < 5:
            return "small_sample"
        if avg_r >= 0.35 and loss_rate <= 45:
            return "promising_pattern"
        if avg_r <= -0.15 or loss_rate >= 55:
            return "protective_hold_pattern"
        return "mixed_pattern"

    def _generate_hypotheses(self, discovered: dict, min_matches: int) -> list[dict]:
        hypotheses = []
        for group_name, patterns in discovered.items():
            for pattern in patterns or []:
                matches = _safe_int(pattern.get("matches"))
                if matches < min_matches:
                    continue
                interpretation = pattern.get("interpretation")
                if interpretation == "promising_pattern":
                    hypotheses.append(self._hypothesis_from_pattern(
                        group_name=group_name,
                        pattern=pattern,
                        hypothesis_type="allow_or_relax_hold",
                    ))
                elif interpretation == "protective_hold_pattern":
                    hypotheses.append(self._hypothesis_from_pattern(
                        group_name=group_name,
                        pattern=pattern,
                        hypothesis_type="protect_or_block",
                    ))

        hypotheses.sort(
            key=lambda item: (
                self._hypothesis_rank(item),
                item.get("confidence_score", 0),
                item.get("matches", 0),
            ),
            reverse=True,
        )
        return hypotheses[:25]

    def _hypothesis_from_pattern(self, group_name: str, pattern: dict, hypothesis_type: str) -> dict:
        avg_r = _safe_float(pattern.get("cf_avg_r"))
        loss_rate = _safe_float(pattern.get("cf_loss_rate_pct"))
        pos_rate = _safe_float(pattern.get("cf_positive_rate_pct"))
        matches = _safe_int(pattern.get("matches"))
        confidence = self._hypothesis_confidence(matches, avg_r, loss_rate, hypothesis_type)
        rule_id = self._hypothesis_rule_id(group_name, pattern.get("pattern"), hypothesis_type)

        if hypothesis_type == "allow_or_relax_hold":
            proposed_action = "shadow_test_relaxed_hold"
            description = (
                f"Test whether HOLD can be relaxed for pattern {pattern.get('pattern')} "
                f"because historical counterfactuals look positive."
            )
            guardrails = [
                "shadow-only",
                "do_not_enable_live_from_single_report",
                "require_repeat_occurrence",
                "require_human_approval",
                "split_by_symbol_before_live",
            ]
        else:
            proposed_action = "preserve_or_strengthen_block"
            description = (
                f"Keep or strengthen protection for pattern {pattern.get('pattern')} "
                f"because historical counterfactuals look risky."
            )
            guardrails = [
                "safe_to_keep_as_protection",
                "do_not_loosen_broad_rules_covering_this_pattern",
                "require_human_approval_for_new_hard_blocks",
            ]

        return {
            "rule_id": rule_id,
            "hypothesis_type": hypothesis_type,
            "group": group_name,
            "pattern": pattern.get("pattern"),
            "proposed_action": proposed_action,
            "description": description,
            "matches": matches,
            "cf_avg_r": avg_r,
            "cf_positive_rate_pct": pos_rate,
            "cf_loss_rate_pct": loss_rate,
            "confidence": confidence,
            "confidence_score": self._confidence_score(confidence),
            "requires_human_approval": True,
            "guardrails": guardrails,
            "sample_cases": pattern.get("sample_cases", [])[:5],
        }

    def _hypothesis_rule_id(self, group_name: str, pattern: Any, hypothesis_type: str) -> str:
        raw = f"{hypothesis_type}|{group_name}|{pattern}"
        safe = "".join(ch if ch.isalnum() else "_" for ch in raw.lower())
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_")[:96]

    def _hypothesis_confidence(self, matches: int, avg_r: float, loss_rate: float, hypothesis_type: str) -> str:
        if matches >= 30:
            sample_score = 2
        elif matches >= 10:
            sample_score = 1
        else:
            sample_score = 0

        if hypothesis_type == "allow_or_relax_hold":
            edge_score = 2 if avg_r >= 0.75 and loss_rate <= 25 else 1 if avg_r >= 0.35 and loss_rate <= 45 else 0
        else:
            edge_score = 2 if avg_r <= -0.35 and loss_rate >= 60 else 1 if avg_r <= -0.15 or loss_rate >= 55 else 0

        total = sample_score + edge_score
        if total >= 4:
            return "high"
        if total >= 2:
            return "medium"
        return "low"

    def _confidence_score(self, confidence: str) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)

    def _hypothesis_rank(self, item: dict) -> int:
        return {
            "allow_or_relax_hold": 2,
            "protect_or_block": 1,
        }.get(str(item.get("hypothesis_type")), 0)


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
        "generated_hypotheses": (report.get("generated_hypotheses") or [])[:8],
        "top_discovered_patterns": {
            name: values[:3]
            for name, values in (report.get("discovered_patterns") or {}).items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
