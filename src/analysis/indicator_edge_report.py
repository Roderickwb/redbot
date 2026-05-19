"""Rank indicator/feature edge from labeled strategy events.

This report is read-only. It asks a narrow question: which existing inputs
have historically separated better and worse counterfactual outcomes?
It does not change live trading behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.analysis.ml_training_dataset import (
    DEFAULT_LATEST_JSONL,
    DEFAULT_OUTPUT_DIR as ML_DATASET_OUTPUT_DIR,
)

DEFAULT_OUTPUT_DIR = os.path.join("analysis", "indicator_edge")
DEFAULT_LATEST_FILE = "latest_indicator_edge_report.json"
DEFAULT_DATASET_PATH = os.path.join(ML_DATASET_OUTPUT_DIR, DEFAULT_LATEST_JSONL)
DEFAULT_MIN_SAMPLES = 25
DEFAULT_MIN_GROUP_SAMPLES = 8

NUMERIC_FEATURES = [
    "scores.trend",
    "scores.entry",
    "scores.risk",
    "scores.learning",
    "scores.sentiment",
    "market_regime.risk_multiplier",
    "market_regime.bull_pct",
    "market_regime.bear_pct",
    "market_regime.range_pct",
    "coin_profile.risk_multiplier",
    "coin_profile.n_trades",
    "coin_profile.expectancy_R",
]

CHART_NUMERIC_FIELDS = [
    "continuation_pressure",
    "breakout_pressure",
    "breakdown_pressure",
    "ema20_distance_pct",
    "ema50_distance_pct",
    "ema_spread_pct",
    "atr_pct",
    "trend_age_bars",
    "pullback_depth_pct",
    "recent_doji_count",
    "recent_opposing_wick_count",
    "recent_directional_body_count",
    "recent_directional_close_count",
    "macd_hist",
    "macd_hist_slope",
    "rsi",
]

CATEGORICAL_FEATURES = [
    "direction",
    "gpt_action",
    "primary_veto",
    "market_regime.regime",
    "market_regime.risk_mode",
    "market_regime.directional_bias",
    "coin_profile.bias",
    "coin_profile.learning_confidence",
]

CHART_CATEGORICAL_FIELDS = [
    "structure_label",
    "chop_subtype",
    "entry_timing",
    "last_candle_quality",
    "directional_continuation",
]


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


def _get_path(payload: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _load_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda row: _safe_int(row.get("timestamp")))
    if limit and len(rows) > limit:
        return rows[-int(limit):]
    return rows


def _target_r(row: dict) -> float:
    return _safe_float((row.get("targets") or {}).get("cf_r"))


def _base_context(row: dict) -> dict:
    features = row.get("features") or {}
    return {
        "overall": "all",
        "symbol": row.get("symbol") or "UNKNOWN",
        "direction": row.get("direction") or "unknown",
        "regime": _get_path(features, "market_regime.regime", "missing"),
    }


def _feature_values(row: dict) -> tuple[dict[str, float], dict[str, str]]:
    features = row.get("features") or {}
    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {}

    for name in NUMERIC_FEATURES:
        numeric[name] = _safe_float(_get_path(features, name))
    for tf in ("chart_1h", "chart_4h"):
        for field in CHART_NUMERIC_FIELDS:
            numeric[f"{tf}.{field}"] = _safe_float(_get_path(features, f"{tf}.{field}"))
        for field in CHART_CATEGORICAL_FIELDS:
            value = _get_path(features, f"{tf}.{field}", "missing")
            categorical[f"{tf}.{field}"] = str(value)

    for name in CATEGORICAL_FEATURES:
        if name in {"direction", "gpt_action", "primary_veto"}:
            value = row.get(name)
        else:
            value = _get_path(features, name, "missing")
        categorical[name] = str(value or "missing")

    flags = [str(flag).lower() for flag in _get_path(features, "coin_profile.flags", []) or []]
    for flag in flags:
        categorical[f"coin_profile.flag.{flag}"] = "present"
    return numeric, categorical


class IndicatorEdgeReport:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path

    def build_report(
        self,
        limit: Optional[int] = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    ) -> dict:
        rows = _load_jsonl(self.dataset_path, limit=limit)
        usable = [row for row in rows if (row.get("targets") or {}).get("cf_r") is not None]
        overall = self._rank_context(usable, min_samples=min_samples, min_group_samples=min_group_samples)
        by_symbol = self._rank_by_context(usable, "symbol", min_samples=min_samples, min_group_samples=min_group_samples)
        by_direction = self._rank_by_context(usable, "direction", min_samples=min_samples, min_group_samples=min_group_samples)
        by_regime = self._rank_by_context(usable, "regime", min_samples=min_samples, min_group_samples=min_group_samples)

        top_features = overall.get("top_features", [])
        weak_features = overall.get("weak_features", [])
        return {
            "created_utc": _utc_now(),
            "status": "OK" if usable else "NO_DATA",
            "meta": {
                "dataset_path": self.dataset_path,
                "loaded_rows": len(rows),
                "usable_rows": len(usable),
                "limit": limit,
                "min_samples": min_samples,
                "min_group_samples": min_group_samples,
                "live_effect": False,
            },
            "summary": {
                "usable_rows": len(usable),
                "ranked_features": len(top_features) + len(weak_features),
                "top_feature": top_features[0] if top_features else None,
                "weak_feature": weak_features[0] if weak_features else None,
                "symbols_ranked": len(by_symbol),
                "directions_ranked": len(by_direction),
                "regimes_ranked": len(by_regime),
            },
            "overall": overall,
            "by_symbol": by_symbol,
            "by_direction": by_direction,
            "by_regime": by_regime,
        }

    def _rank_by_context(self, rows: list[dict], key: str, min_samples: int, min_group_samples: int) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(_base_context(row).get(key) or "missing")].append(row)
        result = []
        for value, group_rows in grouped.items():
            ranked = self._rank_context(group_rows, min_samples=min_samples, min_group_samples=min_group_samples)
            if ranked.get("sample_size", 0) >= min_samples and ranked.get("top_features"):
                result.append({
                    key: value,
                    "sample_size": ranked.get("sample_size"),
                    "cf_avg_r": ranked.get("cf_avg_r"),
                    "top_features": ranked.get("top_features", [])[:5],
                    "weak_features": ranked.get("weak_features", [])[:3],
                })
        result.sort(key=lambda item: abs(_safe_float((item.get("top_features") or [{}])[0].get("edge_r"))), reverse=True)
        return result[:20]

    def _rank_context(self, rows: list[dict], min_samples: int, min_group_samples: int) -> dict:
        if len(rows) < min_samples:
            return {
                "sample_size": len(rows),
                "cf_avg_r": round(sum(_target_r(row) for row in rows) / len(rows), 4) if rows else 0.0,
                "top_features": [],
                "weak_features": [],
            }

        numeric_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        categorical_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            cf_r = _target_r(row)
            numeric, categorical = _feature_values(row)
            for name, value in numeric.items():
                numeric_values[name].append((value, cf_r))
            for name, value in categorical.items():
                categorical_values[name][value].append(cf_r)

        features: list[dict] = []
        for name, values in numeric_values.items():
            scored = self._score_numeric(name, values, min_group_samples)
            if scored:
                features.append(scored)
        for name, values in categorical_values.items():
            scored = self._score_categorical(name, values, min_group_samples)
            if scored:
                features.append(scored)

        features.sort(key=lambda item: abs(_safe_float(item.get("edge_r"))), reverse=True)
        weak = [item for item in features if abs(_safe_float(item.get("edge_r"))) < 0.15]
        weak.sort(key=lambda item: abs(_safe_float(item.get("edge_r"))))
        return {
            "sample_size": len(rows),
            "cf_avg_r": round(sum(_target_r(row) for row in rows) / len(rows), 4),
            "top_features": features[:20],
            "weak_features": weak[:20],
        }

    def _score_numeric(self, name: str, values: list[tuple[float, float]], min_group_samples: int) -> Optional[dict]:
        clean = [(x, y) for x, y in values if x is not None]
        if len(clean) < min_group_samples * 2:
            return None
        sorted_x = sorted(x for x, _ in clean)
        median = sorted_x[len(sorted_x) // 2]
        low = [y for x, y in clean if x <= median]
        high = [y for x, y in clean if x > median]
        if len(low) < min_group_samples or len(high) < min_group_samples:
            return None
        low_avg = sum(low) / len(low)
        high_avg = sum(high) / len(high)
        edge = high_avg - low_avg
        return {
            "feature": name,
            "type": "numeric",
            "sample_size": len(clean),
            "split": round(median, 6),
            "high_avg_r": round(high_avg, 4),
            "low_avg_r": round(low_avg, 4),
            "edge_r": round(edge, 4),
            "interpretation": "higher_better" if edge > 0 else "lower_better",
        }

    def _score_categorical(self, name: str, values: dict[str, list[float]], min_group_samples: int) -> Optional[dict]:
        groups = []
        for value, outcomes in values.items():
            if len(outcomes) >= min_group_samples:
                groups.append({
                    "value": value,
                    "sample_size": len(outcomes),
                    "avg_r": sum(outcomes) / len(outcomes),
                })
        if len(groups) < 2:
            return None
        groups.sort(key=lambda item: item["avg_r"], reverse=True)
        best = groups[0]
        worst = groups[-1]
        edge = best["avg_r"] - worst["avg_r"]
        return {
            "feature": name,
            "type": "categorical",
            "sample_size": sum(group["sample_size"] for group in groups),
            "best_value": best["value"],
            "best_avg_r": round(best["avg_r"], 4),
            "worst_value": worst["value"],
            "worst_avg_r": round(worst["avg_r"], 4),
            "edge_r": round(edge, 4),
            "values_ranked": len(groups),
        }


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_indicator_edge_report(
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
) -> dict:
    report = IndicatorEdgeReport(dataset_path=dataset_path).build_report(
        limit=limit,
        min_samples=min_samples,
        min_group_samples=min_group_samples,
    )
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build read-only indicator edge report.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-group-samples", type=int, default=DEFAULT_MIN_GROUP_SAMPLES)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_indicator_edge_report(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        limit=args.limit,
        min_samples=args.min_samples,
        min_group_samples=args.min_group_samples,
    )
    summary = report.get("summary", {}) or {}
    top = summary.get("top_feature") or {}
    print(json.dumps({
        "status": report.get("status"),
        "summary": summary,
        "top_feature": top,
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())