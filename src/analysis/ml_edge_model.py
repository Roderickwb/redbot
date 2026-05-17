# ============================================================
# src/analysis/ml_edge_model.py
# ============================================================
"""
ML Edge Model V1.

Trains a shadow-only edge predictor from the ML training dataset:
- expected counterfactual R;
- probability of positive counterfactual outcome.

The model is not used for live trading. It reports readiness, validation
metrics and sample predictions for advisor/shadow analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Iterable, Optional

from src.analysis.ml_training_dataset import (
    DEFAULT_LATEST_JSONL as ML_DATASET_LATEST_JSONL,
    DEFAULT_OUTPUT_DIR as ML_DATASET_OUTPUT_DIR,
    _safe_float,
    _safe_int,
)


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "ml_models")
DEFAULT_LATEST_FILE = "latest_edge_model_report.json"
DEFAULT_MODEL_FILE = "latest_edge_model.joblib"
DEFAULT_DATASET_PATH = os.path.join(ML_DATASET_OUTPUT_DIR, ML_DATASET_LATEST_JSONL)

DEFAULT_MIN_ROWS = 1000
DEFAULT_MIN_POSITIVE = 150
DEFAULT_MIN_NEGATIVE = 150
DEFAULT_TEST_SIZE_PCT = 25.0
POST_GPT_FEATURE_SET = "post_gpt_decision_v1"
PRE_GPT_FEATURE_SET = "pre_gpt_context_v1"


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


def _target_cf_r(row: dict) -> float:
    return _safe_float(_nested(row, "targets", "cf_r"))


def _target_positive(row: dict) -> int:
    return 1 if _target_cf_r(row) > 0 else 0


class MlEdgeModel:
    def __init__(self, dataset_path: str = DEFAULT_DATASET_PATH):
        self.dataset_path = dataset_path

    def build_report(
        self,
        limit: Optional[int] = None,
        min_rows: int = DEFAULT_MIN_ROWS,
        min_positive: int = DEFAULT_MIN_POSITIVE,
        min_negative: int = DEFAULT_MIN_NEGATIVE,
        test_size_pct: float = DEFAULT_TEST_SIZE_PCT,
        force_train: bool = False,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> dict:
        rows = _load_jsonl(self.dataset_path, limit=limit)
        readiness = self._readiness(rows, min_rows, min_positive, min_negative, force_train)
        report = {
            "meta": {
                "dataset_path": self.dataset_path,
                "loaded_rows": len(rows),
                "limit": limit,
                "min_rows": min_rows,
                "min_positive": min_positive,
                "min_negative": min_negative,
                "test_size_pct": test_size_pct,
                "force_train": force_train,
                "feature_set": feature_set,
                "feature_contract": self._feature_contract(feature_set),
            },
            "readiness": readiness,
            "dataset_summary": self._dataset_summary(rows),
        }

        if readiness["status"] != "ready":
            report["model"] = {
                "status": "not_trained",
                "reason": readiness["reason"],
            }
            return report

        try:
            trained = self._train(rows, test_size_pct=test_size_pct, output_dir=output_dir, feature_set=feature_set)
            report["model"] = trained
        except Exception as e:
            report["model"] = {
                "status": "failed",
                "reason": str(e),
            }
        return report

    def _readiness(
        self,
        rows: list[dict],
        min_rows: int,
        min_positive: int,
        min_negative: int,
        force_train: bool,
    ) -> dict:
        positives = sum(_target_positive(row) for row in rows)
        negatives = len(rows) - positives
        reasons = []
        if len(rows) < min_rows:
            reasons.append(f"rows {len(rows)} < min_rows {min_rows}")
        if positives < min_positive:
            reasons.append(f"positive outcomes {positives} < min_positive {min_positive}")
        if negatives < min_negative:
            reasons.append(f"non-positive outcomes {negatives} < min_negative {min_negative}")

        if reasons and not force_train:
            return {
                "status": "insufficient_data",
                "reason": "; ".join(reasons),
                "rows": len(rows),
                "positive": positives,
                "non_positive": negatives,
            }
        if force_train and reasons:
            return {
                "status": "ready",
                "reason": "force_train enabled despite: " + "; ".join(reasons),
                "rows": len(rows),
                "positive": positives,
                "non_positive": negatives,
                "forced": True,
            }
        return {
            "status": "ready",
            "reason": "sample thresholds met",
            "rows": len(rows),
            "positive": positives,
            "non_positive": negatives,
        }

    def _dataset_summary(self, rows: list[dict]) -> dict:
        by_action = Counter(row.get("gpt_action") or "missing" for row in rows)
        by_direction = Counter(row.get("direction") or "missing" for row in rows)
        by_regime = Counter(str(_nested(row, "features", "market_regime", "regime") or "missing") for row in rows)
        by_veto = Counter(row.get("primary_veto") or "missing" for row in rows)
        by_chop_subtype = Counter(
            str(_nested(row, "features", "chart_1h", "chop_subtype") or "missing")
            for row in rows
        )
        cf_values = [_target_cf_r(row) for row in rows]
        return {
            "rows": len(rows),
            "positive": sum(1 for value in cf_values if value > 0),
            "non_positive": sum(1 for value in cf_values if value <= 0),
            "cf_avg_r": round(sum(cf_values) / len(cf_values), 4) if cf_values else 0.0,
            "cf_min_r": round(min(cf_values), 4) if cf_values else 0.0,
            "cf_max_r": round(max(cf_values), 4) if cf_values else 0.0,
            "by_action": dict(by_action),
            "by_direction": dict(by_direction),
            "by_regime": dict(by_regime),
            "by_veto": dict(by_veto.most_common(12)),
            "by_chop_subtype": dict(by_chop_subtype),
        }

    def _train(self, rows: list[dict], test_size_pct: float, output_dir: str, feature_set: str) -> dict:
        try:
            import joblib
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
            from sklearn.pipeline import Pipeline
        except Exception as e:
            return {
                "status": "dependency_missing",
                "reason": str(e),
            }

        split_idx = max(1, int(len(rows) * (1.0 - test_size_pct / 100.0)))
        split_idx = min(split_idx, len(rows) - 1)
        train_rows = rows[:split_idx]
        test_rows = rows[split_idx:]

        x_train = [self._feature_dict(row, feature_set=feature_set) for row in train_rows]
        x_test = [self._feature_dict(row, feature_set=feature_set) for row in test_rows]
        y_r_train = [_target_cf_r(row) for row in train_rows]
        y_r_test = [_target_cf_r(row) for row in test_rows]
        y_pos_train = [_target_positive(row) for row in train_rows]
        y_pos_test = [_target_positive(row) for row in test_rows]

        regressor = Pipeline([
            ("features", DictVectorizer(sparse=False)),
            ("model", RandomForestRegressor(
                n_estimators=120,
                max_depth=5,
                min_samples_leaf=8,
                random_state=42,
            )),
        ])
        classifier = Pipeline([
            ("features", DictVectorizer(sparse=False)),
            ("model", RandomForestClassifier(
                n_estimators=120,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        regressor.fit(x_train, y_r_train)
        classifier.fit(x_train, y_pos_train)

        pred_r = regressor.predict(x_test)
        pred_pos = classifier.predict(x_test)
        pred_proba = classifier.predict_proba(x_test)[:, 1]

        model_path = os.path.join(output_dir, DEFAULT_MODEL_FILE)
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump({
            "regressor": regressor,
            "classifier": classifier,
            "feature_version": feature_set,
        }, model_path)

        auc = None
        try:
            if len(set(y_pos_test)) >= 2:
                auc = float(roc_auc_score(y_pos_test, pred_proba))
        except Exception:
            auc = None

        return {
            "status": "trained",
            "model_path": model_path,
            "feature_version": feature_set,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "metrics": {
                "regression_mae_r": round(float(mean_absolute_error(y_r_test, pred_r)), 4),
                "regression_rmse_r": round(float(mean_squared_error(y_r_test, pred_r) ** 0.5), 4),
                "classification_accuracy": round(float(accuracy_score(y_pos_test, pred_pos)), 4),
                "classification_auc": round(auc, 4) if auc is not None else None,
            },
            "prediction_summary": {
                "avg_predicted_r": round(float(np.mean(pred_r)), 4),
                "avg_actual_r": round(float(np.mean(y_r_test)), 4),
                "avg_p_positive": round(float(np.mean(pred_proba)), 4),
            },
            "sample_predictions": self._sample_predictions(test_rows, pred_r, pred_proba),
        }

    def _feature_contract(self, feature_set: str) -> dict:
        if feature_set == PRE_GPT_FEATURE_SET:
            return {
                "allowed_timing": "before_gpt_call",
                "forbidden_fields": ["gpt_action", "gpt_confidence", "gpt_scores", "primary_veto", "learning_effect"],
                "live_effect": False,
            }
        return {
            "allowed_timing": "after_gpt_decision",
            "forbidden_fields": [],
            "live_effect": False,
        }

    def _feature_dict(self, row: dict, feature_set: str = POST_GPT_FEATURE_SET) -> dict:
        features = row.get("features") or {}
        scores = features.get("scores") or {}
        regime = features.get("market_regime") or {}
        profile = features.get("coin_profile") or {}
        c1 = features.get("chart_1h") or {}
        c4 = features.get("chart_4h") or {}

        result = {
            "symbol": row.get("symbol") or "UNKNOWN",
            "direction": row.get("direction") or "unknown",
            "primary_veto": row.get("primary_veto") or "missing",
            "gpt_action": row.get("gpt_action") or "missing",
            "gpt_confidence": _safe_float(row.get("gpt_confidence")),
            "score_trend": _safe_float(scores.get("trend")),
            "score_entry": _safe_float(scores.get("entry")),
            "score_risk": _safe_float(scores.get("risk")),
            "score_learning": _safe_float(scores.get("learning")),
            "score_sentiment": _safe_float(scores.get("sentiment")),
            "regime": regime.get("regime") or "missing",
            "regime_risk_mode": regime.get("risk_mode") or "missing",
            "regime_bull_pct": _safe_float(regime.get("bull_pct")),
            "regime_bear_pct": _safe_float(regime.get("bear_pct")),
            "regime_range_pct": _safe_float(regime.get("range_pct")),
            "profile_risk_multiplier": _safe_float(profile.get("risk_multiplier"), 1.0),
            "profile_confidence": profile.get("learning_confidence") or "missing",
            "profile_bias": profile.get("bias") or "missing",
            "profile_n_trades": _safe_float(profile.get("n_trades")),
            "profile_expectancy_R": _safe_float(profile.get("expectancy_R")),
        }
        if feature_set == PRE_GPT_FEATURE_SET:
            for key in ("primary_veto", "gpt_action", "gpt_confidence", "score_trend", "score_entry", "score_risk", "score_learning", "score_sentiment"):
                result.pop(key, None)

        result.update(self._chart_features(c1, prefix="c1"))
        result.update(self._chart_features(c4, prefix="c4"))

        flags = [str(flag).lower() for flag in (profile.get("flags") or [])]
        for key in (
            "drawdown",
            "counterfactual_edge_negative",
            "counterfactual_edge_positive",
            "conservative_hold",
            "filter_review",
            "range_breakout_candidate",
            "sample_low",
        ):
            result[f"profile_flag_{key}"] = int(any(key in flag for flag in flags))
        return result

    def _chart_features(self, chart: dict, prefix: str) -> dict:
        return {
            f"{prefix}_structure": chart.get("structure_label") or "missing",
            f"{prefix}_chop_subtype": chart.get("chop_subtype") or "missing",
            f"{prefix}_entry_timing": chart.get("entry_timing") or "missing",
            f"{prefix}_last_candle_quality": chart.get("last_candle_quality") or "missing",
            f"{prefix}_directional_continuation": int(bool(chart.get("directional_continuation"))),
            f"{prefix}_continuation_pressure": _safe_float(chart.get("continuation_pressure")),
            f"{prefix}_breakout_pressure": _safe_float(chart.get("breakout_pressure")),
            f"{prefix}_breakdown_pressure": _safe_float(chart.get("breakdown_pressure")),
            f"{prefix}_ema20_distance_pct": _safe_float(chart.get("ema20_distance_pct")),
            f"{prefix}_ema50_distance_pct": _safe_float(chart.get("ema50_distance_pct")),
            f"{prefix}_ema_spread_pct": _safe_float(chart.get("ema_spread_pct")),
            f"{prefix}_atr_pct": _safe_float(chart.get("atr_pct")),
            f"{prefix}_trend_age_bars": _safe_float(chart.get("trend_age_bars")),
            f"{prefix}_pullback_depth_pct": _safe_float(chart.get("pullback_depth_pct")),
            f"{prefix}_recent_doji_count": _safe_float(chart.get("recent_doji_count")),
            f"{prefix}_recent_opposing_wick_count": _safe_float(chart.get("recent_opposing_wick_count")),
            f"{prefix}_recent_directional_body_count": _safe_float(chart.get("recent_directional_body_count")),
            f"{prefix}_recent_directional_close_count": _safe_float(chart.get("recent_directional_close_count")),
            f"{prefix}_macd_hist": _safe_float(chart.get("macd_hist")),
            f"{prefix}_macd_hist_slope": _safe_float(chart.get("macd_hist_slope")),
            f"{prefix}_rsi": _safe_float(chart.get("rsi"), 50.0),
        }

    def _sample_predictions(self, rows: list[dict], pred_r: Iterable[float], pred_proba: Iterable[float]) -> list[dict]:
        items = []
        for row, r_value, p_value in list(zip(rows, pred_r, pred_proba))[-20:]:
            items.append({
                "id": row.get("id"),
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "action": row.get("gpt_action"),
                "primary_veto": row.get("primary_veto"),
                "actual_cf_r": _target_cf_r(row),
                "predicted_r": round(float(r_value), 4),
                "p_positive": round(float(p_value), 4),
            })
        return items


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate ML Edge Model V1.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Input JSONL dataset path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS, help="Minimum rows required to train.")
    parser.add_argument("--min-positive", type=int, default=DEFAULT_MIN_POSITIVE, help="Minimum positive outcomes.")
    parser.add_argument("--min-negative", type=int, default=DEFAULT_MIN_NEGATIVE, help="Minimum non-positive outcomes.")
    parser.add_argument("--test-size-pct", type=float, default=DEFAULT_TEST_SIZE_PCT, help="Time-ordered validation split percent.")
    parser.add_argument("--force-train", action="store_true", help="Train even when readiness thresholds are not met.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = MlEdgeModel(dataset_path=args.dataset)
    report = model.build_report(
        limit=args.limit,
        min_rows=args.min_rows,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        test_size_pct=args.test_size_pct,
        force_train=args.force_train,
        output_dir=args.output_dir,
    )
    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)

    result = {
        "loaded_rows": report.get("meta", {}).get("loaded_rows", 0),
        "readiness": report.get("readiness", {}),
        "model_status": (report.get("model") or {}).get("status"),
        "metrics": (report.get("model") or {}).get("metrics", {}),
        "output_path": output_path,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
