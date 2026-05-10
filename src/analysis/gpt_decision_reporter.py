# ============================================================
# src/analysis/gpt_decision_reporter.py
# ============================================================
"""
Read-only analytics for GPT strategy decisions.

This report connects GPT's structured output (scores, veto, learning effect)
to later labeled outcomes and counterfactual trade results.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "gpt_decisions")
DEFAULT_LATEST_FILE = "latest_gpt_decision_report.json"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _pct(part: int, total: int) -> float:
    return round(part / total * 100.0, 2) if total else 0.0


def _avg(total: float, count: int) -> float:
    return round(total / count, 4) if count else 0.0


def _score_band(score: int) -> str:
    if score >= 80:
        return "80_100"
    if score >= 65:
        return "65_79"
    if score >= 50:
        return "50_64"
    if score > 0:
        return "1_49"
    return "missing_or_zero"


def _new_bucket() -> dict:
    return {
        "events": 0,
        "hold": 0,
        "open": 0,
        "open_long": 0,
        "open_short": 0,
        "zero_conf": 0,
        "missing_scores": 0,
        "cf_count": 0,
        "cf_total_r": 0.0,
        "cf_positive": 0,
        "cf_loss": 0,
        "missed_opportunity": 0,
        "skip_protected": 0,
        "open_followed_through": 0,
        "open_went_against": 0,
        "open_mixed_volatility": 0,
        "open_no_followthrough": 0,
    }


class GptDecisionReporter:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)

    def build_report(self, limit: int = 5000, symbol: Optional[str] = None) -> dict:
        events = self._load_events(limit=limit, symbol=symbol)
        report = self._summarize(events)
        report["meta"] = {
            "loaded_gpt_decisions": len(events),
            "limit": limit,
            "symbol": symbol,
        }
        return report

    def _load_events(self, limit: int, symbol: Optional[str]) -> list[dict]:
        params: list[Any] = []
        where = ["event_type = 'gpt_decision'", "outcome_status = 'labeled'"]
        if symbol:
            where.append("symbol = ?")
            params.append(symbol)

        params.append(int(limit))
        rows = self.db.execute_query(
            f"""
            SELECT id, timestamp, symbol, gpt_action, gpt_confidence, features_json, outcome_json
            FROM strategy_events
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            tuple(params),
        )
        cols = ["id", "timestamp", "symbol", "gpt_action", "gpt_confidence", "features_json", "outcome_json"]
        return [dict(zip(cols, row)) for row in rows] if rows else []

    def _summarize(self, events: list[dict]) -> dict:
        totals = _new_bucket()
        by_symbol = defaultdict(_new_bucket)
        by_veto = defaultdict(_new_bucket)
        by_learning_effect = defaultdict(_new_bucket)
        by_entry_score_band = defaultdict(_new_bucket)
        by_risk_score_band = defaultdict(_new_bucket)
        by_learning_score_band = defaultdict(_new_bucket)
        attention_cases = {
            "holds_with_positive_counterfactual": [],
            "opens_with_weak_entry_score": [],
            "counterfactual_negative_ignored": [],
            "gpt_errors": [],
        }

        score_totals = defaultdict(float)
        score_counts = defaultdict(int)

        for event in events:
            features = self._parse_json(event.get("features_json"))
            outcome = self._parse_json(event.get("outcome_json"))
            scores = features.get("scores") or {}
            primary_veto = features.get("primary_veto") or "missing"
            learning_effect = features.get("learning_effect") or "missing"

            buckets = [
                totals,
                by_symbol[event.get("symbol") or "UNKNOWN"],
                by_veto[primary_veto],
                by_learning_effect[learning_effect],
                by_entry_score_band[_score_band(_safe_int(scores.get("entry")))],
                by_risk_score_band[_score_band(_safe_int(scores.get("risk")))],
                by_learning_score_band[_score_band(_safe_int(scores.get("learning")))],
            ]
            for bucket in buckets:
                self._add_event(bucket, event, features, outcome)

            for key in ("trend", "entry", "risk", "learning", "sentiment"):
                if key in scores:
                    score_totals[key] += _safe_float(scores.get(key))
                    score_counts[key] += 1

            self._collect_attention_case(attention_cases, event, features, outcome)

        return {
            "totals": self._finalize_bucket(totals),
            "score_averages": {
                key: _avg(score_totals[key], score_counts[key])
                for key in ("trend", "entry", "risk", "learning", "sentiment")
            },
            "by_symbol": self._finalize_mapping(by_symbol),
            "by_primary_veto": self._finalize_mapping(by_veto),
            "by_learning_effect": self._finalize_mapping(by_learning_effect),
            "by_entry_score_band": self._finalize_mapping(by_entry_score_band),
            "by_risk_score_band": self._finalize_mapping(by_risk_score_band),
            "by_learning_score_band": self._finalize_mapping(by_learning_score_band),
            "top_attention": self._build_attention_lists(by_symbol, by_veto, by_entry_score_band),
            "attention_cases": attention_cases,
        }

    def _add_event(self, bucket: dict, event: dict, features: dict, outcome: dict) -> None:
        bucket["events"] += 1
        action = event.get("gpt_action")
        confidence = _safe_float(event.get("gpt_confidence"))
        scores = features.get("scores") or {}
        label = outcome.get("label")
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_label = counterfactual.get("label")
        cf_r = _safe_float(counterfactual.get("r_multiple"))

        if action == "HOLD":
            bucket["hold"] += 1
        elif action == "OPEN_LONG":
            bucket["open"] += 1
            bucket["open_long"] += 1
        elif action == "OPEN_SHORT":
            bucket["open"] += 1
            bucket["open_short"] += 1

        if confidence == 0:
            bucket["zero_conf"] += 1
        if not scores or scores.get("entry") is None:
            bucket["missing_scores"] += 1

        if cf_label:
            bucket["cf_count"] += 1
            bucket["cf_total_r"] += cf_r
            if cf_label in ("cf_win", "cf_tp1_then_positive", "cf_small_win"):
                bucket["cf_positive"] += 1
            if cf_label == "cf_loss":
                bucket["cf_loss"] += 1

        if label in bucket:
            bucket[label] += 1

    def _collect_attention_case(self, cases: dict, event: dict, features: dict, outcome: dict) -> None:
        scores = features.get("scores") or {}
        entry_score = _safe_int(scores.get("entry"))
        primary_veto = features.get("primary_veto")
        action = event.get("gpt_action")
        counterfactual = outcome.get("counterfactual_trade") or {}
        cf_r = _safe_float(counterfactual.get("r_multiple"))
        cf_label = counterfactual.get("label")

        row = {
            "id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "symbol": event.get("symbol"),
            "action": action,
            "confidence": event.get("gpt_confidence"),
            "primary_veto": primary_veto,
            "entry_score": entry_score,
            "learning_effect": features.get("learning_effect"),
            "cf_label": cf_label,
            "cf_r": round(cf_r, 4),
            "outcome_label": outcome.get("label"),
            "risk_notes": features.get("risk_notes"),
        }

        if action == "HOLD" and cf_r >= 0.75:
            cases["holds_with_positive_counterfactual"].append(row)
        if action in ("OPEN_LONG", "OPEN_SHORT") and entry_score and entry_score < 60:
            cases["opens_with_weak_entry_score"].append(row)
        if action in ("OPEN_LONG", "OPEN_SHORT") and primary_veto == "counterfactual_negative":
            cases["counterfactual_negative_ignored"].append(row)
        if primary_veto in ("gpt_error", "parse_error") or _safe_float(event.get("gpt_confidence")) == 0:
            cases["gpt_errors"].append(row)

        for key in cases:
            cases[key] = cases[key][:20]

    def _finalize_mapping(self, mapping: dict[str, dict]) -> dict:
        return {
            key: self._finalize_bucket(value)
            for key, value in sorted(mapping.items(), key=lambda item: item[0])
        }

    def _finalize_bucket(self, bucket: dict) -> dict:
        result = dict(bucket)
        events = result["events"]
        opens = result["open"]
        cf_count = result["cf_count"]
        result["hold_rate_pct"] = _pct(result["hold"], events)
        result["open_rate_pct"] = _pct(opens, events)
        result["zero_conf_pct"] = _pct(result["zero_conf"], events)
        result["missing_scores_pct"] = _pct(result["missing_scores"], events)
        result["cf_avg_r"] = _avg(result["cf_total_r"], cf_count)
        result["cf_positive_rate_pct"] = _pct(result["cf_positive"], cf_count)
        result["cf_loss_rate_pct"] = _pct(result["cf_loss"], cf_count)
        result["cf_total_r"] = round(result["cf_total_r"], 4)
        return result

    def _build_attention_lists(self, by_symbol: dict, by_veto: dict, by_entry_band: dict) -> dict:
        finalized_symbols = self._finalize_mapping(by_symbol)
        finalized_veto = self._finalize_mapping(by_veto)
        finalized_entry = self._finalize_mapping(by_entry_band)
        return {
            "symbols_worst_cf_avg_r": self._top(finalized_symbols, "cf_avg_r", reverse=False, only_negative=True),
            "symbols_best_cf_avg_r": self._top(finalized_symbols, "cf_avg_r", reverse=True),
            "symbols_most_zero_conf": self._top(finalized_symbols, "zero_conf", reverse=True),
            "veto_best_cf_avg_r": self._top(finalized_veto, "cf_avg_r", reverse=True),
            "veto_worst_cf_avg_r": self._top(finalized_veto, "cf_avg_r", reverse=False, only_negative=True),
            "entry_bands": finalized_entry,
        }

    def _top(self, mapping: dict, metric: str, reverse: bool = True, only_negative: bool = False) -> list[dict]:
        rows = [
            {"name": name, metric: values.get(metric, 0), "events": values.get("events", 0)}
            for name, values in mapping.items()
        ]
        rows.sort(key=lambda row: row[metric], reverse=reverse)
        if only_negative:
            rows = [row for row in rows if row[metric] < 0]
        else:
            rows = [row for row in rows if row[metric] != 0]
        return rows[:10]

    def _parse_json(self, raw: Any) -> dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build GPT decision analytics report.")
    parser.add_argument("--limit", type=int, default=5000, help="Max labeled GPT decisions to read.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        reporter = GptDecisionReporter(db=db)
        report = reporter.build_report(limit=args.limit, symbol=args.symbol)
    finally:
        db.close_connection()

    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)
    result = {
        "loaded_gpt_decisions": report.get("meta", {}).get("loaded_gpt_decisions", 0),
        "output_path": output_path,
        "zero_conf_pct": report.get("totals", {}).get("zero_conf_pct", 0),
        "cf_avg_r": report.get("totals", {}).get("cf_avg_r", 0),
        "attention_cases": {
            key: len(value)
            for key, value in (report.get("attention_cases") or {}).items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
