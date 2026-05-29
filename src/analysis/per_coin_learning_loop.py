# ============================================================
# src/analysis/per_coin_learning_loop.py
# ============================================================
"""Per-coin learning loop summary.

Bundles existing learning outputs into coin-level improvement proposals:
- entry feature/KPI context;
- loss/opportunity clusters;
- risk sizing advice;
- next concrete phase.

Read-only: no strategy, risk, or live behavior is changed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.analysis.loss_diagnosis_report import DEFAULT_DATASET_PATH, _load_jsonl, _nested, _safe_float, _safe_int


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "per_coin_learning")
DEFAULT_LATEST_FILE = "latest_per_coin_learning_loop.json"
DEFAULT_INDICATOR_EDGE = os.path.join("analysis", "indicator_edge", "latest_indicator_edge_report.json")
DEFAULT_RISK_ADVICE_HISTORY = os.path.join("analysis", "risk", "latest_risk_advice_history_report.json")
DEFAULT_LOSS_DIAGNOSIS = os.path.join("analysis", "loss_diagnosis", "latest_loss_diagnosis_report.json")
DEFAULT_ENTRY_RULE_CANDIDATES = os.path.join("analysis", "entry_rules", "latest_entry_rule_candidate_simulator.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _target_r(row: dict) -> float:
    return _safe_float(_nested(row, "targets", "cf_r"))


def _opened(row: dict) -> bool:
    return bool(_nested(row, "targets", "opened_trade"))


def _symbol_stats(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        symbol = str(row.get("symbol") or "UNKNOWN")
        grouped[symbol].append(row)

    stats = {}
    for symbol, items in grouped.items():
        net_r = sum(_target_r(row) for row in items)
        wins = sum(1 for row in items if _target_r(row) > 0)
        losses = len(items) - wins
        stats[symbol] = {
            "symbol": symbol,
            "opened_trades": len(items),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / len(items)) * 100.0, 2) if items else 0.0,
            "net_R": round(net_r, 6),
            "avg_R": round(net_r / len(items), 6) if items else 0.0,
        }
    return stats


def _index_by_symbol(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("symbol")): row for row in rows if row.get("symbol")}


def _indicator_by_symbol(report: dict) -> dict[str, dict]:
    result = {}
    for row in report.get("by_symbol", []) or []:
        symbol = str(row.get("symbol") or "")
        if symbol:
            top = (row.get("top_features") or [{}])[0] or {}
            result[symbol] = {
                "sample_size": row.get("sample_size"),
                "cf_avg_R": row.get("cf_avg_r"),
                "top_feature": top.get("feature"),
                "top_feature_edge_R": top.get("edge_r"),
                "top_feature_best_value": top.get("best_value"),
            }
    return result


class PerCoinLearningLoop:
    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        indicator_edge_path: str = DEFAULT_INDICATOR_EDGE,
        risk_advice_history_path: str = DEFAULT_RISK_ADVICE_HISTORY,
        loss_diagnosis_path: str = DEFAULT_LOSS_DIAGNOSIS,
        entry_rule_candidates_path: str = DEFAULT_ENTRY_RULE_CANDIDATES,
    ):
        self.dataset_path = dataset_path
        self.indicator_edge_path = indicator_edge_path
        self.risk_advice_history_path = risk_advice_history_path
        self.loss_diagnosis_path = loss_diagnosis_path
        self.entry_rule_candidates_path = entry_rule_candidates_path

    def build_report(self, limit: Optional[int] = None, min_trades: int = 3) -> dict:
        rows = [row for row in _load_jsonl(self.dataset_path, limit=limit) if _opened(row)]
        stats = _symbol_stats(rows)
        indicator = _indicator_by_symbol(_load_json(self.indicator_edge_path, {}))
        risk_report = _load_json(self.risk_advice_history_path, {})
        risk_symbols = _index_by_symbol(risk_report.get("symbols", []) or [])
        loss = _load_json(self.loss_diagnosis_path, {})
        entry_sim = _load_json(self.entry_rule_candidates_path, {})

        symbols = sorted(set(stats.keys()) | set(indicator.keys()) | set(risk_symbols.keys()))
        profiles = []
        for symbol in symbols:
            row = self._coin_profile(
                symbol=symbol,
                stats=stats.get(symbol, {"symbol": symbol}),
                indicator=indicator.get(symbol, {}),
                risk=risk_symbols.get(symbol, {}),
                loss=loss,
                entry_sim=entry_sim,
                min_trades=min_trades,
            )
            profiles.append(row)

        profiles.sort(key=self._sort_key)
        actionable = [row for row in profiles if row.get("decision_needed")]
        return {
            "created_utc": _utc_now(),
            "status": "REVIEW" if actionable else "WATCH",
            "meta": {
                "dataset_path": self.dataset_path,
                "opened_rows": len(rows),
                "symbols": len(profiles),
                "min_trades": min_trades,
                "read_only": True,
                "live_effect": False,
            },
            "summary": {
                "symbols": len(profiles),
                "actionable": len(actionable),
                "underperforming": sum(1 for row in profiles if row.get("status") == "underperforming"),
                "opportunity": sum(1 for row in profiles if row.get("status") == "opportunity"),
                "risk_down_candidates": sum(1 for row in profiles if row.get("risk_advice", {}).get("current_risk_down")),
                "top_actionable": actionable[:5],
            },
            "coins": profiles,
            "actionable_coins": actionable,
        }

    def _coin_profile(
        self,
        symbol: str,
        stats: dict,
        indicator: dict,
        risk: dict,
        loss: dict,
        entry_sim: dict,
        min_trades: int,
    ) -> dict:
        opened = _safe_int(stats.get("opened_trades"))
        net_r = _safe_float(stats.get("net_R"))
        avg_r = _safe_float(stats.get("avg_R"))
        status = "collecting"
        decision_needed = False
        if opened >= min_trades and net_r < -1.0:
            status = "underperforming"
            decision_needed = True
        elif opened >= min_trades and net_r > 1.0:
            status = "opportunity"

        risk_advice = {
            "current_risk_down": bool(risk.get("current_risk_down")),
            "data_down_days": _safe_int(risk.get("data_down_days")),
            "long_multiplier": risk.get("current_long_multiplier"),
            "short_multiplier": risk.get("current_short_multiplier"),
            "data_reasons": risk.get("data_reasons", []),
        }
        if risk_advice["current_risk_down"] and risk_advice["data_down_days"] >= 3:
            decision_needed = True

        proposed = self._proposal(symbol, status, stats, risk_advice, indicator, entry_sim)
        return {
            "symbol": symbol,
            "status": status,
            "decision_needed": decision_needed,
            "performance": stats,
            "entry_feature_context": indicator,
            "risk_advice": risk_advice,
            "global_loss_context": {
                "top_loss": (loss.get("summary") or {}).get("top_loss"),
                "top_opportunity": (loss.get("summary") or {}).get("top_opportunity"),
            },
            "entry_rule_candidate": entry_sim.get("best_candidate"),
            "proposal": proposed,
            "live_effect": False,
        }

    def _proposal(self, symbol: str, status: str, stats: dict, risk: dict, indicator: dict, entry_sim: dict) -> dict:
        if status == "underperforming":
            return {
                "type": "coin_risk_or_entry_review",
                "title": f"{symbol} underperforms; review risk/entry settings",
                "recommended_action": "approve_paper_test",
                "suggested_change": "Test reduced sizing or stricter entry rules for this coin.",
                "why": f"Coin net_R={stats.get('net_R')} over {stats.get('opened_trades')} opened trades.",
            }
        if risk.get("current_risk_down") and _safe_int(risk.get("data_down_days")) >= 3:
            return {
                "type": "coin_risk_sizing",
                "title": f"{symbol} has stable risk-down advice",
                "recommended_action": "approve_paper_test",
                "suggested_change": f"Test long_multiplier={risk.get('long_multiplier')} short_multiplier={risk.get('short_multiplier')} for this coin.",
                "why": f"Risk-down advice persisted for {risk.get('data_down_days')} days.",
            }
        if status == "opportunity":
            return {
                "type": "coin_opportunity",
                "title": f"{symbol} shows positive edge",
                "recommended_action": "keep_collecting",
                "suggested_change": "Keep as positive context; only test priority increase after more cluster-specific proof.",
                "why": f"Coin net_R={stats.get('net_R')} avg_R={stats.get('avg_R')}.",
            }
        if indicator.get("top_feature"):
            return {
                "type": "coin_feature_context",
                "title": f"{symbol} has feature context",
                "recommended_action": "auto_context",
                "suggested_change": "Use top feature as GPT/profile context.",
                "why": f"{indicator.get('top_feature')} edge_R={indicator.get('top_feature_edge_R')}.",
            }
        return {
            "type": "collect_more_evidence",
            "title": f"{symbol} is collecting evidence",
            "recommended_action": "wait",
            "suggested_change": "No coin-specific change yet.",
            "why": "Insufficient or neutral evidence.",
        }

    @staticmethod
    def _sort_key(row: dict) -> tuple[int, float, str]:
        priority = {"underperforming": 0, "opportunity": 1, "collecting": 2}.get(row.get("status"), 9)
        net_r = _safe_float((row.get("performance") or {}).get("net_R"))
        return (priority, net_r, str(row.get("symbol")))


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_per_coin_learning_loop(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    min_trades: int = 3,
) -> dict:
    report = PerCoinLearningLoop().build_report(limit=limit, min_trades=min_trades)
    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    report["output_path"] = output_path
    write_json(output_path, report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build per-coin learning loop report.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-trades", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_per_coin_learning_loop(output_dir=args.output_dir, limit=args.limit, min_trades=args.min_trades)
    print(json.dumps({
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
