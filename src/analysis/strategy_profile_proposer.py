# ============================================================
# src/analysis/strategy_profile_proposer.py
# ============================================================
"""
Proposal layer for strategy-event based coin profiles.

This module turns the learning report into cautious per-coin profile proposals.
It is read-only with respect to trading behavior: it does not update the DB and
does not change live strategy settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("strategy_profile_proposer")


DEFAULT_REPORT_PATH = os.path.join("analysis", "strategy_events", "latest_strategy_learning_report.json")
DEFAULT_OUTPUT_PATH = os.path.join("analysis", "strategy_events", "latest_strategy_profile_proposals.json")


def _empty_metrics() -> Dict[str, Any]:
    return {
        "events": 0,
        "trade_open": 0,
        "trade_profitable": 0,
        "trade_losing": 0,
        "trade_pnl_eur": 0.0,
        "trade_winrate_pct": 0.0,
        "missed_opportunity": 0,
        "missed_rate_pct": 0.0,
        "range_events": 0,
        "range_breakout_rate_pct": 0.0,
    }


class StrategyProfileProposer:
    def __init__(self, min_events: int = 30, min_trades: int = 5):
        self.min_events = int(min_events)
        self.min_trades = int(min_trades)

    def build_proposals(self, learning_payload: Dict[str, Any]) -> Dict[str, Any]:
        report = learning_payload.get("report", learning_payload)
        by_symbol = report.get("by_symbol", {})
        trade_summary = report.get("trade_open_summary", {})
        range_summary = report.get("range_summary", {})

        proposals = {}
        for symbol in sorted(by_symbol.keys()):
            metrics = self._collect_symbol_metrics(
                symbol=symbol,
                symbol_data=by_symbol.get(symbol, {}),
                trade_data=trade_summary.get(symbol, {}),
                range_data=range_summary.get(symbol, {}),
            )
            proposals[symbol] = self._propose_for_symbol(symbol, metrics)

        return {
            "source": "strategy_events",
            "min_events": self.min_events,
            "min_trades": self.min_trades,
            "n_symbols": len(proposals),
            "proposals": proposals,
        }

    def _collect_symbol_metrics(
        self,
        symbol: str,
        symbol_data: Dict[str, Any],
        trade_data: Dict[str, Any],
        range_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = _empty_metrics()
        metrics.update({
            "symbol": symbol,
            "events": int(symbol_data.get("events", 0) or 0),
            "missed_opportunity": int(symbol_data.get("missed_opportunity", 0) or 0),
            "missed_rate_pct": float(symbol_data.get("missed_rate_pct", 0.0) or 0.0),
            "trade_open": int(trade_data.get("trade_open", 0) or 0),
            "trade_profitable": int(trade_data.get("trade_profitable", 0) or 0),
            "trade_losing": int(trade_data.get("trade_losing", 0) or 0),
            "trade_pnl_eur": float(trade_data.get("trade_pnl_eur", 0.0) or 0.0),
            "trade_winrate_pct": float(trade_data.get("trade_winrate_pct", 0.0) or 0.0),
            "range_events": int(range_data.get("events", 0) or 0),
            "range_breakout_rate_pct": float(range_data.get("range_breakout_rate_pct", 0.0) or 0.0),
        })
        return metrics

    def _propose_for_symbol(self, symbol: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        confidence = self._confidence(metrics)
        risk_multiplier = 1.0
        bias = "neutral"
        flags = []

        if metrics["trade_open"] >= self.min_trades:
            if metrics["trade_winrate_pct"] >= 60.0 and metrics["trade_pnl_eur"] > 0:
                risk_multiplier = 1.10
                flags.append("trade_quality_positive")
            elif metrics["trade_winrate_pct"] <= 40.0 or metrics["trade_pnl_eur"] < 0:
                risk_multiplier = 0.75
                flags.append("trade_quality_negative")
        elif metrics["trade_open"] > 0:
            flags.append("trade_sample_low")

        if metrics["missed_opportunity"] >= 3 or metrics["missed_rate_pct"] >= 10.0:
            flags.append("filters_may_be_too_strict")

        if metrics["range_events"] >= 10 and metrics["range_breakout_rate_pct"] >= 50.0:
            flags.append("range_breakout_candidate")

        if confidence == "low":
            flags.append("sample_low")
            risk_multiplier = min(risk_multiplier, 1.0)

        return {
            "symbol": symbol,
            "confidence": confidence,
            "bias": bias,
            "risk_multiplier": round(risk_multiplier, 2),
            "flags": flags,
            "metrics": metrics,
        }

    def _confidence(self, metrics: Dict[str, Any]) -> str:
        events = metrics["events"]
        trades = metrics["trade_open"]
        if events >= 100 or trades >= 20:
            return "high"
        if events >= self.min_events or trades >= self.min_trades:
            return "medium"
        return "low"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build profile proposals from strategy learning report.")
    parser.add_argument("--input", type=str, default=DEFAULT_REPORT_PATH, help="Learning report JSON path.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Proposal output JSON path.")
    parser.add_argument("--min-events", type=int, default=30, help="Min events for medium confidence.")
    parser.add_argument("--min-trades", type=int, default=5, help="Min trades for trade-quality confidence.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    proposer = StrategyProfileProposer(min_events=args.min_events, min_trades=args.min_trades)
    proposals = proposer.build_proposals(load_json(args.input))
    write_json(args.output, proposals)
    print(json.dumps({
        "n_symbols": proposals.get("n_symbols", 0),
        "output_path": args.output,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
