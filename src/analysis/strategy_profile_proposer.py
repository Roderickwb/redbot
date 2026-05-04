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
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager

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
            "summary": self._build_summary(proposals),
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
        risk_multiplier = self._risk_multiplier(metrics)
        hold_behavior = self._hold_behavior(metrics)
        bias = "neutral"
        flags = []

        if metrics["trade_open"] >= self.min_trades:
            if risk_multiplier == 1.0 and metrics["trade_winrate_pct"] >= 60.0 and metrics["trade_pnl_eur"] > 0:
                flags.append("trade_quality_positive")
            elif risk_multiplier < 1.0:
                flags.append("trade_quality_negative")
        elif metrics["trade_open"] > 0:
            flags.append("trade_sample_low")

        if metrics["missed_opportunity"] >= 3 or metrics["missed_rate_pct"] >= 10.0:
            flags.append("filters_may_be_too_strict")
        if hold_behavior == "too_conservative":
            flags.append("CONSERVATIVE_HOLD")
        elif hold_behavior == "hold_ok":
            flags.append("HOLD_OK")

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
            "hold_behavior": hold_behavior,
            "flags": flags,
            "metrics": metrics,
        }

    def _risk_multiplier(self, metrics: Dict[str, Any]) -> float:
        trades = int(metrics.get("trade_open", 0) or 0)
        winrate = float(metrics.get("trade_winrate_pct", 0.0) or 0.0)
        pnl = float(metrics.get("trade_pnl_eur", 0.0) or 0.0)

        if trades < self.min_trades:
            return 1.0
        if trades >= 10 and winrate <= 30.0 and pnl < 0:
            return 0.5
        if winrate <= 40.0 or pnl < 0:
            return 0.75
        return 1.0

    def _hold_behavior(self, metrics: Dict[str, Any]) -> str:
        events = int(metrics.get("events", 0) or 0)
        missed = int(metrics.get("missed_opportunity", 0) or 0)
        missed_rate = float(metrics.get("missed_rate_pct", 0.0) or 0.0)

        if events < self.min_events:
            return "unknown"
        if missed >= 3 or missed_rate >= 10.0:
            return "too_conservative"
        if missed == 0 and missed_rate <= 2.0:
            return "hold_ok"
        return "balanced"

    def build_coin_profiles(self, learning_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        proposals_payload = self.build_proposals(learning_payload)
        profiles = {}
        for symbol, proposal in proposals_payload.get("proposals", {}).items():
            profiles[symbol] = self._proposal_to_coin_profile(proposal)
        return profiles

    def write_coin_profiles_to_db(
        self,
        learning_payload: Dict[str, Any],
        db: Optional[DatabaseManager] = None,
        strategy_name: str = "trend_4h",
    ) -> int:
        local_db = db is None
        if db is None:
            db = DatabaseManager(db_path=DB_FILE)

        profiles = self.build_coin_profiles(learning_payload)
        updated_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        for symbol, profile in profiles.items():
            db.upsert_coin_profile(
                symbol=symbol,
                strategy_name=strategy_name,
                profile=profile,
                updated_ts=updated_ts,
                source="strategy_events_learning",
            )

        if local_db:
            db.close_connection()
        return len(profiles)

    def _proposal_to_coin_profile(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        metrics = proposal.get("metrics", {}) or {}
        flags = list(proposal.get("flags", []) or [])

        mapped_flags = []
        if "trade_quality_negative" in flags:
            mapped_flags.append("DRAWDOWN_RISK")
        if "range_breakout_candidate" in flags:
            mapped_flags.append("RANGE_BREAKOUT_CANDIDATE")
        if "filters_may_be_too_strict" in flags:
            mapped_flags.append("FILTER_REVIEW")
        if "sample_low" in flags:
            mapped_flags.append("SAMPLE_LOW")

        all_flags = flags + [flag for flag in mapped_flags if flag not in flags]
        risk_multiplier = min(1.0, max(0.25, float(proposal.get("risk_multiplier", 1.0) or 1.0)))
        n_trades = int(metrics.get("trade_open", 0) or 0)

        return {
            "symbol": proposal.get("symbol"),
            "profile_version": "learning",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "strategy_events_learning",
            "n_trades": n_trades,
            "market_regime": "range",
            "regime_strength": 0.0,
            "long_edge": 0.0,
            "short_edge": 0.0,
            "bias": proposal.get("bias", "neutral"),
            "winrate": round(float(metrics.get("trade_winrate_pct", 0.0) or 0.0) / 100.0, 3),
            "expectancy_R": 0.0,
            "max_drawdown_R": 0.0,
            "hold_missed_rate": round(float(metrics.get("missed_rate_pct", 0.0) or 0.0) / 100.0, 3),
            "hold_behavior": proposal.get("hold_behavior", "unknown"),
            "risk_multiplier": round(risk_multiplier, 2),
            "flags": all_flags,
            "learning_confidence": proposal.get("confidence"),
            "learning_metrics": metrics,
        }

    def _build_summary(self, proposals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        risk_down = []
        risk_up = []
        filter_review = []
        range_breakout = []
        low_sample = []

        for symbol, proposal in proposals.items():
            flags = set(proposal.get("flags", []))
            row = {
                "symbol": symbol,
                "confidence": proposal.get("confidence"),
                "risk_multiplier": proposal.get("risk_multiplier"),
                "flags": proposal.get("flags", []),
            }

            if proposal.get("risk_multiplier", 1.0) < 1.0:
                risk_down.append(row)
            if proposal.get("risk_multiplier", 1.0) > 1.0:
                risk_up.append(row)
            if "filters_may_be_too_strict" in flags:
                filter_review.append(row)
            if "range_breakout_candidate" in flags:
                range_breakout.append(row)
            if proposal.get("confidence") == "low":
                low_sample.append(row)

        return {
            "risk_down_symbols": risk_down,
            "risk_up_symbols": risk_up,
            "filter_review_symbols": filter_review,
            "range_breakout_candidates": range_breakout,
            "low_sample_symbols": low_sample,
            "actionable_symbols": self._unique_symbols(
                risk_down + risk_up + filter_review + range_breakout
            ),
        }

    def _unique_symbols(self, rows: list[Dict[str, Any]]) -> list[str]:
        seen = set()
        result = []
        for row in rows:
            symbol = row.get("symbol")
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        return result

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
    parser.add_argument("--write-db", action="store_true", help="Write learning profiles to coin_profiles.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    proposer = StrategyProfileProposer(min_events=args.min_events, min_trades=args.min_trades)
    proposals = proposer.build_proposals(load_json(args.input))
    write_json(args.output, proposals)
    profiles_written = 0
    if args.write_db:
        profiles_written = proposer.write_coin_profiles_to_db(load_json(args.input))
    print(json.dumps({
        "n_symbols": proposals.get("n_symbols", 0),
        "output_path": args.output,
        "profiles_written": profiles_written,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
