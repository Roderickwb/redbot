# ============================================================
# src/analysis/market_regime.py
# ============================================================
"""
Market regime layer.

Builds a compact market-wide context from Kraken candles. This is intentionally
read-only: it informs GPT/risk analysis, but does not directly block trades.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Iterable, Optional

import pandas as pd

from src.config.config import DB_FILE, yaml_config
from src.database_manager.database_manager import DatabaseManager


DEFAULT_ANCHORS = ["XBT-EUR", "ETH-EUR"]
DEFAULT_OUTPUT_DIR = os.path.join("analysis", "market_regime")
DEFAULT_LATEST_FILE = "latest_market_regime.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pct(part: int, total: int) -> float:
    return round(part / total * 100.0, 2) if total else 0.0


class MarketRegimeAnalyzer:
    def __init__(self, db: Optional[DatabaseManager] = None, config: Optional[dict] = None):
        self.db = db or DatabaseManager(db_path=DB_FILE)
        self.config = config or {}
        self._cache: dict[str, Any] = {
            "expires_at": 0.0,
            "states": None,
            "anchors": None,
            "interval": None,
        }

    def build_regime(
        self,
        symbols: Optional[list[str]] = None,
        current_symbol: Optional[str] = None,
        interval: str = "4h",
        limit: int = 80,
    ) -> dict:
        cfg = self._regime_cfg()
        if not cfg.get("enabled", True):
            return self._disabled_payload(current_symbol=current_symbol)

        cache_ttl_sec = float(cfg.get("cache_ttl_sec", 300))
        cache_key = self._cache_key(symbols=symbols, interval=interval, limit=limit)
        now = time.time()
        if self._cache.get("key") == cache_key and now < float(self._cache.get("expires_at", 0)):
            cached_states = self._cache.get("states") or {}
            cached_anchors = self._cache.get("anchors") or DEFAULT_ANCHORS
            cached_interval = self._cache.get("interval") or interval
            return self._summarize(
                states=cached_states,
                anchors=cached_anchors,
                current_symbol=current_symbol,
                interval=cached_interval,
            )

        symbols = symbols or self._default_symbols()
        anchors = list(cfg.get("anchor_symbols") or DEFAULT_ANCHORS)
        tracked = sorted(set(symbols + anchors + ([current_symbol] if current_symbol else [])))

        states = {}
        for symbol in tracked:
            state = self._symbol_state(symbol=symbol, interval=interval, limit=limit)
            if state:
                states[symbol] = state

        payload = self._summarize(
            states=states,
            anchors=anchors,
            current_symbol=current_symbol,
            interval=interval,
        )
        self._cache = {
            "key": cache_key,
            "expires_at": now + cache_ttl_sec,
            "states": states,
            "anchors": anchors,
            "interval": interval,
        }
        return payload

    def _regime_cfg(self) -> dict:
        cfg = (yaml_config.get("trend_strategy_4h", {}) or {}).get("market_regime", {}) or {}
        merged = dict(self.config)
        merged.update(cfg)
        return merged

    def _default_symbols(self) -> list[str]:
        kraken_cfg = yaml_config.get("kraken", {}) or {}
        return list(kraken_cfg.get("pairs", []) or DEFAULT_ANCHORS)

    def _symbol_state(self, symbol: str, interval: str, limit: int) -> Optional[dict]:
        rows = self.db.execute_query(
            """
            SELECT timestamp, close
            FROM candles_kraken
            WHERE market=? AND interval=?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, interval, int(limit)),
        )
        if not rows or len(rows) < 30:
            return None

        df = pd.DataFrame(rows, columns=["timestamp", "close"]).sort_values("timestamp")
        close = df["close"].astype(float)
        latest = _safe_float(close.iloc[-1])
        if latest <= 0:
            return None

        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ret_24h = self._return_pct(close, bars=6)
        ret_72h = self._return_pct(close, bars=18)
        ema20_slope = _safe_float(ema20.iloc[-1] - ema20.iloc[-4]) if len(ema20) >= 4 else 0.0

        trend = "range"
        score = 0
        if latest > ema20.iloc[-1] > ema50.iloc[-1]:
            trend = "bull"
            score += 1
        elif latest < ema20.iloc[-1] < ema50.iloc[-1]:
            trend = "bear"
            score -= 1

        if ret_24h > 1.0:
            score += 1
        elif ret_24h < -1.0:
            score -= 1

        if ema20_slope > 0:
            score += 1
        elif ema20_slope < 0:
            score -= 1

        if score >= 2:
            state = "bull"
        elif score <= -2:
            state = "bear"
        else:
            state = "range"

        return {
            "symbol": symbol,
            "state": state,
            "trend": trend,
            "score": score,
            "close": round(latest, 8),
            "ema20": round(_safe_float(ema20.iloc[-1]), 8),
            "ema50": round(_safe_float(ema50.iloc[-1]), 8),
            "ret_24h_pct": round(ret_24h, 3),
            "ret_72h_pct": round(ret_72h, 3),
            "latest_ts": int(df["timestamp"].iloc[-1]),
        }

    def _return_pct(self, close: pd.Series, bars: int) -> float:
        if len(close) <= bars:
            return 0.0
        prev = _safe_float(close.iloc[-(bars + 1)])
        latest = _safe_float(close.iloc[-1])
        if prev <= 0:
            return 0.0
        return (latest - prev) / prev * 100.0

    def _summarize(
        self,
        states: dict[str, dict],
        anchors: list[str],
        current_symbol: Optional[str],
        interval: str,
    ) -> dict:
        values = list(states.values())
        total = len(values)
        bull = sum(1 for s in values if s.get("state") == "bull")
        bear = sum(1 for s in values if s.get("state") == "bear")
        range_count = sum(1 for s in values if s.get("state") == "range")
        anchor_states = {symbol: states[symbol] for symbol in anchors if symbol in states}
        anchor_score = sum(int(s.get("score", 0)) for s in anchor_states.values())
        current_state = states.get(current_symbol or "")

        bull_pct = _pct(bull, total)
        bear_pct = _pct(bear, total)
        range_pct = _pct(range_count, total)

        regime = "mixed"
        risk_mode = "normal"
        directional_bias = "neutral"
        risk_multiplier = 0.85
        flags: list[str] = []

        if total < 5:
            regime = "unknown"
            risk_mode = "normal"
            risk_multiplier = 0.85
            flags.append("MARKET_REGIME_LOW_DATA")
        elif anchor_score <= -2 or bear_pct >= 50:
            regime = "risk_off"
            risk_mode = "defensive"
            directional_bias = "short_or_cash"
            risk_multiplier = 0.5
            flags.append("MARKET_RISK_OFF")
        elif anchor_score >= 2 and bull_pct >= 45 and bear_pct < 35:
            regime = "risk_on"
            risk_mode = "normal"
            directional_bias = "long"
            risk_multiplier = 1.0
            flags.append("MARKET_RISK_ON")
        elif range_pct >= 50:
            regime = "chop"
            risk_mode = "cautious"
            risk_multiplier = 0.7
            flags.append("MARKET_CHOP")

        if current_state:
            if regime == "risk_off" and current_state.get("state") == "bull":
                flags.append("COIN_STRONG_AGAINST_WEAK_MARKET")
            elif regime == "risk_on" and current_state.get("state") == "bear":
                flags.append("COIN_WEAK_AGAINST_STRONG_MARKET")

        return {
            "source": "candles_kraken",
            "interval": interval,
            "regime": regime,
            "risk_mode": risk_mode,
            "directional_bias": directional_bias,
            "risk_multiplier": risk_multiplier,
            "breadth": {
                "symbols": total,
                "bull_pct": bull_pct,
                "bear_pct": bear_pct,
                "range_pct": range_pct,
            },
            "anchors": anchor_states,
            "current_symbol": current_state or {},
            "flags": flags,
        }

    def _cache_key(
        self,
        symbols: Optional[list[str]],
        interval: str,
        limit: int,
    ) -> str:
        values = ",".join(sorted(symbols or []))
        return f"{values}|{interval}|{limit}"

    def _disabled_payload(self, current_symbol: Optional[str]) -> dict:
        return {
            "source": "disabled",
            "regime": "unknown",
            "risk_mode": "normal",
            "directional_bias": "neutral",
            "risk_multiplier": 1.0,
            "breadth": {"symbols": 0, "bull_pct": 0.0, "bear_pct": 0.0, "range_pct": 0.0},
            "anchors": {},
            "current_symbol": {"symbol": current_symbol} if current_symbol else {},
            "flags": ["MARKET_REGIME_DISABLED"],
        }


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build current market regime report.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional current symbol focus.")
    parser.add_argument("--interval", type=str, default="4h", help="Candle interval to use.")
    parser.add_argument("--limit", type=int, default=80, help="Candles per symbol.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db = DatabaseManager(db_path=DB_FILE)
    try:
        analyzer = MarketRegimeAnalyzer(db=db)
        report = analyzer.build_regime(
            current_symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
        )
    finally:
        db.close_connection()

    output_path = os.path.join(args.output_dir, DEFAULT_LATEST_FILE)
    write_json(output_path, report)
    result = {
        "regime": report.get("regime"),
        "risk_mode": report.get("risk_mode"),
        "directional_bias": report.get("directional_bias"),
        "risk_multiplier": report.get("risk_multiplier"),
        "breadth": report.get("breadth"),
        "flags": report.get("flags"),
        "output_path": output_path,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
