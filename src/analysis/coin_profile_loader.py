# ============================================================
# src/analysis/coin_profile_loader.py
# Kleine helper om coin_profile uit de DB te lezen
# ============================================================
from __future__ import annotations

from typing import Dict

from src.database_manager.database_manager import DatabaseManager


def load_coin_profile(
    db: DatabaseManager,
    symbol: str,
    strategy_name: str = "trend_4h",
) -> Dict:
    """
    Haalt het laatste profiel op voor (symbol, strategy_name).
    Geeft een dict terug met o.a. risk_multiplier en bias.
    """
    try:
        rows = db.execute_query(
            """
            SELECT
                risk_multiplier, bias, notes, flags_text,
                n_trades, winrate, expectancy_R, max_drawdown_R
            FROM coin_profile
            WHERE symbol = ? AND strategy_name = ?
            ORDER BY updated_ts DESC
            LIMIT 1;
            """,
            (symbol, strategy_name),
        )
    except Exception:
        return {}

    if not rows:
        return {}

    (
        risk_mult,
        bias,
        notes,
        flags_text,
        n_trades,
        winrate,
        expectancy_R,
        max_dd,
    ) = rows[0]

    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "risk_multiplier": float(risk_mult or 1.0),
        "bias": bias or "neutral",
        "notes": notes or "",
        "flags": (flags_text or "").split("|") if flags_text else [],
        "n_trades": int(n_trades or 0),
        "winrate": float(winrate or 0.0),
        "expectancy_R": float(expectancy_R or 0.0),
        "max_drawdown_R": float(max_dd or 0.0),
    }

import json

def load_coin_profile_json(
    db: DatabaseManager,
    symbol: str,
    strategy_name: str = "trend_4h",
) -> Dict:
    """
    Laadt het coin_profile zoals het écht bedoeld is: 1-op-1 uit coin_profiles.profile_json
    """
    rows = db.execute_query(
        """
        SELECT profile_json
        FROM coin_profiles
        WHERE symbol = ? AND strategy_name = ?
        ORDER BY updated_ts DESC
        LIMIT 1
        """,
        (symbol, strategy_name),
    )

    if not rows or not rows[0] or not rows[0][0]:
        return {}

    raw_json = rows[0][0]
    try:
        return json.loads(raw_json)
    except Exception:
        return {}
