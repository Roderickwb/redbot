# ============================================================
# src/analysis/coin_profile_loader.py
# Single source of truth: coin_profiles.profile_json
# ============================================================

from __future__ import annotations

from typing import Dict
import json

from src.database_manager.database_manager import DatabaseManager


def load_coin_profile_json(
    db: DatabaseManager,
    symbol: str,
    strategy_name: str = "trend_4h",
) -> Dict:
    """
    Laadt het coin_profile 1-op-1 uit coin_profiles.profile_json.
    Dit is de enige waarheid (geen losse kolommen zoals notes/flags_text/winrate).
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
