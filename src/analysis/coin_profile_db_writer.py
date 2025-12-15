import json
import time
from typing import Dict, Any
from src.database_manager.database_manager import DatabaseManager

def upsert_coin_profile(
    db: DatabaseManager,
    symbol: str,
    strategy_name: str,
    profile: Dict[str, Any],
    source: str = "derived_trades_daily",
):
    updated_ts = int(time.time() * 1000)

    risk_multiplier = float(profile.get("risk_multiplier", 1.0) or 1.0)
    bias = str(profile.get("bias", "neutral") or "neutral")
    n_trades = int(profile.get("n_trades", 0) or 0)
    expectancy_r = float(profile.get("expectancy_R", 0.0) or 0.0)

    profile_json = json.dumps(profile, separators=(",", ":"), ensure_ascii=False)

    # IMPORTANT:
    # Dit werkt alleen als je UNIQUE constraint hebt op (symbol, strategy_name).
    sql = """
    INSERT INTO coin_profiles (
        symbol, strategy_name,
        risk_multiplier, bias, n_trades, expectancy_r,
        source, updated_ts, profile_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol, strategy_name) DO UPDATE SET
        risk_multiplier=excluded.risk_multiplier,
        bias=excluded.bias,
        n_trades=excluded.n_trades,
        expectancy_r=excluded.expectancy_r,
        source=excluded.source,
        updated_ts=excluded.updated_ts,
        profile_json=excluded.profile_json
    ;
    """

    db.execute_query(sql, (
        symbol, strategy_name,
        risk_multiplier, bias, n_trades, expectancy_r,
        source, updated_ts, profile_json
    ))
