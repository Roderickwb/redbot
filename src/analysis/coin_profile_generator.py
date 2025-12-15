import os
import json
import logging
import time
from typing import Dict, Any
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE


# Map met per-coin analyse JSON’s (gemaakt door analysis_job / analysis_reporter)
ANALYSIS_DIR = os.path.join("analysis", "coins")

# Map waar we de coin_profiles wegschrijven
OUTPUT_DIR = os.path.join("analysis", "coin_profiles")

logger = logging.getLogger("coin_profile_generator")


def load_analysis_files() -> Dict[str, Any]:
    """
    Leest alle <COIN>.json bestanden uit analysis/coins/
    en retourneert een dict {symbol: analysis_json}.
    """
    results: Dict[str, Any] = {}

    if not os.path.exists(ANALYSIS_DIR):
        logger.warning("[coin_profile] Analysis directory ontbreekt: %s", ANALYSIS_DIR)
        return results

    for fname in os.listdir(ANALYSIS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ANALYSIS_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            symbol = data.get("symbol")
            if symbol:
                results[symbol] = data
        except Exception as e:
            logger.warning("[coin_profile] Kon %s niet lezen: %s", fname, e)

    return results

def write_profiles_to_db(profiles: Dict[str, Dict[str, Any]], strategy_name: str = "trend_4h"):
    db = DatabaseManager(db_path=DB_FILE)

    for sym, prof in profiles.items():
        profile_json = json.dumps(prof, ensure_ascii=False)

        risk_mult = float(prof.get("risk_multiplier", 1.0))
        bias = prof.get("bias", "neutral")
        n_trades = int(prof.get("n_trades", 0))
        expectancy_r = float(prof.get("expectancy_R", 0.0))

        updated_ts = int(time.time() * 1000)

        db.execute_query(
            """
            INSERT INTO coin_profiles (
                symbol,
                strategy_name,
                risk_multiplier,
                bias,
                n_trades,
                expectancy_r,
                source,
                updated_ts,
                profile_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, strategy_name) DO UPDATE SET
                risk_multiplier=excluded.risk_multiplier,
                bias=excluded.bias,
                n_trades=excluded.n_trades,
                expectancy_r=excluded.expectancy_r,
                source=excluded.source,
                updated_ts=excluded.updated_ts,
                profile_json=excluded.profile_json
            """,
            (
                sym,
                strategy_name,
                risk_mult,
                bias,
                n_trades,
                expectancy_r,
                "derived_trades_daily",
                updated_ts,
                profile_json,
            ),
        )

def derive_profile(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zet een analysis JSON (zoals AAVE-EUR.json, ADA-EUR.json, ...) om
    naar een klein coin_profile dat we later aan GPT en de strategie meegeven.
    """
    tm = analysis.get("trade_metrics", {}) or {}
    gm = analysis.get("gpt_metrics", {}) or {}
    flags = analysis.get("flags", []) or []

    from datetime import datetime, timezone

    n_trades = int(tm.get("n_trades", 0) or 0)
    generated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    symbol = analysis.get("symbol", "UNKNOWN")
    winrate = tm.get("winrate", 0.0)
    expectancy_R = tm.get("expectancy_R", 0.0)
    max_dd = tm.get("max_drawdown_R", 0.0)


    # -------------------------
    # 1) Regime bepalen
    # -------------------------
    if expectancy_R < -0.2:
        regime = "bear"
    elif expectancy_R > 0.2:
        regime = "bull"
    else:
        regime = "range"

    # strength (0–1)
    regime_strength = min(1.0, abs(expectancy_R) / 1.0)

    # -------------------------
    # 2) Long/short edge
    # -------------------------
    long_edge = tm.get("avg_R_long", 0.0)
    short_edge = tm.get("avg_R_short", 0.0)

    if short_edge > long_edge:
        bias = "short_edge"
    elif long_edge > short_edge:
        bias = "long_edge"
    else:
        bias = "neutral"

    # -------------------------
    # 3) Risk multiplier (0–1) – sizing richting
    # -------------------------
    if max_dd < -3.0:
        risk_multiplier = 0.25
    elif max_dd < -2.0:
        risk_multiplier = 0.5
    elif max_dd < -1.0:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 1.0

    # -------------------------
    # 4) HOLD behaviour (GPT)
    # -------------------------
    hold_missed = gm.get("hold_missed_rate", 0.0)
    if hold_missed > 0.7:
        hold_behavior = "too_conservative"
    elif hold_missed < 0.3:
        hold_behavior = "aggressive"
    else:
        hold_behavior = "balanced"

    # -------------------------
    # Final profile
    # -------------------------
    profile = {
        "symbol": symbol,

        "profile_version": "v1",
        "generated_at_utc": generated_at_utc,
        "source": "derived_trades_daily",
        "n_trades": n_trades,

        "market_regime": regime,
        "regime_strength": round(regime_strength, 3),

        "long_edge": round(long_edge, 4),
        "short_edge": round(short_edge, 4),
        "bias": bias,

        "winrate": round(winrate, 3),
        "expectancy_R": round(expectancy_R, 4),
        "max_drawdown_R": round(max_dd, 4),

        "hold_missed_rate": round(hold_missed, 3),
        "hold_behavior": hold_behavior,

        "risk_multiplier": risk_multiplier,

        "flags": flags,
    }

    return profile


def write_profiles(profiles: Dict[str, Dict[str, Any]]):
    """
    Schrijf:
    - per coin een JSON-file in analysis/coin_profiles/<SYMBOL>.json
    - één gecombineerde file analysis/coin_profiles/all_profiles.json
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Per coin
    for sym, prof in profiles.items():
        path = os.path.join(OUTPUT_DIR, f"{sym}.json")
        with open(path, "w") as f:
            json.dump(prof, f, indent=2)

    # All-in-one
    all_path = os.path.join(OUTPUT_DIR, "all_profiles.json")
    with open(all_path, "w") as f:
        json.dump(profiles, f, indent=2)

    logger.info("[coin_profile] %d profiles geschreven naar %s", len(profiles), OUTPUT_DIR)

def generate_coin_profiles():
    print("[coin_profile] Start genereren coin profiles...")
    analyses = load_analysis_files()
    profiles = {}

    for symbol, analysis in analyses.items():
        profiles[symbol] = derive_profile(analysis)

    write_profiles(profiles)
    write_profiles_to_db(profiles, strategy_name="trend_4h")
    print("[coin_profile] Klaar.")

if __name__ == "__main__":
    generate_coin_profiles()
