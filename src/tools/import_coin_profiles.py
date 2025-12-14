import os
import json
import time

from src.database_manager.database_manager import DatabaseManager

PROFILE_DIR = "/home/redbot/redbot/analysis/coin_profiles"
STRATEGY_NAME = "trend_4h"


def main():
    db = DatabaseManager()
    db.create_tables()  # zorgt dat coin_profiles bestaat

    imported = 0

    for fname in os.listdir(PROFILE_DIR):
        if not fname.endswith(".json"):
            continue

        symbol = fname.replace(".json", "")
        path = os.path.join(PROFILE_DIR, fname)

        try:
            with open(path, "r") as f:
                profile = json.load(f)

            db.upsert_coin_profile(
                symbol=symbol,
                strategy_name=STRATEGY_NAME,
                profile=profile,
                updated_ts=int(time.time() * 1000),
                source="import_json"
            )

            imported += 1
            print(f"[OK] Imported profile for {symbol}")

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    print(f"\nDone. Imported {imported} coin profiles.")


if __name__ == "__main__":
    main()
