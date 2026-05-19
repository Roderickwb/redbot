"""Run the live coin-profile refresh used by GPT context.

This updates strategy_name="trend_4h" in coin_profiles, matching the existing
hourly learning flow. It refreshes GPT context only; it does not approve
experiments, enable live enforcement, or change strategy code.
"""

from __future__ import annotations

import json

from src.analysis.coin_profile_generator import generate_coin_profiles


def run():
    profiles_written = generate_coin_profiles()
    result = {
        "status": "OK",
        "profiles_written": profiles_written,
        "strategy_name": "trend_4h",
        "live_effect": "gpt_context_refresh_only",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()