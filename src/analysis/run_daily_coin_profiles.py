"""Deprecated live coin-profile writer.

This script used to write directly to strategy_name="trend_4h", which can affect
live sizing. Keep it disabled; use strategy_learning_job/strategy_profile_proposer
for proposed profiles and promote through operator approval later.
"""

from __future__ import annotations

import json


def run():
    result = {
        "status": "disabled",
        "reason": "direct live coin profile writes are disabled; use proposed profiles + operator approval",
        "live_effect": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
