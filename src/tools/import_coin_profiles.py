"""Disabled legacy live coin-profile importer.

This tool used to import JSON profiles directly into strategy_name="trend_4h",
which can affect live sizing. Keep it disabled until operator approval/promotion
flow exists.
"""

from __future__ import annotations

import json


def main():
    result = {
        "status": "disabled",
        "reason": "direct live coin profile imports are disabled; use proposed profiles + operator approval",
        "live_effect": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
