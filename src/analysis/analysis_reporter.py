import os
import json
import datetime
from typing import List, Dict


class AnalysisReporter:
    """
    Responsible for writing analysis output to JSON files:
    - analysis/latest_report.json
    - analysis/<timestamp>.json
    - analysis/coins/<symbol>.json
    """

    def __init__(self, base_dir: str = "analysis"):
        self.base_dir = base_dir
        self.coins_dir = os.path.join(base_dir, "coins")

        # Ensure folder structure exists
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.coins_dir, exist_ok=True)

    def write_full_report(self, reports: List[Dict]):
        """
        Writes:
        - analysis/latest_report.json
        - analysis/<timestamp>.json
        """
        if not reports:
            return

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname_latest = os.path.join(self.base_dir, "latest_report.json")
        fname_timestamped = os.path.join(self.base_dir, f"{timestamp}.json")

        data = {
            "generated_utc": timestamp,
            "coins": reports,
        }

        # Write both versions
        with open(fname_latest, "w") as f:
            json.dump(data, f, indent=4)

        with open(fname_timestamped, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[AnalysisReporter] Report written → {fname_latest}")
        print(f"[AnalysisReporter] Report written → {fname_timestamped}")

    def write_per_coin_reports(self, reports: List[Dict]):
        """
        Writes per coin:
        analysis/coins/SYMBOL.json
        """
        for rep in reports:
            symbol = rep.get("symbol")
            if not symbol:
                continue

            fname = os.path.join(self.coins_dir, f"{symbol}.json")
            with open(fname, "w") as f:
                json.dump(rep, f, indent=4)

        print(f"[AnalysisReporter] Wrote per-coin JSON files under {self.coins_dir}")
