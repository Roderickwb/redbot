import os
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import List

from src.analysis.coin_analyzer import CoinReport


class AnalysisReporter:
    """
    Schrijft de output van CoinAnalyzer naar JSON-bestanden:

    - analysis/latest_report.json        → laatste volledige overzicht
    - analysis/<TIMESTAMP>.json         → historisch snapshot
    - analysis/coins/<SYMBOL>.json      → per-coin report (1 coin per file)
    """

    def __init__(self, base_dir: str = "analysis"):
        self.base_dir = base_dir
        self.coins_dir = os.path.join(base_dir, "coins")

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.coins_dir, exist_ok=True)

    def write_reports(self, reports: List[CoinReport]) -> None:
        """
        Neemt een lijst CoinReport objects en schrijft:
        - één global summary (latest + timestamped)
        - per-coin JSON onder analysis/coins
        """

        if not reports:
            print("[AnalysisReporter] Geen reports ontvangen, niets geschreven.")
            return

        now_ms = int(time.time() * 1000)
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # -------------------------
        # 1) Global summary bouwen
        # -------------------------
        summary = {
            "created_ts": now_ms,
            "n_coins": len(reports),
            "coins": {}
        }

        for rep in reports:
            sym = rep.symbol

            tm = asdict(rep.trade_metrics) if rep.trade_metrics else {}
            gm = asdict(rep.gpt_metrics) if rep.gpt_metrics else {}
            flags = rep.flags or []

            # In global summary
            summary["coins"][sym] = {
                "symbol": sym,
                "trade_metrics": tm,
                "gpt_metrics": gm,
                "flags": flags,
            }

            # -------------------------
            # 2) Per-coin JSON schrijven
            # -------------------------
            coin_payload = {
                "symbol": sym,
                "trade_metrics": tm,
                "gpt_metrics": gm,
                "flags": flags,
            }

            coin_path = os.path.join(self.coins_dir, f"{sym}.json")
            try:
                with open(coin_path, "w") as f:
                    json.dump(coin_payload, f, indent=2)
                print(f"[AnalysisReporter] Coin report geschreven → {coin_path}")
            except Exception as e:
                print(f"[AnalysisReporter] Kon coin report voor {sym} niet schrijven: {e}")

        # -------------------------
        # 3) Global latest + historisch
        # -------------------------
        latest_path = os.path.join(self.base_dir, "latest_report.json")
        ts_path = os.path.join(self.base_dir, f"{ts_str}.json")

        try:
            with open(latest_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[AnalysisReporter] Report written → {latest_path}")
        except Exception as e:
            print(f"[AnalysisReporter] Kon latest_report.json niet schrijven: {e}")

        try:
            with open(ts_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[AnalysisReporter] Report written → {ts_path}")
        except Exception as e:
            print(f"[AnalysisReporter] Kon timestamp report niet schrijven: {e}")
