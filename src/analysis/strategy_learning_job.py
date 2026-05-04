# ============================================================
# src/analysis/strategy_learning_job.py
# ============================================================
"""
Autonomous learning job for strategy_events.

One run:
- labels pending strategy_events when enough candles are available
- builds the rolling-window strategy event report
- writes the latest report to disk

This job does not change trading rules or coin profiles yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from src.analysis.strategy_event_outcome_labeler import StrategyEventOutcomeLabeler
from src.analysis.strategy_event_reporter import StrategyEventReporter
from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager

logger = logging.getLogger("strategy_learning_job")


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "strategy_events")
DEFAULT_LATEST_FILE = "latest_strategy_learning_report.json"


def _parse_windows(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_strategy_learning_job(
    apply_labels: bool = True,
    label_limit: int = 1000,
    report_limit: int = 5000,
    windows: Optional[list[int]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    started_ts = int(time.time() * 1000)
    db = DatabaseManager(db_path=DB_FILE)

    try:
        labeler = StrategyEventOutcomeLabeler(db=db)
        label_stats = labeler.label_pending_events(apply=apply_labels, limit=label_limit)

        reporter = StrategyEventReporter(db=db)
        report = reporter.build_report(limit=report_limit, windows=windows or [30, 100, 500])

        payload = {
            "created_ts": int(time.time() * 1000),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "job": {
                "apply_labels": apply_labels,
                "label_limit": label_limit,
                "report_limit": report_limit,
                "windows": windows or [30, 100, 500],
                "duration_sec": round((int(time.time() * 1000) - started_ts) / 1000.0, 3),
            },
            "label_stats": label_stats,
            "report": report,
        }

        os.makedirs(output_dir, exist_ok=True)
        latest_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("[strategy_learning_job] wrote %s", latest_path)
        return {
            "label_stats": label_stats,
            "report_loaded_events": report.get("meta", {}).get("loaded_labeled_events", 0),
            "output_path": latest_path,
        }
    finally:
        db.close_connection()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run autonomous strategy-event learning job.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write event labels; still writes report file.")
    parser.add_argument("--label-limit", type=int, default=1000, help="Max pending events to label.")
    parser.add_argument("--report-limit", type=int, default=5000, help="Max labeled events to read for report.")
    parser.add_argument("--windows", type=str, default="30,100,500", help="Comma-separated report windows.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for report output.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = run_strategy_learning_job(
        apply_labels=not args.dry_run,
        label_limit=args.label_limit,
        report_limit=args.report_limit,
        windows=_parse_windows(args.windows),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
