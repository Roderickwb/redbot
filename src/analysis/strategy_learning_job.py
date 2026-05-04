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

from dotenv import load_dotenv

from src.analysis.strategy_event_outcome_labeler import StrategyEventOutcomeLabeler
from src.analysis.strategy_profile_proposer import StrategyProfileProposer
from src.analysis.strategy_event_reporter import StrategyEventReporter
from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager
from src.notifier.telegram_notifier import TelegramNotifier

logger = logging.getLogger("strategy_learning_job")


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "strategy_events")
DEFAULT_LATEST_FILE = "latest_strategy_learning_report.json"
DEFAULT_PROPOSALS_FILE = "latest_strategy_profile_proposals.json"
DAILY_SUMMARY_SENT_FILE = ".daily_learning_summary_sent"


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
    notify_daily_summary: bool = False,
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

        proposer = StrategyProfileProposer()
        proposals = proposer.build_proposals(payload)
        proposals_path = os.path.join(output_dir, DEFAULT_PROPOSALS_FILE)
        with open(proposals_path, "w", encoding="utf-8") as f:
            json.dump(proposals, f, indent=2, ensure_ascii=False)
        profiles_written = 0
        if apply_labels:
            profiles_written = proposer.write_coin_profiles_to_db(payload, db=db)

        notification_sent = False
        if notify_daily_summary:
            notification_sent = _send_daily_summary_if_due(
                payload=payload,
                proposals=proposals,
                output_dir=output_dir,
            )

        logger.info("[strategy_learning_job] wrote %s", latest_path)
        return {
            "label_stats": label_stats,
            "report_loaded_events": report.get("meta", {}).get("loaded_labeled_events", 0),
            "output_path": latest_path,
            "proposals_path": proposals_path,
            "proposal_symbols": proposals.get("n_symbols", 0),
            "profiles_written": profiles_written,
            "notification_sent": notification_sent,
        }
    finally:
        db.close_connection()


def _send_daily_summary_if_due(payload: dict, proposals: dict, output_dir: str) -> bool:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    if not (now.hour == 17 and 25 <= now.minute <= 39):
        return False

    os.makedirs(output_dir, exist_ok=True)
    sent_path = os.path.join(output_dir, DAILY_SUMMARY_SENT_FILE)
    if os.path.exists(sent_path):
        try:
            with open(sent_path, "r", encoding="utf-8") as f:
                if f.read().strip() == today:
                    return False
        except Exception:
            pass

    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.info("[strategy_learning_job] Telegram not configured; skip daily summary.")
        return False

    msg = _format_daily_summary(payload, proposals)
    TelegramNotifier(token, chat_id).safe_send(msg)

    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(today)
    return True


def _format_daily_summary(payload: dict, proposals: dict) -> str:
    label_stats = payload.get("label_stats", {}) or {}
    report = payload.get("report", {}) or {}
    totals = report.get("totals", {}) or {}
    summary = proposals.get("summary", {}) or {}

    def names(key: str, max_items: int = 5) -> str:
        rows = summary.get(key, []) or []
        values = [str(row.get("symbol")) for row in rows[:max_items] if row.get("symbol")]
        return ", ".join(values) if values else "-"

    return (
        "Learning Summary 17:30\n"
        f"Events labeled: +{label_stats.get('updated', 0)} | total={report.get('meta', {}).get('loaded_labeled_events', 0)}\n"
        f"Skips: {totals.get('skips', 0)} | missed={totals.get('missed_opportunity', 0)} | protected={totals.get('skip_protected', 0)}\n"
        f"Trades: {totals.get('trade_open', 0)} | winrate={totals.get('trade_winrate_pct', 0)}% | pnl={totals.get('trade_pnl_eur', 0)}\n"
        f"Risk down: {names('risk_down_symbols')}\n"
        f"Filter review: {names('filter_review_symbols')}\n"
        f"Range candidates: {names('range_breakout_candidates')}"
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run autonomous strategy-event learning job.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write event labels; still writes report file.")
    parser.add_argument("--label-limit", type=int, default=1000, help="Max pending events to label.")
    parser.add_argument("--report-limit", type=int, default=5000, help="Max labeled events to read for report.")
    parser.add_argument("--windows", type=str, default="30,100,500", help="Comma-separated report windows.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for report output.")
    parser.add_argument("--notify-daily-summary", action="store_true", help="Send Telegram summary around 17:30.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = run_strategy_learning_job(
        apply_labels=not args.dry_run,
        label_limit=args.label_limit,
        report_limit=args.report_limit,
        windows=_parse_windows(args.windows),
        output_dir=args.output_dir,
        notify_daily_summary=args.notify_daily_summary,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
