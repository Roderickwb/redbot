# ============================================================
# src/analysis/daily_analysis_job.py
# ============================================================
"""
Daily analysis orchestrator.

Runs the autonomous analysis stack in a fixed order and optionally sends one
combined advisor message to Telegram. This keeps cron simple and prevents five
separate daily jobs from drifting out of sync.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from src.analysis.bot_advisor import (
    BotAdvisor,
    DEFAULT_LATEST_FILE as ADVISOR_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as ADVISOR_OUTPUT_DIR,
    send_telegram as send_advice_telegram,
    sync_registry as sync_advice_registry,
    write_json as write_advisor_json,
)
from src.analysis.bot_alerts_reporter import (
    BotAlertsReporter,
    DEFAULT_LATEST_FILE as ALERTS_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as ALERTS_OUTPUT_DIR,
    send_telegram_once_per_day,
    write_json as write_alerts_json,
)
from src.analysis.chart_vision_reporter import (
    ChartVisionReporter,
    DEFAULT_LATEST_FILE as CHART_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as CHART_OUTPUT_DIR,
    write_json as write_chart_json,
)
from src.analysis.daily_control_report import run_daily_control_report
from src.analysis.gpt_decision_reporter import (
    GptDecisionReporter,
    DEFAULT_LATEST_FILE as GPT_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as GPT_OUTPUT_DIR,
    write_json as write_gpt_json,
)
from src.analysis.experiment_planner import run_experiment_planner, send_experiment_digest
from src.analysis.market_regime import (
    MarketRegimeAnalyzer,
    DEFAULT_LATEST_FILE as REGIME_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as REGIME_OUTPUT_DIR,
    write_json as write_regime_json,
)
from src.analysis.ml_training_dataset import (
    MlTrainingDatasetBuilder,
    DEFAULT_OUTPUT_DIR as ML_DATASET_OUTPUT_DIR,
    write_dataset as write_ml_dataset,
)
from src.analysis.ml_edge_model import (
    MlEdgeModel,
    DEFAULT_LATEST_FILE as ML_EDGE_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as ML_EDGE_OUTPUT_DIR,
    write_json as write_ml_edge_json,
)
from src.analysis.opportunity_reporter import (
    OpportunityReporter,
    DEFAULT_LATEST_FILE as OPPORTUNITY_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as OPPORTUNITY_OUTPUT_DIR,
    write_json as write_opportunity_json,
)
from src.analysis.promotion_gate import run_promotion_gate
from src.analysis.recommendation_registry import RecommendationRegistry
from src.analysis.shadow_model_evaluator import (
    ShadowModelEvaluator,
    DEFAULT_LATEST_FILE as SHADOW_LATEST_FILE,
    DEFAULT_OUTPUT_DIR as SHADOW_OUTPUT_DIR,
    write_json as write_shadow_json,
)
from src.analysis.shadow_experiment_runner import run_shadow_experiment_runner
from src.analysis.strategy_learning_job import run_strategy_learning_job
from src.config.config import DB_FILE
from src.database_manager.database_manager import DatabaseManager


DEFAULT_OUTPUT_DIR = os.path.join("analysis", "daily")
DEFAULT_LATEST_FILE = "latest_daily_analysis_job.json"


def _parse_windows(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_step(fn: Callable[[], dict]) -> dict:
    started = time.time()
    try:
        result = fn()
        return {
            "status": "ok",
            "duration_sec": round(time.time() - started, 3),
            "result": result,
        }
    except Exception as e:
        return {
            "status": "failed",
            "duration_sec": round(time.time() - started, 3),
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
        }


def _build_gpt_report(limit: int) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = GptDecisionReporter(db=db).build_report(limit=limit)
    finally:
        db.close_connection()

    output_path = os.path.join(GPT_OUTPUT_DIR, GPT_LATEST_FILE)
    write_gpt_json(output_path, report)
    return {
        "loaded_gpt_decisions": report.get("meta", {}).get("loaded_gpt_decisions", 0),
        "output_path": output_path,
        "zero_conf_pct": report.get("totals", {}).get("zero_conf_pct", 0),
        "cf_avg_r": report.get("totals", {}).get("cf_avg_r", 0),
        "attention_cases": {
            key: len(value)
            for key, value in (report.get("attention_cases") or {}).items()
        },
    }


def _build_chart_report(limit: int, structured_only: bool) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = ChartVisionReporter(db=db).build_report(
            limit=limit,
            structured_only=structured_only,
        )
    finally:
        db.close_connection()

    output_path = os.path.join(CHART_OUTPUT_DIR, CHART_LATEST_FILE)
    write_chart_json(output_path, report)
    return {
        "loaded_events": report.get("meta", {}).get("loaded_events", 0),
        "output_path": output_path,
        "structured_only": structured_only,
        "cf_avg_r": report.get("totals", {}).get("cf_avg_r", 0),
        "attention_cases": {
            key: len(value)
            for key, value in (report.get("attention_cases") or {}).items()
        },
    }


def _build_alerts_report(hours: int, send_health: bool, force_health_send: bool) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = BotAlertsReporter(db=db).build_report(hours=hours)
    finally:
        db.close_connection()

    output_path = os.path.join(ALERTS_OUTPUT_DIR, ALERTS_LATEST_FILE)
    write_alerts_json(output_path, report)
    sent = False
    if send_health or force_health_send:
        sent = send_telegram_once_per_day(report, ALERTS_OUTPUT_DIR, force=force_health_send)

    return {
        "status": report.get("status"),
        "alerts": len(report.get("alerts", [])),
        "output_path": output_path,
        "telegram_sent": sent,
    }


def _build_market_regime_report() -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = MarketRegimeAnalyzer(db=db).build_regime()
    finally:
        db.close_connection()

    output_path = os.path.join(REGIME_OUTPUT_DIR, REGIME_LATEST_FILE)
    write_regime_json(output_path, report)
    return {
        "regime": report.get("regime"),
        "risk_mode": report.get("risk_mode"),
        "directional_bias": report.get("directional_bias"),
        "risk_multiplier": report.get("risk_multiplier"),
        "breadth": report.get("breadth"),
        "flags": report.get("flags"),
        "output_path": output_path,
    }


def _build_ml_training_dataset(limit: int, structured_only: bool) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        payload = MlTrainingDatasetBuilder(db=db).build_dataset(
            limit=limit,
            event_type="gpt_decision",
            structured_only=structured_only,
        )
    finally:
        db.close_connection()

    jsonl_path, summary_path = write_ml_dataset(ML_DATASET_OUTPUT_DIR, payload)
    return {
        "rows": payload.get("meta", {}).get("rows", 0),
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "by_action": payload.get("summary", {}).get("by_action", {}),
        "by_direction": payload.get("summary", {}).get("by_direction", {}),
    }


def _build_shadow_model_report(limit: int) -> dict:
    report = ShadowModelEvaluator().build_report(limit=limit)
    output_path = os.path.join(SHADOW_OUTPUT_DIR, SHADOW_LATEST_FILE)
    write_shadow_json(output_path, report)
    return {
        "loaded_rows": report.get("meta", {}).get("loaded_rows", 0),
        "output_path": output_path,
        "rules": {
            name: {
                "matches": values.get("matches", 0),
                "cf_avg_r": values.get("cf_avg_r", 0),
                "cf_loss_rate_pct": values.get("cf_loss_rate_pct", 0),
            }
            for name, values in (report.get("rules") or {}).items()
        },
        "recommendations": report.get("recommendations", [])[:5],
    }


def _build_ml_edge_model(limit: int) -> dict:
    report = MlEdgeModel().build_report(limit=limit)
    output_path = os.path.join(ML_EDGE_OUTPUT_DIR, ML_EDGE_LATEST_FILE)
    write_ml_edge_json(output_path, report)
    return {
        "loaded_rows": report.get("meta", {}).get("loaded_rows", 0),
        "readiness": report.get("readiness", {}),
        "model_status": (report.get("model") or {}).get("status"),
        "metrics": (report.get("model") or {}).get("metrics", {}),
        "output_path": output_path,
    }


def _build_opportunity_report(limit: int) -> dict:
    db = DatabaseManager(db_path=DB_FILE)
    try:
        report = OpportunityReporter(db=db).build_report(limit=limit)
    finally:
        db.close_connection()

    output_path = os.path.join(OPPORTUNITY_OUTPUT_DIR, OPPORTUNITY_LATEST_FILE)
    write_opportunity_json(output_path, report)
    return {
        "loaded_candidates": report.get("meta", {}).get("loaded_candidates", 0),
        "output_path": output_path,
        "hold_rate_pct": report.get("totals", {}).get("hold_rate_pct", 0),
        "open_rate_pct": report.get("totals", {}).get("open_rate_pct", 0),
        "cf_avg_r": report.get("totals", {}).get("cf_avg_r", 0),
        "attention_cases": {
            key: len(value)
            for key, value in (report.get("attention_cases") or {}).items()
        },
    }


def _build_advisor(
    send_advice: bool,
    cleanup_registry: bool,
    cleanup_missing_count: int,
    cleanup_stale_days: int,
) -> dict:
    advice = BotAdvisor().build_advice()
    output_path = os.path.join(ADVISOR_OUTPUT_DIR, ADVISOR_LATEST_FILE)
    write_advisor_json(output_path, advice)
    registry_summary = sync_advice_registry(advice)
    cleanup_result = None
    if cleanup_registry:
        cleanup_result = RecommendationRegistry().cleanup(
            apply=True,
            missing_count=cleanup_missing_count,
            stale_days=cleanup_stale_days,
        )
        registry_summary = RecommendationRegistry().summary()
    sent = send_advice_telegram(advice) if send_advice else False

    return {
        "status": advice.get("status"),
        "summary": advice.get("summary"),
        "output_path": output_path,
        "registry": {
            "total": registry_summary.get("total", 0),
            "by_status": registry_summary.get("by_status", {}),
            "active": len(registry_summary.get("active", [])),
        },
        "registry_cleanup": cleanup_result,
        "telegram_sent": sent,
    }


def _build_experiment_plan(send_experiments: bool) -> dict:
    report = run_experiment_planner()
    sent = send_experiment_digest(report) if send_experiments else False
    return {
        "status": report.get("meta", {}).get("status"),
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
        "telegram_sent": sent,
    }


def _build_shadow_experiment_results(hours: int) -> dict:
    report = run_shadow_experiment_runner(forward_hours=hours)
    return {
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }


def _build_promotion_gate() -> dict:
    report = run_promotion_gate()
    return {
        "summary": report.get("summary", {}),
        "output_path": report.get("output_path"),
    }


def _build_daily_control(send_control: bool) -> dict:
    report = run_daily_control_report(send=send_control)
    return {
        "status": report.get("status"),
        "blockers": len(report.get("blockers", []) or []),
        "approval_queue": len(report.get("approval_queue", []) or []),
        "next_actions": report.get("next_actions", []),
        "output_path": report.get("output_path"),
        "telegram_sent": report.get("telegram_sent", False),
    }


def run_daily_analysis_job(
    apply_labels: bool = True,
    relabel_existing: bool = False,
    label_limit: int = 1000,
    report_limit: int = 5000,
    windows: Optional[list[int]] = None,
    hours: int = 24,
    send_advice: bool = False,
    send_health: bool = False,
    send_experiments: bool = False,
    send_control: bool = False,
    cleanup_registry: bool = False,
    cleanup_missing_count: int = 2,
    cleanup_stale_days: int = 14,
    force_health_send: bool = False,
    structured_chart_only: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    started = time.time()
    steps: dict[str, dict] = {}

    steps["strategy_learning"] = _run_step(
        lambda: run_strategy_learning_job(
            apply_labels=apply_labels,
            label_limit=label_limit,
            report_limit=report_limit,
            windows=windows or [30, 100, 500],
            notify_daily_summary=False,
            relabel_existing=relabel_existing,
        ),
    )
    steps["gpt_decision_report"] = _run_step(
        lambda: _build_gpt_report(limit=report_limit),
    )
    steps["chart_vision_report"] = _run_step(
        lambda: _build_chart_report(limit=report_limit, structured_only=structured_chart_only),
    )
    steps["bot_alerts"] = _run_step(
        lambda: _build_alerts_report(
            hours=hours,
            send_health=send_health,
            force_health_send=force_health_send,
        ),
    )
    steps["market_regime"] = _run_step(
        lambda: _build_market_regime_report(),
    )
    steps["ml_training_dataset"] = _run_step(
        lambda: _build_ml_training_dataset(
            limit=report_limit,
            structured_only=structured_chart_only,
        ),
    )
    steps["shadow_model_report"] = _run_step(
        lambda: _build_shadow_model_report(limit=report_limit),
    )
    steps["ml_edge_model"] = _run_step(
        lambda: _build_ml_edge_model(limit=report_limit),
    )
    steps["opportunity_report"] = _run_step(
        lambda: _build_opportunity_report(limit=report_limit),
    )
    steps["experiment_plan"] = _run_step(
        lambda: _build_experiment_plan(send_experiments=send_experiments),
    )
    steps["shadow_experiment_results"] = _run_step(
        lambda: _build_shadow_experiment_results(hours=hours),
    )
    steps["promotion_gate"] = _run_step(
        lambda: _build_promotion_gate(),
    )
    steps["bot_advisor"] = _run_step(
        lambda: _build_advisor(
            send_advice=send_advice,
            cleanup_registry=cleanup_registry,
            cleanup_missing_count=cleanup_missing_count,
            cleanup_stale_days=cleanup_stale_days,
        ),
    )

    failed_steps = [name for name, step in steps.items() if step.get("status") != "ok"]
    advisor_result = steps.get("bot_advisor", {}).get("result") or {}
    advisor_status = advisor_result.get("status", "UNKNOWN")
    overall_status = "FAILED" if failed_steps else advisor_status

    payload = {
        "created_ts": int(time.time() * 1000),
        "created_utc": _utc_now(),
        "status": overall_status,
        "failed_steps": failed_steps,
        "duration_sec": round(time.time() - started, 3),
        "config": {
            "apply_labels": apply_labels,
            "relabel_existing": relabel_existing,
            "label_limit": label_limit,
            "report_limit": report_limit,
            "windows": windows or [30, 100, 500],
            "hours": hours,
            "structured_chart_only": structured_chart_only,
            "send_advice": send_advice,
            "send_health": send_health,
            "send_experiments": send_experiments,
            "send_control": send_control,
            "cleanup_registry": cleanup_registry,
            "cleanup_missing_count": cleanup_missing_count,
            "cleanup_stale_days": cleanup_stale_days,
            "force_health_send": force_health_send,
        },
        "steps": steps,
    }

    output_path = os.path.join(output_dir, DEFAULT_LATEST_FILE)
    payload["output_path"] = output_path
    _write_json(output_path, payload)

    steps["daily_control_report"] = _run_step(
        lambda: _build_daily_control(send_control=send_control),
    )
    failed_steps = [name for name, step in steps.items() if step.get("status") != "ok"]
    payload["failed_steps"] = failed_steps
    payload["status"] = "FAILED" if failed_steps else advisor_status
    payload["duration_sec"] = round(time.time() - started, 3)
    _write_json(output_path, payload)
    return payload


def _summary_for_cli(payload: dict) -> dict:
    steps = payload.get("steps", {})
    return {
        "status": payload.get("status"),
        "failed_steps": payload.get("failed_steps", []),
        "duration_sec": payload.get("duration_sec"),
        "output_path": payload.get("output_path"),
        "steps": {
            name: {
                "status": step.get("status"),
                "duration_sec": step.get("duration_sec"),
                "result": step.get("result"),
                "error": step.get("error"),
            }
            for name, step in steps.items()
        },
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full daily bot analysis stack.")
    parser.add_argument("--dry-run-labels", action="store_true", help="Do not write strategy_event labels or coin profiles.")
    parser.add_argument("--relabel", action="store_true", help="Rebuild existing labeled outcomes too.")
    parser.add_argument("--label-limit", type=int, default=1000, help="Max events to label.")
    parser.add_argument("--report-limit", type=int, default=5000, help="Max labeled events to read per report.")
    parser.add_argument("--windows", type=str, default="30,100,500", help="Comma-separated learning report windows.")
    parser.add_argument("--hours", type=int, default=24, help="Bot-alert lookback window.")
    parser.add_argument("--send-advice", action="store_true", help="Send the combined advisor message to Telegram.")
    parser.add_argument("--send-health", action="store_true", help="Also send the health digest once per day.")
    parser.add_argument("--send-experiments", action="store_true", help="Also send experiment planner digest to Telegram.")
    parser.add_argument("--send-control", action="store_true", help="Also send compact daily control report to Telegram.")
    parser.add_argument("--cleanup-registry", action="store_true", help="Archive stale/missing proposed recommendations after advisor sync.")
    parser.add_argument("--cleanup-missing-count", type=int, default=2, help="Archive proposed recommendations missing from this many syncs.")
    parser.add_argument("--cleanup-stale-days", type=int, default=14, help="Archive proposed recommendations not seen for this many days.")
    parser.add_argument("--force-health-send", action="store_true", help="Ignore daily health digest send marker.")
    parser.add_argument("--include-unstructured-chart", action="store_true", help="Include older/fallback chart events without structured scores.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for the daily job summary.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_daily_analysis_job(
        apply_labels=not args.dry_run_labels,
        relabel_existing=args.relabel,
        label_limit=args.label_limit,
        report_limit=args.report_limit,
        windows=_parse_windows(args.windows),
        hours=args.hours,
        send_advice=args.send_advice,
        send_health=args.send_health,
        send_experiments=args.send_experiments,
        send_control=args.send_control,
        cleanup_registry=args.cleanup_registry,
        cleanup_missing_count=args.cleanup_missing_count,
        cleanup_stale_days=args.cleanup_stale_days,
        force_health_send=args.force_health_send,
        structured_chart_only=not args.include_unstructured_chart,
        output_dir=args.output_dir,
    )
    print(json.dumps(_summary_for_cli(result), indent=2, ensure_ascii=False))
    return 1 if result.get("failed_steps") else 0


if __name__ == "__main__":
    raise SystemExit(main())
