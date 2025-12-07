import logging
from typing import Dict, Any, List

from src.analysis.coin_analyzer import CoinAnalyzer
from src.database_manager.database_manager import DatabaseManager, get_current_utc_timestamp_ms
from src.config.config import DB_FILE
from src.analysis.analysis_reporter import AnalysisReporter  # <--- NIEUW

logger = logging.getLogger("analysis_job")


def _coin_report_to_dict(rep) -> Dict[str, Any]:
    """
    Helper: zet een CoinReport (dataclass) om naar een JSON-vriendelijke dict.
    Dit gebruiken we voor de AnalysisReporter.
    """
    tm = rep.trade_metrics
    gm = rep.gpt_metrics

    return {
        "symbol": rep.symbol,
        "trade_metrics": {
            "n_trades": tm.n_trades,
            "winrate": tm.winrate,
            "avg_R": tm.avg_R,
            "median_R": tm.median_R,
            "expectancy_R": tm.expectancy_R,
            "avg_roi_pct": tm.avg_roi_pct,
            "max_drawdown_R": tm.max_drawdown_R,
            "avg_hold_hours": tm.avg_hold_hours,
            "long_winrate": tm.long_winrate,
            "short_winrate": tm.short_winrate,
            "avg_R_long": tm.avg_R_long,
            "avg_R_short": tm.avg_R_short,
        },
        "gpt_metrics": {
            "n_open_calls": gm.n_open_calls,
            "gpt_winrate": gm.gpt_winrate,
            "gpt_avg_R": gm.gpt_avg_R,
            "gpt_expectancy_R": gm.gpt_expectancy_R,
            "high_conf_winrate": gm.high_conf_winrate,
            "low_conf_winrate": gm.low_conf_winrate,
            "avg_confidence_open": gm.avg_confidence_open,
            "gpt_long_winrate": gm.gpt_long_winrate,
            "gpt_short_winrate": gm.gpt_short_winrate,
            "n_hold_decisions": gm.n_hold_decisions,
            "hold_missed_opportunities": gm.hold_missed_opportunities,
            "hold_missed_rate": gm.hold_missed_rate,
        },
        "flags": rep.flags or [],
    }


def run_analysis_job():
    """
    Draait de coin-analyse over alle coins en:
    1) schrijft per coin één regel naar de tabel 'coin_analysis_summary'
    2) schrijft JSON-rapporten naar de 'analysis/' map (voor AI agents)
    """
    logger.info("[run_analysis_job] Start analysis job...")

    # 1) DB-manager + tabellen (incl. coin_analysis_summary) zekerstellen
    db = DatabaseManager(db_path=DB_FILE)
    db.init_db()

    # 2) Analyzer gebruiken voor trend_4h (default in CoinAnalyzer)
    analyzer = CoinAnalyzer()

    # 3) Alle reports ophalen
    reports = analyzer.analyze_all_coins(
        min_trades=1,
        last_n_trades=50,
        last_n_hold_decisions=100,
    )

    if not reports:
        logger.info("[run_analysis_job] Geen coins met trades, niets om op te slaan.")
        db.close_connection()
        return

    # Eén timestamp voor deze hele run
    now_ms = get_current_utc_timestamp_ms()

    insert_sql = """
    INSERT INTO coin_analysis_summary (
        created_ts, strategy_name, symbol,
        n_trades, winrate, expectancy_R, max_drawdown_R,
        long_winrate, short_winrate,
        gpt_winrate, gpt_expectancy_R, hold_missed_rate,
        flags_text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0

    # 4) Bestaande DB-logging: 1 regel per coin in coin_analysis_summary
    for rep in reports:
        tm = rep.trade_metrics
        gm = rep.gpt_metrics

        flags_text = "|".join(rep.flags) if rep.flags else ""

        params = (
            now_ms,
            analyzer.strategy_name,
            rep.symbol,
            tm.n_trades,
            tm.winrate,
            tm.expectancy_R,
            tm.max_drawdown_R,
            tm.long_winrate,
            tm.short_winrate,
            gm.gpt_winrate,
            gm.gpt_expectancy_R,
            gm.hold_missed_rate,
            flags_text,
        )
        db.execute_query(insert_sql, params)
        inserted += 1

    logger.info(f"[run_analysis_job] Klaar. {inserted} rows in coin_analysis_summary geschreven.")

    # 5) NIEUW: JSON-rapporten schrijven voor AI-agent / tooling
    try:
        reporter = AnalysisReporter(base_dir="analysis")
        json_reports: List[Dict[str, Any]] = [_coin_report_to_dict(r) for r in reports]

        reporter.write_full_report(json_reports)
        reporter.write_per_coin_reports(json_reports)
        logger.info("[run_analysis_job] JSON reports geschreven in 'analysis/' map.")
    except Exception as e:
        logger.warning(f"[run_analysis_job] Kon JSON reports niet schrijven: {e}")

    db.close_connection()


if __name__ == "__main__":
    run_analysis_job()
