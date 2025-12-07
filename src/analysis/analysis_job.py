import logging
from typing import Dict, Any, List

from src.analysis.coin_analyzer import CoinAnalyzer
from src.database_manager.database_manager import DatabaseManager, get_current_utc_timestamp_ms
from src.config.config import DB_FILE
from src.analysis.analysis_reporter import AnalysisReporter  # <--- NIEUW
from src.analysis.coin_profile_generator import generate_coin_profiles  # <-- NIEUW

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
    1) Analyseer alle coins (trades + GPT + signals).
    2) Schrijf JSON-rapporten naar de map analysis/.
    3) Update coin_profile-tabel in de DB.
    """
    logger.info("[run_analysis_job] Start analysis job...")

    db = DatabaseManager(db_path=DB_FILE)
    db.init_db()

    analyzer = CoinAnalyzer()
    reports = analyzer.analyze_all_coins(
        min_trades=1,
        last_n_trades=50,
        last_n_hold_decisions=100,
    )

    if not reports:
        logger.info("[run_analysis_job] Geen coins met trades, niets om op te slaan.")
        db.close_connection()
        return

    # 1) JSON-rapporten schrijven
    reporter = AnalysisReporter(base_dir="analysis")
    reporter.write_reports(reports)

    # 2) Coin profiles maken / updaten in DB
    strategy_name = getattr(analyzer, "strategy_name", "trend_4h")
    n_profiles = generate_coin_profiles(db, reports, strategy_name=strategy_name)
    logger.info("[run_analysis_job] Coin profiles updated for %d symbols.", n_profiles)

    db.close_connection()

if __name__ == "__main__":
    run_analysis_job()
