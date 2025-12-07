import logging

from src.analysis.coin_analyzer import CoinAnalyzer
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE
from src.analysis.analysis_reporter import AnalysisReporter
from src.analysis.coin_profile_generator import generate_coin_profiles

logger = logging.getLogger("analysis_job")


def run_analysis_job():
    """
    Draait de coin-analyse over alle coins, schrijft JSON-rapporten
    én genereert coin_profiles op basis van die JSON files.
    """
    logger.info("[run_analysis_job] Start analysis job...")

    # 1) DB-manager (voor CoinAnalyzer)
    db = DatabaseManager(db_path=DB_FILE)
    db.init_db()

    # 2) Analyzer voor trend_4h (default in CoinAnalyzer)
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

    # 4) JSON-rapporten schrijven (global + per-coin)
    reporter = AnalysisReporter(base_dir="analysis")
    reporter.write_reports(reports)

    # 5) Coin profiles genereren op basis van analysis/coins/*.json
    try:
        n_profiles = generate_coin_profiles()
        logger.info("[run_analysis_job] Coin profiles gegenereerd: %s", n_profiles)
    except TypeError:
        # fallback als jouw generate_coin_profiles() niets teruggeeft
        generate_coin_profiles()
        logger.info("[run_analysis_job] Coin profiles gegenereerd (zonder count).")

    # 6) DB netjes sluiten
    db.close_connection()
    logger.info("[run_analysis_job] Klaar.")


if __name__ == "__main__":
    run_analysis_job()
