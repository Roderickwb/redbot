import logging

from src.analysis.coin_analyzer import CoinAnalyzer
from src.database_manager.database_manager import DatabaseManager, get_current_utc_timestamp_ms
from src.config.config import DB_FILE

logger = logging.getLogger("analysis_job")


def run_analysis_job():
    """
    Draait de coin-analyse over alle coins en schrijft per coin
    één regel naar de tabel 'coin_analysis_summary'.
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
    db.close_connection()


if __name__ == "__main__":
    run_analysis_job()
