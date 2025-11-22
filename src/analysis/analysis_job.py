# analysis_job.py
import time
from src.analysis.coin_analyzer import CoinAnalyzer
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE

def run_analysis_job():
    db = DatabaseManager(DB_FILE)
    analyzer = CoinAnalyzer()

    # alle reports ophalen
    reports = analyzer.analyze_all_coins(
        min_trades=1,
        last_n_trades=50,
        last_n_hold_decisions=100,
    )

    now_ms = int(time.time() * 1000)

    insert_sql = """
    INSERT INTO coin_analysis_summary (
        created_ts, strategy_name, symbol,
        n_trades, winrate, expectancy_R, max_drawdown_R,
        long_winrate, short_winrate,
        gpt_winrate, gpt_expectancy_R, hold_missed_rate,
        flags_text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

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

    db.close()

if __name__ == "__main__":
    run_analysis_job()
