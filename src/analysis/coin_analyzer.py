import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from decimal import Decimal

import pandas as pd
import numpy as np

from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE, yaml_config

logger = logging.getLogger("coin_analyzer")


# ===========================
# Dataclasses voor duidelijke structuur
# ===========================

@dataclass
class TradeMetrics:
    symbol: str
    n_trades: int
    winrate: float
    avg_R: float
    median_R: float
    expectancy_R: float
    avg_roi_pct: float
    max_drawdown_R: float
    avg_hold_hours: float

    # Extra breakdowns
    long_winrate: float
    short_winrate: float
    avg_R_long: float
    avg_R_short: float


@dataclass
class GPTMetrics:
    # Open-calls
    n_open_calls: int
    gpt_winrate: float
    gpt_avg_R: float
    gpt_expectancy_R: float
    high_conf_winrate: float
    low_conf_winrate: float
    avg_confidence_open: float
    gpt_long_winrate: float
    gpt_short_winrate: float

    # HOLD-analyse
    n_hold_decisions: int
    hold_missed_opportunities: int
    hold_missed_rate: float


@dataclass
class CoinReport:
    symbol: str
    trade_metrics: TradeMetrics
    gpt_metrics: GPTMetrics
    flags: List[str]


# ===========================
# Hoofdklasse: CoinAnalyzer
# ===========================

class CoinAnalyzer:
    """
    Analyse-module V1.5 voor trend_4h:
      - Koppelt trades, trade_signals, gpt_decisions
      - Berekent performance metrics per coin
      - Evalueert GPT-beslissingen (OPEN vs HOLD)
      - Markeert eenvoudige 'flags' / lessons learned
      - Kan ook over ALLE coins lopen (analyze_all_coins)
    """

    def __init__(
        self,
        db_path: str = DB_FILE,
        strategy_name: str = "trend_4h"
    ):
        self.db = DatabaseManager(db_path=db_path)
        self.strategy_name = strategy_name

        cfg = yaml_config.get("trend_strategy_4h", {})
        self.sl_atr_mult: float = float(Decimal(str(cfg.get("sl_atr_mult", "1.5"))))

        # Voor de HOLD-evaluatie
        self.lookahead_hours_for_hold: float = float(cfg.get("analysis_hold_lookahead_hours", 8.0))
        self.hold_eval_interval: str = cfg.get("analysis_hold_interval", "1h")
        self.missed_move_threshold_pct: float = float(cfg.get("analysis_hold_missed_move_pct", 2.0))

        logger.info(
            "[CoinAnalyzer] initialized for strategy=%s "
            "(sl_atr_mult=%.2f, hold_lookahead=%.1fh, missed_pct=%.1f%%)",
            self.strategy_name,
            self.sl_atr_mult,
            self.lookahead_hours_for_hold,
            self.missed_move_threshold_pct,
        )

    # ===========================
    # Publieke API
    # ===========================

    def analyze_coin(
        self,
        symbol: str,
        last_n_trades: int = 50,
        last_n_hold_decisions: int = 100
    ) -> Optional[CoinReport]:
        """
        Hoofdfunctie: levert een CoinReport op basis van:
          - laatste N gesloten master-trades
          - GPT-beslissingen (open + hold)
        """
        trades_df = self._load_master_trades(symbol, last_n_trades)
        if trades_df.empty:
            logger.info("[analyze_coin][%s] Geen trades gevonden.", symbol)
            return None

        signals_df = self._load_signals_for_trades(trades_df["id"].tolist())
        gpt_df = self._load_gpt_for_trades(trades_df["id"].tolist())

        joined = self._join_trades_signals_gpt(trades_df, signals_df, gpt_df)
        if joined.empty:
            logger.info("[analyze_coin][%s] Geen joined rows na merge.", symbol)
            return None

        joined = self._compute_derived_fields(joined)

        trade_metrics = self._compute_trade_metrics(symbol, joined)
        gpt_metrics_open = self._compute_gpt_open_metrics(joined)
        hold_stats = self._analyze_hold_decisions(symbol, last_n_hold_decisions)

        gpt_metrics = GPTMetrics(
            n_open_calls=gpt_metrics_open["n_open_calls"],
            gpt_winrate=gpt_metrics_open["gpt_winrate"],
            gpt_avg_R=gpt_metrics_open["gpt_avg_R"],
            gpt_expectancy_R=gpt_metrics_open["gpt_expectancy_R"],
            high_conf_winrate=gpt_metrics_open["high_conf_winrate"],
            low_conf_winrate=gpt_metrics_open["low_conf_winrate"],
            avg_confidence_open=gpt_metrics_open["avg_confidence_open"],
            gpt_long_winrate=gpt_metrics_open["gpt_long_winrate"],
            gpt_short_winrate=gpt_metrics_open["gpt_short_winrate"],
            n_hold_decisions=hold_stats["n_hold"],
            hold_missed_opportunities=hold_stats["n_missed"],
            hold_missed_rate=hold_stats["missed_rate"],
        )

        flags = self._derive_flags(trade_metrics, gpt_metrics, joined)

        report = CoinReport(
            symbol=symbol,
            trade_metrics=trade_metrics,
            gpt_metrics=gpt_metrics,
            flags=flags
        )
        return report

    def analyze_all_coins(
        self,
        min_trades: int = 5,
        last_n_trades: int = 50,
        last_n_hold_decisions: int = 100
    ) -> List[CoinReport]:
        """
        Loop over alle coins waar deze strategy master-trades heeft.
        Handig voor een snel totaaloverzicht.
        """
        symbols = self._load_all_symbols_for_strategy()
        reports: List[CoinReport] = []

        for sym in symbols:
            rep = self.analyze_coin(
                sym,
                last_n_trades=last_n_trades,
                last_n_hold_decisions=last_n_hold_decisions
            )
            if rep is None:
                continue
            if rep.trade_metrics.n_trades < min_trades:
                continue
            reports.append(rep)

        logger.info("[analyze_all_coins] %d coins met voldoende sample.", len(reports))
        return reports

    # ===========================
    # Data laden
    # ===========================

    def _load_all_symbols_for_strategy(self) -> List[str]:
        """
        Haal alle unieke symbols op voor deze strategy waar master-trades bestaan.
        """
        sql = """
        SELECT DISTINCT symbol
        FROM trades
        WHERE strategy_name = ?
          AND is_master = 1
          AND status IN ('open','partial','closed')
        """
        rows = self.db.execute_query(sql, (self.strategy_name,))
        if not rows:
            return []
        return [r[0] for r in rows if r[0]]

    def _load_master_trades(self, symbol: str, last_n: int) -> pd.DataFrame:
        """
        Haal master-trades voor deze strategie en coin op.

        Extra stap:
        - we halen per master-trade de som van de child-amounts op
          (is_master = 0, zelfde position_id)
        - daaruit bouwen we 'effective_amount' die we straks voor R gebruiken
        """
        sql = """
        SELECT
            t.id,
            t.timestamp,
            t.datetime_utc,
            t.symbol,
            t.side,
            t.price,
            t.amount AS master_amount,
            t.position_id,
            t.position_type,
            t.status,
            t.pnl_eur,
            t.fees,
            t.trade_cost,
            t.exchange,
            t.strategy_name,
            t.is_master,
            COALESCE((
                SELECT SUM(c.amount)
                FROM trades c
                WHERE c.position_id = t.position_id
                  AND c.is_master = 0
            ), 0) AS child_amount_sum
        FROM trades t
        WHERE t.strategy_name = ?
          AND t.symbol = ?
          AND t.is_master = 1
          AND t.status IN ('open','partial','closed')
        ORDER BY t.timestamp DESC
        LIMIT ?
        """
        rows = self.db.execute_query(sql, (self.strategy_name, symbol, last_n))
        if not rows:
            return pd.DataFrame()

        cols = [
            "id", "timestamp", "datetime_utc", "symbol",
            "side", "price", "master_amount", "position_id",
            "position_type", "status", "pnl_eur", "fees",
            "trade_cost", "exchange", "strategy_name", "is_master",
            "child_amount_sum",
        ]
        df = pd.DataFrame(rows, columns=cols)

        # Alleen gesloten trades voor de analyse
        df = df[df["status"] == "closed"].copy()
        if df.empty:
            return df

        # Type-correcties
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["master_amount"] = pd.to_numeric(df["master_amount"], errors="coerce")
        df["child_amount_sum"] = pd.to_numeric(df["child_amount_sum"], errors="coerce")
        df["pnl_eur"] = pd.to_numeric(df["pnl_eur"], errors="coerce")
        df["trade_cost"] = pd.to_numeric(df["trade_cost"], errors="coerce")

        # Nieuwe kolom: effective_amount
        df["effective_amount"] = df["child_amount_sum"]
        # fallback: als child-som 0 of NaN is, pak master_amount (voor het geval je ooit een trade zonder partials hebt)
        mask_fallback = (df["effective_amount"].isna()) | (df["effective_amount"] <= 0)
        df.loc[mask_fallback, "effective_amount"] = df.loc[mask_fallback, "master_amount"]

        return df

    def _load_child_sizes(self, position_ids: List[str]) -> pd.DataFrame:
        """
        Haal per position_id de som van de child-amounts op (is_master = 0).
        """
        if not position_ids:
            return pd.DataFrame(columns=["position_id", "child_amount"])

        placeholders = ",".join(["?"] * len(position_ids))
        sql = f"""
           SELECT
               position_id,
               SUM(amount) AS child_amount
           FROM trades
           WHERE position_id IN ({placeholders})
             AND is_master = 0
           GROUP BY position_id
           """
        rows = self.db.execute_query(sql, tuple(position_ids))
        if not rows:
            return pd.DataFrame(columns=["position_id", "child_amount"])

        df = pd.DataFrame(rows, columns=["position_id", "child_amount"])
        df["child_amount"] = pd.to_numeric(df["child_amount"], errors="coerce")
        return df

    def _load_signals_for_trades(self, trade_ids: List[int]) -> pd.DataFrame:
        """
        Laad trade_signals voor een lijst trade_ids.
        In V1.5 pakken we vooral de 'open' events (of anders de laatste per trade).
        """
        if not trade_ids:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(trade_ids))
        sql = f"""
        SELECT
            id,
            trade_id,
            event_type,
            symbol,
            strategy_name,
            rsi_daily,
            rsi_h4,
            rsi_1h,
            macd_val,
            macd_signal,
            atr_value,
            depth_score,
            ml_signal,
            timestamp
        FROM trade_signals
        WHERE trade_id IN ({placeholders})
        ORDER BY timestamp ASC
        """
        rows = self.db.execute_query(sql, tuple(trade_ids))
        if not rows:
            return pd.DataFrame()

        cols = [
            "id", "trade_id", "event_type", "symbol", "strategy_name",
            "rsi_daily", "rsi_h4", "rsi_1h",
            "macd_val", "macd_signal",
            "atr_value", "depth_score", "ml_signal",
            "timestamp"
        ]
        df = pd.DataFrame(rows, columns=cols)

        num_cols = [
            "rsi_daily", "rsi_h4", "rsi_1h",
            "macd_val", "macd_signal",
            "atr_value", "depth_score", "ml_signal", "timestamp"
        ]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 1 regel per trade: het open-event (of anders de laatste)
        df_sorted = df.sort_values(["trade_id", "timestamp"])
        open_mask = df_sorted["event_type"] == "open"
        df_open = df_sorted[open_mask].copy()
        if df_open.empty:
            df_open = df_sorted.groupby("trade_id").tail(1)
        return df_open

    def _load_gpt_for_trades(self, trade_ids: List[int]) -> pd.DataFrame:
        """
        Laad GPT-decisions gekoppeld aan echte trades (trade_id != NULL).
        Bij meerdere logs per trade pakken we de laatste.
        """
        if not trade_ids:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(trade_ids))
        sql = f"""
        SELECT
            id,
            timestamp,
            symbol,
            strategy_name,
            algo_signal,
            gpt_action,
            confidence,
            rationale,
            journal_tags,
            gpt_version,
            trade_id
        FROM gpt_decisions
        WHERE trade_id IN ({placeholders})
        ORDER BY timestamp ASC
        """
        rows = self.db.execute_query(sql, tuple(trade_ids))
        if not rows:
            return pd.DataFrame()

        cols = [
            "id", "timestamp", "symbol", "strategy_name",
            "algo_signal", "gpt_action", "confidence",
            "rationale", "journal_tags", "gpt_version",
            "trade_id"
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

        df_sorted = df.sort_values(["trade_id", "timestamp"])
        df_last = df_sorted.groupby("trade_id").tail(1)
        return df_last

    # ===========================
    # Join & derived fields
    # ===========================

    def _join_trades_signals_gpt(
        self,
        trades_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        gpt_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge trades + signals + GPT in één DataFrame.
        - trades.id -> signals.trade_id
        - trades.id -> gpt.trade_id
        """
        merged = trades_df.merge(
            signals_df,
            how="left",
            left_on="id",
            right_on="trade_id",
            suffixes=("", "_sig"),
        )

        merged = merged.merge(
            gpt_df,
            how="left",
            left_on="id",
            right_on="trade_id",
            suffixes=("", "_gpt"),
        )

        return merged


    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        R, ROI, hold-time in uren etc.
        - gebruikt 'effective_amount' uit _load_master_trades
        - heeft een simpele fallback als die kolom er toch niet is
        """
        df = df.copy()

        # Zorg dat er een kolom 'effective_amount' is
        if "effective_amount" not in df.columns:
            def fallback_effective_amount(row):
                # fallback: probeer trade_cost/price als ruwe schatting
                try:
                    price = float(row.get("price", np.nan))
                    trade_cost = float(row.get("trade_cost", np.nan))
                    if price > 0 and not np.isnan(trade_cost):
                        return trade_cost / price
                    return np.nan
                except Exception:
                    return np.nan

            df["effective_amount"] = df.apply(fallback_effective_amount, axis=1)

        # ROI in %
        with np.errstate(divide="ignore", invalid="ignore"):
            df["roi_pct"] = (df["pnl_eur"] / df["trade_cost"]) * 100.0

        # Base-risk in EUR op basis van ATR bij open
        # R ≈ pnl_eur / (atr_value * sl_atr_mult * effective_amount)
        def compute_R(row):
            try:
                atr = float(row.get("atr_value", np.nan))
                amt = float(row.get("effective_amount", np.nan))
                pnl = float(row.get("pnl_eur", np.nan))
                if atr <= 0 or amt <= 0 or np.isnan(atr) or np.isnan(amt):
                    return np.nan
                base_risk = atr * self.sl_atr_mult * amt
                if base_risk == 0:
                    return np.nan
                return pnl / base_risk
            except Exception:
                return np.nan

        df["R"] = df.apply(compute_R, axis=1)

        # Voor nu: geen echte hold-times → alles NaN
        df["hold_hours"] = np.nan

        return df

    def _estimate_position_size_from_partials(self, position_id: str, side: str) -> Optional[float]:
        """
        Pragmatic V1:
        - Als amount op de master-trade 0 is,
          probeer de totale positie-size uit de partial trades te halen.
        - We sommen de abs(amount) van alle child trades met dezelfde side.
        """
        if not position_id:
            return None

        try:
            sql = """
            SELECT amount, side
            FROM trades
            WHERE position_id = ?
              AND is_master = 0
            """
            rows = self.db.execute_query(sql, (position_id,))
            if not rows:
                return None

            total = 0.0
            for amt, child_side in rows:
                if amt is None:
                    continue
                try:
                    amt_f = float(amt)
                except Exception:
                    continue

                # Idealiter alleen de openingskant (zelfde side als master)
                if child_side == side or side is None:
                    total += abs(amt_f)

            if total <= 0:
                return None
            return total

        except Exception as e:
            logger.error(f"[_estimate_position_size_from_partials] Fout voor position_id={position_id}: {e}")
            return None

    # ===========================
    # Metrics
    # ===========================

    def _compute_trade_metrics(self, symbol: str, df: pd.DataFrame) -> TradeMetrics:
        """
        Basis performance metrics voor trades + long/short breakdown.
        """
        n = len(df)
        if n == 0:
            return TradeMetrics(
                symbol=symbol,
                n_trades=0,
                winrate=0.0,
                avg_R=0.0,
                median_R=0.0,
                expectancy_R=0.0,
                avg_roi_pct=0.0,
                max_drawdown_R=0.0,
                avg_hold_hours=0.0,
                long_winrate=0.0,
                short_winrate=0.0,
                avg_R_long=0.0,
                avg_R_short=0.0,
            )

        df_R = df["R"].dropna()
        df_roi = df["roi_pct"].dropna()

        wins = df_R[df_R > 0]
        winrate = float(len(wins) / len(df_R)) if len(df_R) > 0 else 0.0
        avg_R = float(df_R.mean()) if len(df_R) > 0 else 0.0
        median_R = float(df_R.median()) if len(df_R) > 0 else 0.0
        expectancy_R = avg_R  # Gemiddelde R/trade

        avg_roi_pct = float(df_roi.mean()) if len(df_roi) > 0 else 0.0

        # Max drawdown in R
        if len(df_R) > 0:
            cum_R = df_R.cumsum()
            running_max = cum_R.cummax()
            drawdowns = cum_R - running_max
            max_drawdown_R = float(drawdowns.min())
        else:
            max_drawdown_R = 0.0

        avg_hold_hours = float(df["hold_hours"].dropna().mean()) if df["hold_hours"].notna().any() else 0.0

        # Long vs short breakdown
        df_long = df[df["side"].str.upper() == "BUY"]
        df_short = df[df["side"].str.upper() == "SELL"]

        def subset_stats(sub: pd.DataFrame) -> (float, float):
            R = sub["R"].dropna()
            if len(R) == 0:
                return 0.0, 0.0
            w = R[R > 0]
            wr = float(len(w) / len(R))
            avg_r = float(R.mean())
            return wr, avg_r

        long_wr, avg_R_long = subset_stats(df_long)
        short_wr, avg_R_short = subset_stats(df_short)

        return TradeMetrics(
            symbol=symbol,
            n_trades=n,
            winrate=winrate,
            avg_R=avg_R,
            median_R=median_R,
            expectancy_R=expectancy_R,
            avg_roi_pct=avg_roi_pct,
            max_drawdown_R=max_drawdown_R,
            avg_hold_hours=avg_hold_hours,
            long_winrate=long_wr,
            short_winrate=short_wr,
            avg_R_long=avg_R_long,
            avg_R_short=avg_R_short,
        )

    def _compute_gpt_open_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Stats alleen over trades waar GPT daadwerkelijk een OPEN-call gaf.
        We kijken naar:
          - winrate / avg R
          - high/low confidence performance
          - long vs short performance
        """
        gpt_open = df[df["gpt_action"].isin(["OPEN_LONG", "OPEN_SHORT"])].copy()
        n_open = len(gpt_open)

        if n_open == 0:
            return {
                "n_open_calls": 0,
                "gpt_winrate": 0.0,
                "gpt_avg_R": 0.0,
                "gpt_expectancy_R": 0.0,
                "high_conf_winrate": 0.0,
                "low_conf_winrate": 0.0,
                "avg_confidence_open": 0.0,
                "gpt_long_winrate": 0.0,
                "gpt_short_winrate": 0.0,
            }

        R = gpt_open["R"].dropna()
        wins = R[R > 0]
        gpt_winrate = float(len(wins) / len(R)) if len(R) > 0 else 0.0
        gpt_avg_R = float(R.mean()) if len(R) > 0 else 0.0
        gpt_expectancy_R = gpt_avg_R

        # Confidence
        conf = pd.to_numeric(gpt_open["confidence"], errors="coerce").fillna(0)

        high = gpt_open[conf >= 70]
        low = gpt_open[conf < 40]

        def winrate_subset(subset: pd.DataFrame) -> float:
            rr = subset["R"].dropna()
            if len(rr) == 0:
                return 0.0
            return float((rr > 0).sum() / len(rr))

        high_winrate = winrate_subset(high)
        low_winrate = winrate_subset(low)
        avg_conf_open = float(conf.mean()) if len(conf) > 0 else 0.0

        # Long/short view op GPT-open
        long_trades = gpt_open[gpt_open["gpt_action"] == "OPEN_LONG"]
        short_trades = gpt_open[gpt_open["gpt_action"] == "OPEN_SHORT"]

        gpt_long_winrate = winrate_subset(long_trades)
        gpt_short_winrate = winrate_subset(short_trades)

        return {
            "n_open_calls": n_open,
            "gpt_winrate": gpt_winrate,
            "gpt_avg_R": gpt_avg_R,
            "gpt_expectancy_R": gpt_expectancy_R,
            "high_conf_winrate": high_winrate,
            "low_conf_winrate": low_winrate,
            "avg_confidence_open": avg_conf_open,
            "gpt_long_winrate": gpt_long_winrate,
            "gpt_short_winrate": gpt_short_winrate,
        }

    # ===========================
    # GPT HOLD-analyse (te conservatief)
    # ===========================

    def _analyze_hold_decisions(
        self,
        symbol: str,
        last_n_hold: int
    ) -> Dict[str, float]:
        """
        Kijk naar GPT-HOLD beslissingen:
          - gpt_action = 'HOLD'
          - Kijk vervolgens naar de prijsbeweging na de beslissing op candles_kraken.
          - Markeer 'missed opportunity' als de prijs in de trendrichting
            meer dan X% beweegt (config) zonder eerst grote tegenbeweging.
        """
        sql = """
        SELECT
            id,
            timestamp,
            symbol,
            strategy_name,
            algo_signal,
            gpt_action,
            confidence
        FROM gpt_decisions
        WHERE symbol = ?
          AND strategy_name = ?
          AND gpt_action = 'HOLD'
        ORDER BY timestamp DESC
        LIMIT ?
        """
        rows = self.db.execute_query(sql, (symbol, self.strategy_name, last_n_hold))
        if not rows:
            return {"n_hold": 0, "n_missed": 0, "missed_rate": 0.0}

        cols = [
            "id", "timestamp", "symbol", "strategy_name",
            "algo_signal", "gpt_action", "confidence"
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        n_hold = len(df)
        n_missed = 0

        for _, row in df.iterrows():
            if self._is_hold_missed_opportunity(row, symbol):
                n_missed += 1

        missed_rate = float(n_missed / n_hold) if n_hold > 0 else 0.0

        return {
            "n_hold": n_hold,
            "n_missed": n_missed,
            "missed_rate": missed_rate,
        }

    def _is_hold_missed_opportunity(self, hold_row: pd.Series, symbol: str) -> bool:
        """
        Bepaalt, voor één HOLD-beslissing, of de markt daarna een
        duidelijke move in signaalrichting maakt (boven threshold).
        """
        ts = int(hold_row["timestamp"])
        algo_signal = hold_row.get("algo_signal", None)
        if algo_signal not in ("long_candidate", "short_candidate"):
            # Geen duidelijke richting => we beoordelen niet.
            return False

        lookahead_ms = int(self.lookahead_hours_for_hold * 3600 * 1000)
        start_ts = ts
        end_ts = ts + lookahead_ms

        candles = self._fetch_candles_interval(
            symbol=symbol,
            interval=self.hold_eval_interval,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if candles.empty:
            return False

        candles = candles.sort_values("timestamp")
        entry_px = float(candles["close"].iloc[0])
        if entry_px <= 0:
            return False

        high_after = float(candles["high"].max())
        low_after = float(candles["low"].min())

        if algo_signal == "long_candidate":
            move_pct = (high_after - entry_px) / entry_px * 100.0
        else:
            move_pct = (entry_px - low_after) / entry_px * 100.0

        return move_pct >= self.missed_move_threshold_pct

    def _fetch_candles_interval(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int
    ) -> pd.DataFrame:
        """
        Haal candles_kraken op in een time-window via een directe SQL-query.
        """
        sql = """
        SELECT
            timestamp,
            market,
            interval,
            open,
            high,
            low,
            close,
            volume
        FROM candles_kraken
        WHERE market = ?
          AND interval = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp ASC
        """
        rows = self.db.execute_query(sql, (symbol, interval, start_ts, end_ts))
        if not rows:
            return pd.DataFrame()

        cols = [
            "timestamp", "market", "interval",
            "open", "high", "low", "close", "volume"
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        return df

    # ===========================
    # Flags / qualitative conclusions
    # ===========================

    def _derive_flags(
        self,
        trade_metrics: TradeMetrics,
        gpt_metrics: GPTMetrics,
        df_joined: pd.DataFrame
    ) -> List[str]:
        """
        Heuristische 'lessons learned'-flags per coin.
        Dit is V1.5: simpel, maar al nuttig.
        """
        flags: List[str] = []

        tm = trade_metrics
        gm = gpt_metrics

        # 1) General profitability / hitrate
        if tm.n_trades < 10:
            flags.append("LOW_SAMPLE: weinig trades, conclusies zijn onzeker.")
        else:
            if tm.expectancy_R > 0.2 and tm.winrate > 0.55:
                flags.append("A_GRADE: trend_4h op deze coin heeft sterke expectancy en winrate.")
            elif tm.expectancy_R < 0 and tm.winrate < 0.45:
                flags.append("UNDERPERFORM: trend_4h verliest structureel op deze coin.")

        # 2) Drawdown
        if tm.max_drawdown_R <= -3.0:
            flags.append(f"DRAWDOWN_RISK: max drawdown was {tm.max_drawdown_R:.1f} R of erger.")

        # 3) Long/short asymmetry
        if tm.long_winrate > tm.short_winrate + 0.15:
            flags.append("LONG_BIAS: longs doen het duidelijk beter dan shorts.")
        elif tm.short_winrate > tm.long_winrate + 0.15:
            flags.append("SHORT_BIAS: shorts doen het duidelijk beter dan longs.")

        # 4) GPT OPEN vergelijking
        if gm.n_open_calls >= 10:
            if gm.gpt_expectancy_R > tm.expectancy_R + 0.1:
                flags.append("GPT_STRONG: trades op basis van GPT-open calls presteren beter dan gemiddelde trade.")
            elif gm.gpt_expectancy_R < tm.expectancy_R - 0.1:
                flags.append("GPT_WEAK: GPT-open calls doen het slechter dan de strategie als geheel.")

            if gm.high_conf_winrate > gm.gpt_winrate + 0.1:
                flags.append("HIGH_CONF_SIGNAL: hoge-confidence GPT-calls zijn duidelijk beter dan gemiddeld.")
            if gm.low_conf_winrate < gm.gpt_winrate - 0.1:
                flags.append("LOW_CONF_WARNING: lage-confidence GPT-calls zijn relatief zwak; overweeg ze te filteren.")

            # GPT long/short
            if gm.gpt_long_winrate > gm.gpt_short_winrate + 0.15:
                flags.append("GPT_LONG_EDGE: GPT doet het duidelijk beter op long-entries.")
            elif gm.gpt_short_winrate > gm.gpt_long_winrate + 0.15:
                flags.append("GPT_SHORT_EDGE: GPT doet het duidelijk beter op short-entries.")

        # 5) HOLD conservatisme
        if gm.n_hold_decisions >= 10:
            if gm.hold_missed_rate > 0.3:
                flags.append(
                    f"CONSERVATIVE_HOLD: ~{gm.hold_missed_rate*100:.0f}% van HOLD-beslissingen missen een duidelijke move; GPT mogelijk te voorzichtig."
                )
            elif gm.hold_missed_rate < 0.1:
                flags.append("HOLD_OK: GPT-HOLD beslissingen lijken meestal terecht; weinig gemiste moves.")

        return flags


# ===========================
# Convenience-functies
# ===========================

def analyze_and_print(
    symbol: str,
    last_n_trades: int = 50,
    last_n_hold: int = 100
) -> None:
    """
    Handige helper om snel in een Python-shell of script te draaien:

        from src.analysis.coin_analyzer import analyze_and_print
        analyze_and_print("BTC-EUR")

    """
    analyzer = CoinAnalyzer()
    report = analyzer.analyze_coin(
        symbol,
        last_n_trades=last_n_trades,
        last_n_hold_decisions=last_n_hold
    )
    if report is None:
        print(f"No report for {symbol} (no trades).")
        return

    print(f"=== Coin report: {report.symbol} ===\n")

    tm = report.trade_metrics
    gm = report.gpt_metrics

    print("Trades:")
    print(f"  n_trades        : {tm.n_trades}")
    print(f"  winrate         : {tm.winrate*100:.1f}%")
    print(f"  avg_R           : {tm.avg_R:.2f}")
    print(f"  median_R        : {tm.median_R:.2f}")
    print(f"  expectancy_R    : {tm.expectancy_R:.2f}")
    print(f"  avg_roi_pct     : {tm.avg_roi_pct:.2f}%")
    print(f"  max_drawdown_R  : {tm.max_drawdown_R:.2f}")
    print(f"  long_winrate    : {tm.long_winrate*100:.1f}%")
    print(f"  short_winrate   : {tm.short_winrate*100:.1f}%")
    print(f"  avg_R_long      : {tm.avg_R_long:.2f}")
    print(f"  avg_R_short     : {tm.avg_R_short:.2f}")
    print()

    print("GPT (open calls):")
    print(f"  n_open_calls    : {gm.n_open_calls}")
    print(f"  gpt_winrate     : {gm.gpt_winrate*100:.1f}%")
    print(f"  gpt_avg_R       : {gm.gpt_avg_R:.2f}")
    print(f"  gpt_expectancy_R: {gm.gpt_expectancy_R:.2f}")
    print(f"  high_conf_win   : {gm.high_conf_winrate*100:.1f}%")
    print(f"  low_conf_win    : {gm.low_conf_winrate*100:.1f}%")
    print(f"  avg_conf_open   : {gm.avg_confidence_open:.1f}")
    print(f"  gpt_long_win    : {gm.gpt_long_winrate*100:.1f}%")
    print(f"  gpt_short_win   : {gm.gpt_short_winrate*100:.1f}%")
    print()

    print("GPT (HOLD):")
    print(f"  n_hold          : {gm.n_hold_decisions}")
    print(f"  missed_opps     : {gm.hold_missed_opportunities}")
    print(f"  missed_rate     : {gm.hold_missed_rate*100:.1f}%")
    print()

    if report.flags:
        print("Flags / lessons learned:")
        for f in report.flags:
            print(f"  - {f}")
    else:
        print("Geen specifieke flags.")


def analyze_all_and_print(
    last_n_trades: int = 50,
    last_n_hold: int = 100,
    min_trades: int = 5
) -> None:
    """
    Snel overzicht over alle coins:

        from src.analysis.coin_analyzer import analyze_all_and_print
        analyze_all_and_print()
    """
    analyzer = CoinAnalyzer()
    reports = analyzer.analyze_all_coins(
        min_trades=min_trades,
        last_n_trades=last_n_trades,
        last_n_hold_decisions=last_n_hold
    )

    if not reports:
        print("Geen coins met voldoende sample.")
        return

    print("=== Overzicht per coin ===\n")
    for rep in reports:
        tm = rep.trade_metrics
        print(
            f"{rep.symbol:10s} | n={tm.n_trades:3d} | win={tm.winrate*100:5.1f}% | "
            f"exp_R={tm.expectancy_R:5.2f} | dd={tm.max_drawdown_R:5.2f} | flags={len(rep.flags)}"
        )
