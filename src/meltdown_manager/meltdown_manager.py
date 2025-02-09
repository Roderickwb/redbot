# src/meltdown_manager/meltdown_manager.py

import time
import logging
from decimal import Decimal
import pandas as pd

class MeltdownManager:
    """
    Universeel meltdown-mechanisme:
      - daily_loss check => meltdown als portfolio drop >= daily_loss_pct
      - flash_crash => meltdown als >= meltdown_coins_needed coins in meltdown_coins
                       >= flash_crash_pct drop (over meltdown_tf & meltdown_lookback)
      - meltdown_active => skip new pos, close all
      - RSI-based re-entry => meltdown eindigt als RSI>rsi_reentry_threshold
    """

    def __init__(self, config: dict, db_manager, logger=None):
        """
        config verwacht bijv.:
          {
            "daily_loss_pct": 20,
            "flash_crash_pct": 20,
            "rsi_reentry_threshold": 30,
            "meltdown_coins": ["BTC-EUR","XRP-EUR","ETH-EUR"],
            "meltdown_coins_needed": 2,
            "meltdown_tf": "5m",
            "meltdown_lookback": 3
          }

        db_manager => je DatabaseManager
        logger => optioneel (valt terug op logging.getLogger("meltdown_manager"))
        """
        self.config = config
        self.db_manager = db_manager

        self.logger = logger if logger else logging.getLogger("meltdown_manager")
        self.logger.info("[MeltdownManager] init ...")

        # meltdown state
        self.meltdown_active = False
        self.meltdown_reason = None
        self.meltdown_start_time = 0.0

        # parameters
        self.daily_loss_pct = Decimal(str(self.config.get("daily_loss_pct", "20")))
        self.flash_crash_pct = Decimal(str(self.config.get("flash_crash_pct", "20")))
        self.rsi_reentry_threshold = Decimal(str(self.config.get("rsi_reentry_threshold", "30")))

        self.meltdown_coins = self.config.get("meltdown_coins", ["BTC-EUR","ETH-EUR","XRP-EUR"])
        self.meltdown_coins_needed = int(self.config.get("meltdown_coins_needed", 2))
        self.meltdown_tf = self.config.get("meltdown_tf", "5m")
        self.meltdown_lookback = int(self.config.get("meltdown_lookback", 3))

    def update_meltdown_state(self, strategy, symbol: str) -> bool:
        """
        1) Als meltdown_active=True => check RSI-reentry => meltdown eventueel uit
        2) Anders check daily_loss & flash_crash => meltdown in
        3) Return meltdown_active => True => meltdown => skip open pos, close all
        """
        # 1) meltdown al actief => check RSI-based re-entry
        if self.meltdown_active:
            reentry = self._check_reentry_rsi(strategy, symbol)
            if reentry:
                self.logger.info("[MeltdownManager] RSI-based re-entry => meltdown ended.")
                self.meltdown_active = False
                self.meltdown_reason = None
            return self.meltdown_active

        # meltdown not active => check daily & flash crash
        meltdown_triggered = False

        # a) daily
        meltdown_daily = self._check_daily_loss(strategy)
        # b) flash crash => check meltdown_coins => if >= meltdown_coins_needed => meltdown
        meltdown_flash = self._check_flash_crash(strategy)

        if meltdown_daily:
            meltdown_triggered = True
            self.meltdown_reason = "daily_loss"
        elif meltdown_flash:
            meltdown_triggered = True
            self.meltdown_reason = "flash_crash"

        if meltdown_triggered:
            self.logger.warning(f"[MeltdownManager] meltdown triggered => reason={self.meltdown_reason}")
            self._close_all_positions(strategy)
            self.meltdown_active = True
            self.meltdown_start_time = time.time()

        return self.meltdown_active

    def _check_daily_loss(self, strategy) -> bool:
        """
        Check of (initial_capital - equity_now)/initial_capital >= daily_loss_pct
        """
        equity_now = strategy._get_equity_estimate()
        capital_dec = Decimal(str(strategy.initial_capital))

        drop_val = capital_dec - equity_now
        if capital_dec <= 0:
            self.logger.warning("[MeltdownManager] initial_capital <= 0 => skip daily loss check.")
            return False

        drop_pct = (drop_val / capital_dec) * Decimal("100")

        # -- Toevoeging om duidelijk te loggen wat er gebeurt --
        self.logger.info(
            f"[MeltdownManager] daily_loss_check => equity_now={equity_now:.2f}, "
            f"initial_capital={capital_dec:.2f}, drop_val={drop_val:.2f}, "
            f"drop_pct={drop_pct:.2f}, threshold={self.daily_loss_pct}"
        )
        # ------------------------------------------------------

        if drop_pct >= self.daily_loss_pct:
            self.logger.warning(
                f"[MeltdownManager] daily loss {drop_pct:.2f}% >= {self.daily_loss_pct}% => meltdown."
            )
            return True
        return False

    def _check_flash_crash(self, strategy) -> bool:
        """
        Check meltdown_coins =>
          for each coin => fetch meltdown_lookback candles meltdown_tf => if drop>=flash_crash_pct => count
          if count >= meltdown_coins_needed => meltdown
        """
        drop_count = 0
        for coin in self.meltdown_coins:
            df = strategy._fetch_and_indicator(coin, self.meltdown_tf, limit=self.meltdown_lookback)
            if df.empty or len(df) < 2:
                continue

            first_close_val = df["close"].iloc[0]
            last_close_val = df["close"].iloc[-1]

            # Omzetten naar Decimal
            try:
                first_dec = Decimal(str(first_close_val))
                last_dec = Decimal(str(last_close_val))
            except:
                continue

            if first_dec <= 0:
                continue

            # (first - last)/first * 100 in Decimal
            drop_pct = (first_dec - last_dec) / first_dec * Decimal("100")

            if drop_pct >= self.flash_crash_pct:
                drop_count += 1

        # meltdown als >= meltdown_coins_needed
        if drop_count >= self.meltdown_coins_needed:
            self.logger.warning(
                f"[MeltdownManager] flash_crash => {drop_count}/{len(self.meltdown_coins)} coins >= {self.flash_crash_pct}% drop => meltdown."
            )
            return True
        return False

    def _close_all_positions(self, strategy):
        """
        Alle open positions van deze strategy sluiten
        """
        self.logger.warning("[MeltdownManager] meltdown => close all open positions.")
        for sym in list(strategy.open_positions.keys()):
            self.logger.warning(f"[MeltdownManager] meltdown => close {sym}")
            strategy._close_position(sym)

    def _check_reentry_rsi(self, strategy, symbol: str) -> bool:
        """
        meltdown => re-entry als RSI>rsi_reentry_threshold
        => we gebruiken bv. strategy.entry_timeframe of meltdown_coins[0], net wat je wilt.
        """
        # Pak bijv. strategy.entry_timeframe (pullback= "15m" / breakout= "4h" etc.)
        tf = getattr(strategy, "entry_timeframe", "15m")  # fallback= "15m"
        df_entry = strategy._fetch_and_indicator(symbol, tf, limit=1)
        if df_entry.empty:
            return False

        if "rsi" not in df_entry.columns:
            return False

        last_rsi = df_entry["rsi"].iloc[-1]
        last_rsi_dec = Decimal(str(last_rsi))

        self.logger.info(
            f"[MeltdownManager] RSI re-entry check => RSI={last_rsi_dec:.2f}, threshold={self.rsi_reentry_threshold}"
        )
        return last_rsi_dec > self.rsi_reentry_threshold
