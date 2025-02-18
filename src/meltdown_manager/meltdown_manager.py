# src/meltdown_manager/meltdown_manager.py

import time
import logging
from decimal import Decimal, InvalidOperation
import pandas as pd

# [CHANGED] => import yaml + os om in src/config/peak_state.yaml te schrijven
import yaml
import os

# We nemen aan dat je deze functie WEL hebt, uit je eigen logger.py
from src.logger.logger import setup_logger

try:
    from logger import setup_logger
except ImportError:
    setup_logger = None  # Fallback als je geen logger.py hebt; pas dit naar wens aan.


class MeltdownManager:
    """
    Universeel meltdown-mechanisme:
      - daily_loss check => meltdown als portfolio drop >= daily_loss_pct
        (NU PEAK-BASED: we meten drop% vanaf self.peak_equity i.p.v. initial_capital)
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
            "meltdown_lookback": 3,
            "initial_capital": 350
            # We slaan peak_equity in "src/config/peak_state.yaml" op
          }

        db_manager => je DatabaseManager
        logger => optioneel. Als None, maakt deze class zélf meltdown_manager.log aan.
        """
        self.config = config
        self.db_manager = db_manager

        # Als er geen logger is doorgegeven, maken we er zelf een
        if logger is None:
            logger = setup_logger(
                name="meltdown_manager",
                log_file="logs/meltdown_manager.log",
                level=logging.DEBUG,  # Hoger debug-niveau
                max_bytes=5_000_000,
                backup_count=5,
                use_json=False
            )

        self.logger = logger
        self.logger.info("[MeltdownManager] init ...")
        self.logger.info(f"[MeltdownManager DEBUG] config dict => {config}")

        # meltdown state
        self.meltdown_active = False
        self.meltdown_reason = None
        self.meltdown_start_time = 0.0

        # parameters
        self.daily_loss_pct = Decimal(str(config.get("daily_loss_pct", "20")))
        self.logger.info(f"[DEBUG] meltdown_manager daily_loss_pct => {self.daily_loss_pct}")
        self.flash_crash_pct = Decimal(str(config.get("flash_crash_pct", "20")))
        self.rsi_reentry_threshold = Decimal(str(config.get("rsi_reentry_threshold", "30")))

        self.meltdown_coins = config.get("meltdown_coins", ["BTC-EUR", "ETH-EUR", "XRP-EUR"])
        self.meltdown_coins_needed = int(config.get("meltdown_coins_needed", 2))
        self.meltdown_tf = config.get("meltdown_tf", "5m")
        self.meltdown_lookback = int(config.get("meltdown_lookback", 3))

        # Log de config om zeker te weten dat daily_loss_pct etc. klopt
        self.logger.info(
            f"[MeltdownManager] config => daily_loss_pct={self.daily_loss_pct}, "
            f"flash_crash_pct={self.flash_crash_pct}, rsi_reentry={self.rsi_reentry_threshold}"
        )

        # [CHANGED] => We bouwen een absoluut pad naar src/config/peak_state.yaml,
        #             gebaseerd op de map van meltdown_manager.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "..", "config")
        self.peak_file = os.path.join(config_dir, "peak_state.yaml")

        # [CHANGED] => Fallback = initial_capital
        init_cap_str = str(config.get("initial_capital", "350"))
        try:
            init_cap = Decimal(init_cap_str)
        except InvalidOperation:
            init_cap = Decimal("350")

        # [CHANGED] => load peak_equity (als 'peak_state.yaml' bestaat)
        loaded_peak = self._load_peak_equity()
        if loaded_peak is not None:
            self.peak_equity = loaded_peak
            self.logger.info(f"[MeltdownManager] Loaded peak_equity={self.peak_equity} from {self.peak_file}")
        else:
            self.peak_equity = init_cap
            self.logger.info(f"[MeltdownManager] No peak file => start peak_equity={self.peak_equity}")

    def update_meltdown_state(self, strategy, symbol: str) -> bool:
        """
        Als meltdown actief => check RSI => meltdown end
        Anders => check flash_crash & 'daily_loss' (peak-based => _check_daily_loss),
                  meltdown trigger.
        """
        # update peak_equity als current_equity > peak_equity
        current_eq = strategy._get_equity_estimate()
        self.logger.debug(
            f"[MeltdownManager] update_meltdown_state => current_eq={current_eq:.2f}, peak={self.peak_equity:.2f}")
        if current_eq > self.peak_equity:
            self.peak_equity = current_eq
            self.logger.info(f"[MeltdownManager] new peak_equity => {self.peak_equity:.2f}")
            # [CHANGED] => meteen saven in peak_state.yaml
            self._save_peak_equity(self.peak_equity)

        if self.meltdown_active:
            # meltdown => skip daily/flash, alleen RSI re-entry
            reentry = self._check_reentry_rsi(strategy, symbol)
            if reentry:
                self.logger.info("[MeltdownManager] RSI-based re-entry => meltdown ended.")
                self.meltdown_active = False
                self.meltdown_reason = None
            return self.meltdown_active

        # meltdown not active => check daily_loss & flash_crash
        meltdown_daily = self._check_daily_loss(strategy)
        meltdown_flash = self._check_flash_crash(strategy)

        if meltdown_daily or meltdown_flash:
            self.logger.warning("[MeltdownManager] meltdown triggered => ...")
            self._close_all_positions(strategy)
            self.meltdown_active = True
            self.meltdown_start_time = time.time()

        return self.meltdown_active

    def _check_daily_loss(self, strategy) -> bool:
        """
        Oorspronkelijke docstring:
        Check of (initial_capital - equity_now)/initial_capital >= daily_loss_pct

        [NU PEAK-BASED]:
        we meten drawdown% = (peak_equity - eq_now)/peak_equity *100
        >= self.daily_loss_pct => meltdown
        """
        eq_now = strategy._get_equity_estimate()

        # 1) Als eq_now=0 (of <0), skip meltdown (omdat eq=0 in 99,9% van de gevallen een fout is)
        if eq_now <= 0:
            self.logger.warning("[MeltdownManager] eq_now=0 => skip meltdown this round.")
            return False

        if self.peak_equity <= 0:
            self.logger.warning("[MeltdownManager] peak_equity<=0 => skip meltdown drawdown.")
            return False

        drawdown_val = self.peak_equity - eq_now
        drawdown_pct = (drawdown_val / self.peak_equity) * Decimal("100") if self.peak_equity > 0 else Decimal("0")

        # Log
        self.logger.info(
            f"[MeltdownManager] daily_loss_check(peak-based) => eq_now={eq_now:.2f}, "
            f"peak={self.peak_equity:.2f}, drawdown_val={drawdown_val:.2f}, "
            f"drawdown_pct={drawdown_pct:.2f}, threshold={self.daily_loss_pct}"
        )

        if drawdown_pct >= self.daily_loss_pct:
            self.logger.warning(
                f"[MeltdownManager] meltdown => drawdown {drawdown_pct:.2f}% >= {self.daily_loss_pct}%"
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
            except InvalidOperation:
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
                f"[MeltdownManager] flash_crash => {drop_count}/{len(self.meltdown_coins)} "
                f"coins >= {self.flash_crash_pct}% drop => meltdown."
            )
            return True

        return False

    def _close_all_positions(self, strategy):
        self.logger.warning("[MeltdownManager] meltdown => close all open positions.")
        for sym in list(strategy.open_positions.keys()):
            strategy._close_position(sym, reason="Meltdown")

    def _check_reentry_rsi(self, strategy, symbol: str) -> bool:
        """
        meltdown => re-entry als RSI>rsi_reentry_threshold
        => we gebruiken bv. strategy.entry_timeframe of meltdown_coins[0].
        """
        tf = getattr(strategy, "entry_timeframe", "15m")
        df_entry = strategy._fetch_and_indicator(symbol, tf, limit=30)
        if df_entry.empty or "rsi" not in df_entry.columns or len(df_entry) < 3:
            self.logger.warning("[MeltdownManager] RSI re-entry => not enough data or no RSI column.")
            return False

        last_rsi = df_entry["rsi"].iloc[-1]

        if last_rsi is None or pd.isna(last_rsi):
            self.logger.warning(f"[MeltdownManager] RSI re-entry => ongeldige RSI: {last_rsi}")
            return False

        try:
            last_rsi_dec = Decimal(str(last_rsi))
        except InvalidOperation:
            self.logger.warning(f"[MeltdownManager] RSI re-entry => ongeldige RSI-string: {last_rsi}")
            return False

        self.logger.info(
            f"[MeltdownManager] RSI re-entry check => RSI={last_rsi_dec:.2f}, threshold={self.rsi_reentry_threshold}"
        )
        return last_rsi_dec > self.rsi_reentry_threshold

    # [CHANGED] => helperfuncties voor peak_state.yaml
    def _load_peak_equity(self):
        """
        Probeert 'peak_equity' uit peak_state.yaml te laden.
        Returnt None als het niet bestaat of ongeldige data.
        """
        if not os.path.exists(self.peak_file):
            return None
        try:
            with open(self.peak_file, "r") as f:
                data = yaml.safe_load(f)
            if not data or "peak_equity" not in data:
                return None
            return Decimal(str(data["peak_equity"]))
        except Exception as e:
            self.logger.warning(f"[MeltdownManager] _load_peak_equity => fout: {e}")
            return None

    def _save_peak_equity(self, val: Decimal):
        """
        Sla 'peak_equity' op in src/config/peak_state.yaml.
        """
        data = {"peak_equity": str(val)}
        try:
            with open(self.peak_file, "w") as f:
                yaml.safe_dump(data, f)
            self.logger.info(f"[MeltdownManager] Saved peak_equity={val} to {self.peak_file}")
        except Exception as e:
            self.logger.warning(f"[MeltdownManager] _save_peak_equity => fout: {e}")
