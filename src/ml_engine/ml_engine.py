import os
import yaml
import random
import logging
import shutil
from decimal import Decimal
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from src.logger.logger import setup_logger
from src.config.config import load_config_file  # Zorg dat dit bestaat in je project
from src.indicator_analysis.indicators import IndicatorAnalysis

try:
    import joblib
except ImportError:
    joblib = None


class MLEngine:
    """
    Doel van MLEngine:
      1) Bieden van een random-search (of grid-search) over de parameterruimte
         van je Pullback Accumulate Strategy.
      2) Een (rudimentaire) backtest op historische data (zonder slippage e.d.),
         returnt een 'score'.
      3) Opslaan van beste paramset en evt. config-overschrijven.

    Met de nieuwe code hieronder wordt de config alléén overschreven
    als 'auto_overwrite_params' op True staat in 'ml_settings'.
    Anders logt hij wel de beste params, maar past niets aan.
    """

    def __init__(self, db_manager, config_path: str = "src/config/config.yaml", model_path: str = None):
        """
        :param db_manager: je DatabaseManager-instance
        :param config_path: pad naar je config.yaml
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger("ml_engine")
        self.model_path = model_path  # <-- sla het pad op

        self.logger.info("[MLEngine] Init, model_path=%s", model_path)

        # Zet eigen logger, zodat hij in logs/ml_engine.log schrijft
        self.logger = setup_logger("ml_engine", "logs/ml_engine.log", level=logging.DEBUG)
        self.logger.info("[ml_engine Engine] Initialisatie gestart.")

        self.db_manager = db_manager
        self.config_path = config_path

        # Laad in-memory de huidige config
        self.current_config = load_config_file(config_path)
        self.logger.info("[ml_engine Engine] Initialisatie voltooid. config_path=%s", config_path)

        # Definieer param-space (random ranges) voor random search
        self.param_candidates = self._build_param_candidates()

        self.model = None  # scikit-learn model, als je dat wilt gebruiken

    def _build_param_candidates(self) -> Dict[str, Tuple[float, float]]:
        """
        Definieer hier de bandbreedtes voor random search.
        Voor elke param: (min_value, max_value).
        """
        param_space = {
            "pullback_threshold_pct": (0.3, 1.0),    # 0.3% .. 1.0%
            "rsi_bull_threshold": (40.0, 65.0),
            "rsi_bear_threshold": (35.0, 60.0),
            "macd_bull_threshold": (-1.0, 0.0),
            "macd_bear_threshold": (0.0, 1.0),
            "tp1_atr_mult": (0.5, 1.5),
            "tp2_atr_mult": (1.5, 3.0),
            "trail_atr_mult": (0.5, 1.5),
            "stop_loss_pct": (0.005, 0.03)
        }
        return param_space

    # --------------------------------------------------------
    # 1) train_model
    # --------------------------------------------------------
    def train_model(self, n_samples: int = 15):
        """
        1) Haal baseline-score op uit je huidige config
        2) Generate random n_samples param-sets
        3) Voor ieder => backtest => score
        4) Vergelijk met baseline
        5) Overschrijf config alleen als auto_overwrite=True
        """
        self.logger.info("[ml_engine] train_model => Start random param search, n_samples=%d", n_samples)

        # baseline: pak je huidige config
        current_params = self._extract_params_from_config()
        baseline_score = self._simulate_and_score(current_params)
        self.logger.info(f"Baseline-score (huidige config) = {baseline_score:.3f}")

        best_score = baseline_score
        best_params = current_params

        for i in range(n_samples):
            candidate = self._random_paramset()
            sc = self._simulate_and_score(candidate)
            self.logger.info(f"[ml_engine] candidate #{i + 1}/{n_samples}, score={sc:.3f}, param={candidate}")

            if sc > best_score:
                best_score = sc
                best_params = candidate

        self.logger.info(f"[ml_engine] Best_score={best_score:.3f}, Best_params={best_params}")

        # Check of er verbetering is
        if best_score > baseline_score:
            self.logger.info("[ml_engine] => Er is verbetering tov baseline.")
            # Lees auto_overwrite_params uit config
            auto_overwrite = self.current_config.get("ml_settings", {}).get("auto_overwrite_params", False)
            if auto_overwrite:
                self.logger.info("[ml_engine] => auto_overwrite_params=True => config.yaml overschrijven.")
                self._update_config_and_backup(best_params)
            else:
                self.logger.info("[ml_engine] => auto_overwrite_params=False => NIET overschrijven.")
                self.logger.info("Nieuwe params = %s", best_params)
        else:
            self.logger.info("[ml_engine] => Geen verbetering tov baseline => niets veranderd.")

    def run_scenario_tests(self) -> dict:
        """
        Placeholder-methode voor executor.run_daily_tasks().
        Kan later uitgebreid worden met echte scenario-test/backtest-logica.
        """
        self.logger.info("[MLEngine] run_scenario_tests => start placeholder")
        best_paramset = None
        best_score = 0.0
        # later kun je hier code integreren
        return {
            "best_paramset": best_paramset,
            "best_score": best_score
        }

    def predict_signal(self, features: list) -> int:
        """
        Geef een int-signaal terug:
          0 => neutraal
          1 => bullish
         -1 => bearish
        Nu: dummy-implementatie => return 0
        """
        return 0

    def _extract_params_from_config(self) -> Dict[str, float]:
        """
        Lees de pullback_accumulate_strategy-sectie en pak relevante param’s.
        """
        strat_conf = self.current_config.get("pullback_accumulate_strategy", {})
        out = {
            "pullback_threshold_pct": float(strat_conf.get("pullback_threshold_pct", 0.5)),
            "rsi_bull_threshold": float(strat_conf.get("rsi_bull_threshold", 50)),
            "rsi_bear_threshold": float(strat_conf.get("rsi_bear_threshold", 50)),
            "macd_bull_threshold": float(strat_conf.get("macd_bull_threshold", -0.5)),
            "macd_bear_threshold": float(strat_conf.get("macd_bear_threshold", 0.5)),
            "tp1_atr_mult": float(strat_conf.get("tp1_atr_mult", 1.0)),
            "tp2_atr_mult": float(strat_conf.get("tp2_atr_mult", 2.0)),
            "trail_atr_mult": float(strat_conf.get("trail_atr_mult", 1.0)),
            "stop_loss_pct": float(strat_conf.get("stop_loss_pct", 0.01))
        }
        return out

    def _random_paramset(self) -> Dict[str, float]:
        """
        Genereer 1 random paramset binnen de ranges in self.param_candidates.
        """
        out = {}
        for key, (low, high) in self.param_candidates.items():
            out[key] = round(random.uniform(low, high), 4)
        return out

    def _simulate_and_score(self, paramset: Dict[str, float]) -> float:
        """
        Rudimentaire backtest over set markten/timeframes.
        Return ~ eind-waarde – start-waarde (start=125).
        """
        markets = ["BTC-EUR", "ETH-EUR", "XRP-EUR", "DOGE-EUR", "SOL-EUR"]
        timeframes = ["5m", "15m", "1h", "4h", "1d"]

        start_eur = 125.0
        eur_balance = start_eur
        coin_balances = {m: 0.0 for m in markets}

        for mk in markets:
            for tf in timeframes:
                df = self._fetch_candles_for_backtest(mk, tf, limit=500)
                eur_balance, coin_balances = self._run_one_pass(
                    mk, tf, df, paramset, eur_balance, coin_balances
                )

        # eind-waarde berekenen
        total_value = eur_balance
        for mk in markets:
            lp = self._get_last_price(mk)
            total_value += coin_balances[mk] * lp

        return total_value - start_eur

    def _fetch_candles_for_backtest(self, market: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        df = self.db_manager.fetch_data(
            table_name="candles",
            limit=limit,
            market=market,
            interval=timeframe
        )
        if not df.empty:
            df = df.sort_values("timestamp")
        return df

    def _run_one_pass(self, mk: str, tf: str, df: pd.DataFrame,
                      paramset: Dict[str, float],
                      eur_balance: float,
                      coin_balances: Dict[str, float]) -> (float, Dict[str, float]):
        """
        Minimale 'random-lus' om buy/sell te simuleren.
        Evt. vervangbaar door echte PullbackAccumulateStrategy in offline mode.
        """
        if df.empty:
            return eur_balance, coin_balances

        # hier zou je paramset integreren => buy/sell logic
        for i in range(1, len(df)):
            row = df.iloc[i]
            close_price = float(row["close"])

            # random buy?
            if random.random() > 0.99:  # 1% kans
                invest_eur = eur_balance * 0.1
                if invest_eur > 2.0:
                    amt = invest_eur / close_price
                    coin_balances[mk] += amt
                    eur_balance -= invest_eur

            # random sell?
            elif random.random() > 0.99 and coin_balances[mk] > 0:
                amt_coin = coin_balances[mk] * 0.5
                proceeds = amt_coin * close_price
                coin_balances[mk] -= amt_coin
                eur_balance += proceeds

        return eur_balance, coin_balances

    def _get_last_price(self, mk: str) -> float:
        df = self.db_manager.fetch_data(table_name="candles", market=mk, limit=1)
        if not df.empty:
            return float(df.iloc[0]["close"])
        return 1.0

    # -------------------------------------------------
    # Opslaan van config => alleen bij auto_overwrite=True
    # -------------------------------------------------
    def _update_config_and_backup(self, best_params: Dict[str, float]):
        """
        Maak backup van je config_path, en overschrijf met best_params.
        """
        backup_path = f"{self.config_path}.bak"
        try:
            shutil.copyfile(self.config_path, backup_path)
            self.logger.info(f"[ml_engine] Backup config => {backup_path}")
        except Exception as e:
            self.logger.warning(f"[ml_engine] geen backup gelukt: {e}")

        updated_conf = load_config_file(self.config_path)
        strat_conf = updated_conf.get("pullback_accumulate_strategy", {})

        # overschrijf paramset
        for k, v in best_params.items():
            old_val = strat_conf.get(k, '?')
            self.logger.info(f" - param {k}: old={old_val} -> new={v}")
            strat_conf[k] = v

        updated_conf["pullback_accumulate_strategy"] = strat_conf

        try:
            with open(self.config_path, "w") as f:
                yaml.dump(updated_conf, f, default_flow_style=False, sort_keys=False)
            self.logger.info("[ml_engine] config.yaml overschreven met best_params.")
        except Exception as e:
            self.logger.error(f"[ml_engine] Fout bij overschrijven config: {e}")

    # -------------------------------------------------
    # load_model_from_db (optioneel) + weekly_report
    # -------------------------------------------------
    def load_model_from_db(self):
        """
        Als je echt ML-model in DB opslaat, kan hier logic komen.
        Nu placeholder.
        """
        pass

    def weekly_report(self):
        """
        Eventueel wekelijks rapport genereren/opslaan.
        Nu placeholder.
        """
        pass
