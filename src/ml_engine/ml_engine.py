# src/ml_engine/ml_engine.py

import os
import subprocess
import joblib  # <-- Nieuw toegevoegd (indien niet al aanwezig)
from decimal import Decimal
from typing import Optional

from src.logger.logger import setup_logger
from src.config.config import ML_ENGINE_LOG_FILE
from src.utils.config_loader import load_config  # <-- Optioneel, als je ML-config ook via YAML wilt

# Lokale imports
# from src.database_manager.database_manager import DatabaseManager (indien nodig, vaak inject je db_manager)
# from src.indicator_analysis.indicators import ... (als je ML direct op indicator-data loslaat)

class MLEngine:
    def __init__(self, db_manager, config_path: Optional[str] = None):
        """
        ML Engine voor modeltraining, parameteroptimalisatie en (optioneel) inference.

        :param db_manager: DatabaseManager instantie voor opslag en ophalen van modellen/parameters.
        :param config_path: pad naar config.yaml (optioneel).
        """
        self.db_manager = db_manager
        self.logger = setup_logger("MLEngine", ML_ENGINE_LOG_FILE)
        self.logger.info("[ML Engine] Initialisatie voltooid.")

        # [1] Optioneel: inladen config.yaml
        if config_path and os.path.isfile(config_path):
            global_config = load_config(config_path)
            self.ml_config = global_config.get("ml_engine", {})
        else:
            self.ml_config = {}

        # [2] Model-pad en evt. parameters
        # fallback: "models/pullback_model.pkl"
        self.model_path = self.ml_config.get("model_path", "models/pullback_model.pkl")

        # [3] Placeholder voor een geladen model
        self.model = None

    def train_model(self):
        """
        Voert het trainingsproces uit (bijvoorbeeld een extern script).
        """
        self.logger.info("[ML Engine] Start modeltraining.")
        try:
            # Start bijvoorbeeld een trainingsscript
            cmd = ["python", "train_model.py"]  # <-- Je eigen script die data ophaalt & model traint
            subprocess.run(cmd, check=True)
            self.logger.info("[ML Engine] Modeltraining succesvol afgerond.")

            # Eventueel: meteen het getrainde model inladen en opslaan in db
            self._load_model_from_disk()
            self._save_model_to_db()

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[ML Engine] Fout bij modeltraining: {e}")
        except Exception as e:
            self.logger.error(f"[ML Engine] Onverwachte fout: {e}")

    def save_parameters(self, params):
        """
        Sla geoptimaliseerde parameters op in de database.
        Voorbeeld: RSI-threshold, ATR-multiplier, etc.
        """
        try:
            self.db_manager.save_data("parameters", params)
            self.logger.info(f"[ML Engine] Parameters opgeslagen: {params}")
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij opslaan van parameters: {e}")

    def load_parameters(self, symbol):
        """
        Laad de nieuwste parameters uit de database.
        """
        try:
            parameters = self.db_manager.fetch_data("parameters", market=symbol, limit=1)
            self.logger.info(f"[ML Engine] Parameters geladen voor {symbol}: {parameters}")
            return parameters
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij ophalen van parameters: {e}")
            return {}

    # ----------------------------------------------------------------
    # Nieuw: _fetch_training_data(...)
    # ----------------------------------------------------------------
    def _fetch_training_data(self):
        """
        Haal historische candles, trades, indicators etc. uit de DB
        om een dataset te bouwen voor modeltraining.

        [!] Pas aan op jouw data-strategie:
            - Haal X dagen/weken data op
            - Combineer candles en indicatoren
            - Return pandas DataFrame of numpy array
        """
        self.logger.info("[ML Engine] Historische data ophalen voor training.")
        try:
            # Voorbeeld: fetch_data() roept bv. DB-tabel 'candles' en 'indicators'
            # data = self.db_manager.fetch_data("candles", limit=20000)
            # Verwerk data => DF => features => labels

            # Voor demo:
            data = []
            # ...
            return data
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij _fetch_training_data: {e}")
            return []

    # ----------------------------------------------------------------
    # Nieuw: model laden/opslaan
    # ----------------------------------------------------------------
    def _load_model_from_disk(self):
        """
        Lees het modelbestand (pkl) vanaf schijf met joblib (indien dat je flow is).
        """
        if not os.path.exists(self.model_path):
            self.logger.warning(f"[ML Engine] Modelbestand {self.model_path} niet gevonden.")
            return

        try:
            self.model = joblib.load(self.model_path)
            self.logger.info(f"[ML Engine] Model geladen van {self.model_path}")
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij laden van model: {e}")

    def _save_model_to_db(self):
        """
        Opslaan van het ingelezen model in de DB, bijv. in binaire vorm (base64)
        of als bytes (afhankelijk van hoe je DB_manager is opgezet).
        """
        if self.model is None:
            self.logger.warning("[ML Engine] Geen model in geheugen om op te slaan.")
            return

        try:
            # Voorbeeld: je serialiseert 'model' in memory en slaat het op
            model_bytes = joblib.dump(self.model, compress=3)
            # model_bytes is normaliter None als je joblib.dump direct naar bestand schrijft
            # Dus je kunt joblib.dump naar 'joblib.dump()' arg => BytesIO()

            # Voor nu: simplistisch
            self.db_manager.save_model_bytes("ml_models", model_bytes)
            self.logger.info("[ML Engine] Model bytes opgeslagen in DB.")
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij opslaan van model in DB: {e}")

    def load_model_from_db(self):
        """
        Het model weer uit de DB halen en in self.model zetten.
        """
        try:
            model_bytes = self.db_manager.fetch_model_bytes("ml_models", limit=1)
            if not model_bytes:
                self.logger.warning("[ML Engine] Geen model gevonden in DB.")
                return

            # model_bytes decoderen naar python object
            self.model = joblib.loads(model_bytes)
            self.logger.info("[ML Engine] Model geladen vanuit DB in geheugen.")
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij load_model_from_db: {e}")
            self.model = None

    # ----------------------------------------------------------------
    # Nieuw: _predict(...)
    # ----------------------------------------------------------------
    def predict_signal(self, features):
        """
        Predictie op basis van het geladen model,
        return bijv. -1 (short), 0 (geen signaal), +1 (long)
        """
        if self.model is None:
            self.logger.warning("[ML Engine] Geen model geladen, return 0 (geen signaal).")
            return 0

        # features is bijv. [rsi, macd, volume, ...]
        try:
            import numpy as np
            X = np.array([features], dtype=float)
            pred = self.model.predict(X)
            return int(pred[0])
        except Exception as e:
            self.logger.error(f"[ML Engine] Fout bij predict_signal: {e}")
            return 0

