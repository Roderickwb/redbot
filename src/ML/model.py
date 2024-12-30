import logging
import subprocess
import os


class MLEngine:
    def __init__(self, db_manager):
        """
        ML Engine voor modeltraining en parameteroptimalisatie.

        :param db_manager: DatabaseManager instantie voor opslag en ophalen van modellen/parameters.
        """
        self.db_manager = db_manager

    def train_model(self):
        """
        Voert het trainingsproces uit (bijvoorbeeld een extern script).
        """
        logging.info("[ML Engine] Start modeltraining.")
        try:
            # Start bijvoorbeeld een trainingsscript
            cmd = ["python", "train_model.py"]
            subprocess.run(cmd, check=True)
            logging.info("[ML Engine] Modeltraining succesvol afgerond.")
        except subprocess.CalledProcessError as e:
            logging.error(f"[ML Engine] Fout bij modeltraining: {e}")
        except Exception as e:
            logging.error(f"[ML Engine] Onverwachte fout: {e}")

    def save_parameters(self, params):
        """
        Sla geoptimaliseerde parameters op in de database.
        """
        try:
            self.db_manager.save_data("parameters", params)
            logging.info(f"[ML Engine] Parameters opgeslagen: {params}")
        except Exception as e:
            logging.error(f"[ML Engine] Fout bij opslaan van parameters: {e}")

    def load_parameters(self, symbol):
        """
        Laad de nieuwste parameters uit de database.
        """
        try:
            return self.db_manager.fetch_data("parameters", market=symbol, limit=1)
        except Exception as e:
            logging.error(f"[ML Engine] Fout bij ophalen van parameters: {e}")
            return {}
