import os
import yaml
import logging
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------
# Logger voor deze config-module
# -----------------------------------------------------------
config_logger = logging.getLogger("config")
config_logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------
# 1) Bepaal pad naar config.yaml
# -----------------------------------------------------------
current_dir = os.path.dirname(__file__)  # Directory van dit bestand
yaml_file = os.path.join(current_dir, 'config.yaml')  # Zorg dat dit het juiste pad is

# -----------------------------------------------------------
# 2) Lees het YAML-bestand in
# -----------------------------------------------------------
def load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config-bestand bestaat niet: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    config_logger.debug(f"Config geladen van {path}: {data}")
    return data

yaml_config = load_config_file(yaml_file)

# -----------------------------------------------------------
# 3) Stel belangrijke paden in
# -----------------------------------------------------------
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Hoofdmap van het project
DB_FILE = os.path.join(project_root, 'data', 'market_data.db')        # Correct pad naar de data map

# -----------------------------------------------------------
# 4) Haal subsecties uit de YAML
# -----------------------------------------------------------
DATABASE_CONFIG        = yaml_config.get('database', {})
PAIRS_CONFIG           = yaml_config.get('pairs', [])
WEBSOCKET_CONFIG       = yaml_config.get('websocket', {})
SCALPING_CONFIG        = yaml_config.get('scalping', {})
PULLBACK_CONFIG        = yaml_config.get('pullback_accumulate_strategy', {})
ML_ENGINE_CONFIG       = yaml_config.get('ml_engine', {})
ML_SETTINGS_CONFIG     = yaml_config.get('ml_settings', {})
MULTI_DIRECTION_CONFIG = yaml_config.get('multi_direction_strategy', {})
KRKN_CONFIG            = yaml_config.get('kraken', {})

# -----------------------------------------------------------
# 5) Eventuele debug-logging
# -----------------------------------------------------------
logging.debug(f"KRKN_CONFIG geladen: {KRKN_CONFIG}")

# -----------------------------------------------------------
# 6) Diverse logbestanden
# -----------------------------------------------------------
MAIN_LOG_FILE               = os.path.join(project_root, "logs", "main.log")
EXECUTOR_LOG_FILE           = os.path.join(project_root, "logs", "executor.log")
SCALPING_LOG_FILE           = os.path.join(project_root, "logs", "scalping.log")
WEBSOCKET_LOG_FILE          = os.path.join(project_root, "logs", "websocket_client.log")
ML_ENGINE_LOG_FILE          = os.path.join(project_root, "logs", "ml_engine.log")
TEST_LOG_FILE               = os.path.join(project_root, 'logs', 'test_logger.log')
DASHBOARD_LOG_FILE          = os.path.join(project_root, "logs", "dashboard.log")
PULLBACK_STRATEGY_LOG_FILE  = os.path.join(project_root, "logs", "pullback_strategy.log")

# -----------------------------------------------------------
# 7) Nieuw toegevoegde constants voor logbestanden
#    (Breakout + KrakenAltcoinScanner)
# -----------------------------------------------------------
BREAKOUT_STRATEGY_LOG_FILE     = os.path.join(project_root, "logs", "breakout_strategy.log")
KRKN_ALTCOIN_SCANNER_LOG_FILE  = os.path.join(project_root, "logs", "kraken_altcoin_scanner.log")
