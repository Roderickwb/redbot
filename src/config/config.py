# src/config/config.py

import os
import yaml
import logging

# Configureer logging
logging.basicConfig(level=logging.DEBUG)

# 1) Bepaal pad naar config.yaml
current_dir = os.path.dirname(__file__)  # Directory van config.py
yaml_file = os.path.join(current_dir, 'config.yaml')  # Pad naar config.yaml

# 2) Lees de YAML in
def load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config-bestand bestaat niet: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    logging.debug(f"Config geladen van {path}: {data}")
    return data

yaml_config = load_config_file(yaml_file)

# 3) Stel belangrijke paden in
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Hoofdmap van het project
DB_FILE = os.path.join(project_root, 'data', 'market_data.db')  # Correct pad naar data map

# 4) Haal subsecties uit de YAML
DATABASE_CONFIG        = yaml_config.get('database', {})
PAIRS_CONFIG           = yaml_config.get('pairs', [])
WEBSOCKET_CONFIG       = yaml_config.get('websocket', {})
SCALPING_CONFIG        = yaml_config.get('scalping', {})
PULLBACK_CONFIG        = yaml_config.get('pullback_accumulate_strategy', {})
ML_ENGINE_CONFIG       = yaml_config.get('ml_engine', {})
ML_SETTINGS_CONFIG     = yaml_config.get('ml_settings', {})
MULTI_DIRECTION_CONFIG = yaml_config.get('multi_direction_strategy', {})

# 5) Diverse log-bestanden
MAIN_LOG_FILE               = os.path.join(project_root, "logs", "main.log")
EXECUTOR_LOG_FILE           = os.path.join(project_root, "logs", "executor.log")
SCALPING_LOG_FILE           = os.path.join(project_root, "logs", "scalping.log")
WEBSOCKET_LOG_FILE          = os.path.join(project_root, "logs", "websocket_client.log")
ML_ENGINE_LOG_FILE          = os.path.join(project_root, "logs", "ml_engine.log")
TEST_LOG_FILE               = os.path.join(project_root, 'logs', 'test_logger.log')
DASHBOARD_LOG_FILE          = os.path.join(project_root, "logs", "dashboard.log")
PULLBACK_STRATEGY_LOG_FILE  = os.path.join(project_root, "logs", "pullback_strategy.log")

# Eventueel: Voeg extra logica toe (env. overrides, fallback, etc.)
