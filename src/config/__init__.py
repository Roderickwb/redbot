# src/config/config.py

import os
import yaml
import logging

# Configureer logging
logging.basicConfig(level=logging.DEBUG)

# 1) Bepaal pad naar project_root en config.yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
yaml_file = os.path.join(os.path.dirname(__file__), 'config.yaml')

# 2) Lees de YAML in
with open(yaml_file, 'r') as f:
    yaml_config = yaml.safe_load(f)

def load_config_file(path: str) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"[DEBUG] load_config => {data}")  # of logger.debug(...)
    return data

    #"""
    #Laadt een YAML-bestand in als dict.
    #"""
    #if not os.path.exists(path):
    #    raise FileNotFoundError(f"Config-bestand bestaat niet: {path}")
    #with open(path, 'r') as f:
    #    return yaml.safe_load(f)

# 3) Stel belangrijke paden in
DB_FILE = os.path.join(project_root, 'data', 'market_data.db')

# 4) Haal subsecties uit de YAML (zoals voor DB, pairs, websockets, scalping)
DATABASE_CONFIG  = yaml_config.get('database', {})
PAIRS_CONFIG     = yaml_config.get('pairs', {})
WEBSOCKET_CONFIG = yaml_config.get('websocket', {})
SCALPING_CONFIG  = yaml_config.get('scalping', {})  # voor scalping
scalping_config  = yaml_config.get('scalping', {})  # zelfde, mogelijk overbodig

# Zorg dat je de budgetsectie inlaadt
BUDGET_CONFIG = yaml_config.get("budget_settings", {})
# Of, als je het bij pullback_accumulate_strategy hebt:


# 5) Specifieke subsectie voor de Pullback & Accumulate-strategy
#    (zodat je het niet nog eens in de strategy zelf hoeft te laden)
PULLBACK_CONFIG = yaml_config.get('pullback_accumulate_strategy', {})

# ---- HIER: ML-engine sectie uit de YAML ----
ML_ENGINE_CONFIG = yaml_config.get('ml_engine', {})

# 6) Diverse log-bestanden
MAIN_LOG_FILE        = os.path.join(project_root, "logs", "main.log")
EXECUTOR_LOG_FILE    = os.path.join(project_root, "logs", "executor.log")
SCALPING_LOG_FILE    = os.path.join(project_root, "logs", "scalping.log")
WEBSOCKET_LOG_FILE   = os.path.join(project_root, "logs", "websocket_client.log")
ML_ENGINE_LOG_FILE   = os.path.join(project_root, "logs", "ml_engine.log")
TEST_LOG_FILE        = os.path.join(project_root, 'logs', 'test_logger.log')
DASHBOARD_LOG_FILE   = os.path.join(project_root, "logs", "dashboard.log")
PULLBACK_STRATEGY_LOG_FILE = os.path.join(project_root, "logs", "pullback_strategy.log")



# En je kunt hier evt. extra logic doen (env. overrides, fallback, etc.)




