# test_config.py

import sys
import os

# Voeg de project root toe aan sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # Omdat test_config.py zich in de hoofdmap bevindt
sys.path.append(project_root)

from src.config.config import yaml_config, PULLBACK_CONFIG, PAIRS_CONFIG, DB_FILE


def test_config_loading():
    print("Algemene Configuratie:")
    print(yaml_config)

    print("\nPullback Accumulate Strategy Configuratie:")
    print(PULLBACK_CONFIG)

    print("\nPairs Configuratie:")
    print(PAIRS_CONFIG)

    print("\nDatabase File Path:")
    print(DB_FILE)


if __name__ == "__main__":
    test_config_loading()
