# tests/test_bitvavo_import.py

import os
import sys

# Debugging voor huidige directory
print("Huidige werkdirectory:", os.getcwd())

# Voeg de src-map toe aan sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Debugging sys.path
print("Debugging sys.path:")
for path in sys.path:
    print(f" - {path}")

# Probeer de logger-module te importeren
try:
    from src.logger.logger import setup_logger
    print("Import van logger module geslaagd!")
except ModuleNotFoundError as e:
    print(f"Fout bij importeren: {e}")
