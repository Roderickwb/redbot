#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/redbot/redbot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python3}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "operator app install failed: python not found or not executable: $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" -m pip install \
  fastapi==0.115.6 \
  uvicorn==0.34.0 \
  "pydantic>=2.0,<3.0" \
  "starlette>=0.40.0,<0.42.0"

echo "operator app dependencies installed"
