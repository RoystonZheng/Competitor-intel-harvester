#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOST="${HARVESTER_HOST:-127.0.0.1}"
PORT="${HARVESTER_PORT:-8765}"

exec "$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
