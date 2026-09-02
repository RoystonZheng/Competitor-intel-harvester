#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOST="${HARVESTER_HOST:-127.0.0.1}"
PORT="${HARVESTER_PORT:-8765}"
export PATH="$HOME/.local/bin:$HOME/.codex/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -z "${CODEX_COMMAND:-}" ]] && command -v codex >/dev/null 2>&1; then
  export CODEX_COMMAND="$(command -v codex)"
fi

exec "$PYTHON_BIN" app.py --host "$HOST" --port "$PORT"
