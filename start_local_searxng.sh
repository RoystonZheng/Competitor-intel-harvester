#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARXNG_DIR="${SEARXNG_DIR:-}"

if lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "SearXNG already listening at http://127.0.0.1:8888"
  exit 0
fi

if [[ -z "$SEARXNG_DIR" ]]; then
  echo "Please set SEARXNG_DIR to your local SearXNG source directory." >&2
  echo "Example: SEARXNG_DIR=/path/to/searxng ./start_local_searxng.sh" >&2
  exit 1
fi

cd "$SEARXNG_DIR"
export SEARXNG_SETTINGS_PATH="${SEARXNG_SETTINGS_PATH:-local/settings.yml}"
exec "${SEARXNG_PYTHON:-local/venv/bin/python}" -m searx.webapp
