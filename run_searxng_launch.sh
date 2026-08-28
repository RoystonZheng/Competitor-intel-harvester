#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SEARXNG_DIR:-}" ]]; then
  echo "Please set SEARXNG_DIR to your local SearXNG source directory." >&2
  exit 1
fi

cd "$SEARXNG_DIR"
export SEARXNG_SETTINGS_PATH="${SEARXNG_SETTINGS_PATH:-local/settings.yml}"
exec "${SEARXNG_PYTHON:-local/venv/bin/python}" -m searx.webapp
