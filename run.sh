#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
  echo "Create .venv and install requirements.txt first. See README.md."
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py "$@"
