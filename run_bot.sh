#!/usr/bin/env bash
set -euo pipefail

# load .env if present (local dev)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
fi

python -m pip install -r requirements.txt
python bot_telegram.py
