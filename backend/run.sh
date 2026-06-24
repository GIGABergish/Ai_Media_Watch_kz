#!/usr/bin/env bash
# AI Media Watch — POSIX launcher.
# Creates a local virtualenv, installs the core requirements, and starts the API.
# Usage:  ./run.sh            (lite mode)
#         pip install -r requirements-ml.txt  inside .venv for full ML mode.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing core requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting AI Media Watch engine on http://127.0.0.1:8000 ..."
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
