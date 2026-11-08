#!/usr/bin/env bash
# Live Hindi <-> English translation.   bash live.sh [--speak] [--device N] [--file x.wav]  |  bash live.sh devices
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [ "$1" = "devices" ]; then exec .venv/bin/python -m qasr devices; fi
exec .venv/bin/python -m qasr live "$@"
