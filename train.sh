#!/usr/bin/env bash
# One command: download -> features -> train -> evaluate -> export -> benchmark.   bash train.sh [--quick]
# A full log is written to results/logs/ automatically. Re-running resumes where it stopped.
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
exec .venv/bin/python -m qasr all "$@"
