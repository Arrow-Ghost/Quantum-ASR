#!/usr/bin/env bash
# Quantum ASR - one-time setup for Linux / WSL2 (NVIDIA GPU).   bash setup.sh
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY=${PYTHON:-python3.11}
command -v "$PY" >/dev/null || PY=python3
echo "=== Using $($PY --version)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: nvidia-smi not found"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python
$VPY -m pip install --upgrade pip wheel setuptools
$VPY -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$VPY -m pip install -r requirements.txt
# portaudio is needed by sounddevice for the live microphone (Ubuntu/WSL):  sudo apt install libportaudio2
$VPY -m pip install cupy-cuda12x || echo "cupy not installed -> torch quantum backend"
$VPY -m pip install IndicTransToolkit || echo "IndicTransToolkit not installed -> fallback preprocessing"
$VPY -m qasr selftest || true
.venv/bin/hf auth whoami >/dev/null 2>&1 || .venv/bin/hf auth login
$VPY -m qasr check-access || true
echo "Setup finished. Next: bash train.sh --quick   then   bash train.sh"
