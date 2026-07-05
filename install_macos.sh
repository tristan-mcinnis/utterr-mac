#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-macos.txt
python -m sounddevice
printf '\nNow run: source .venv/bin/activate && python utterr_macos.py\n'
