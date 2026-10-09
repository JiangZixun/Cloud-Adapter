#!/usr/bin/env bash
# HRC-WHU CAFBR tuning: lr=3e-5, 30,000 updates, CAFBR after 75%, top-1.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/tuning/hrc_whu_30k_cafbr75.json "$@"
