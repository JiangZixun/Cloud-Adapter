#!/usr/bin/env bash
# Native CAFBR FMamba: HRC-WHU, 5 x CE + 5 x Dice, 40,000 iterations.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_native.json "$@"
