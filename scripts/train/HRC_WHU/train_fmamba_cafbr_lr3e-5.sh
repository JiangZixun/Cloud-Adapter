#!/usr/bin/env bash
# H04 LR ablation: 10% warmup + cosine; 5CE + 5Dice; 40,000 steps.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /root/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_native.json \
  --lr 3e-5 \
  --work-dir experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5 "$@"
