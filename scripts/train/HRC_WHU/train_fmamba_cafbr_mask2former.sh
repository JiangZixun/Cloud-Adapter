#!/usr/bin/env bash
# CAFBR FMamba + Mask2Former: HRC-WHU, official query losses, 40,000 iterations.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /root/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_mask2former.json "$@"
