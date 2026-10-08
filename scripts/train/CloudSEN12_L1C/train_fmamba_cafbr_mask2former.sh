#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
exec /home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/cloudsen12_l1c_mask2former.json "$@"
