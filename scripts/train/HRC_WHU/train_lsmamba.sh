#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
exec /root/anaconda3/envs/qwen3/bin/python tools/train_lsmamba.py \
  --config configs/lsmamba/hrc_whu_native.json "$@"
