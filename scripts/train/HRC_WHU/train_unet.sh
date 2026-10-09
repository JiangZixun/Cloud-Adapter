#!/usr/bin/env bash
# Full-width UNet: HRC-WHU, pixel CE, 40,000 iterations.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /root/anaconda3/envs/qwen3/bin/python tools/train_unet.py \
  --config configs/unet/hrc_whu.json "$@"
