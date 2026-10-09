#!/usr/bin/env bash
# UNet control: same 5 x CE + 5 x Dice as native CAFBR FMamba.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec /root/anaconda3/envs/qwen3/bin/python tools/train_unet.py \
  --config configs/unet/hrc_whu_ce5_dice5.json "$@"
