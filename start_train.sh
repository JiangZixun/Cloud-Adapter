#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
# L1C base16 and base32: each job uses both A6000 GPUs; run sequentially.
# 20 epochs, test every epoch; five down/up stages, SCGM4/SiLU, batch4/GPU, W&B.
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base32_down5.sh --wandb
