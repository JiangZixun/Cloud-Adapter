#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
# Sequential L1C base16/base32 min-max runs, each using both GPUs.
# Each child launcher enables W&B, 30 epochs, batch4/GPU and epoch-end tests.
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5_minmax.sh
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base32_down5_minmax.sh
