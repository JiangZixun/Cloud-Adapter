#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
# # Done
# bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh --wandb
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh --wandb
# bash scripts/train/HRC_WHU/train_lsmamba.sh --wandb

# # TBD
# bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh --wandb

# H04 LR ablations.
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr5e-5.sh --wandb
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr3e-5.sh --wandb

# Native CAFBR FMamba: each job uses both A6000 GPUs; run sequentially.
# bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh --wandb
# bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh --wandb

# 20 epochs, test every epoch; five down/up stages, SCGM4/SiLU, batch4/GPU, W&B.
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base32_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base32_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base64_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base64_down5.sh --wandb
