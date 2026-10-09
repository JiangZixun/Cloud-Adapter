#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# # Done
# bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh --wandb
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh --wandb
# bash scripts/train/HRC_WHU/train_lsmamba.sh --wandb

# # TBD
# bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh --wandb
# bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh --wandb
# bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh --wandb

# Completed LR ablations.
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr5e-5.sh --wandb
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr3e-5.sh --wandb

# New tuning runs, sequentially; each retains top-1 plus best/last.
bash scripts/train/HRC_WHU/train_fmamba_cafbr_30k.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_40k_two_phase.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_30k_cafbr25.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_30k_cafbr75.sh --wandb
