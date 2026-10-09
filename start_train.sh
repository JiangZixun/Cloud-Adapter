#!/usr/bin/env bash
# # Done
# bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh --wandb
# bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh --wandb
# bash scripts/train/HRC_WHU/train_lsmamba.sh --wandb

# # TBD
# bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh --wandb
# bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh --wandb
# bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh --wandb

# H04 LR ablations, run sequentially with separate output directories.
bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr5e-5.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_lr3e-5.sh --wandb
