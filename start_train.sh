#!/usr/bin/env bash
bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh --wandb
bash scripts/train/HRC_WHU/train_lsmamba.sh --wandb
# bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh --wandb

# bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh --wandb
# bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh --wandb