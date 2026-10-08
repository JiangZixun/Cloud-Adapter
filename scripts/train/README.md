# Training launchers

The first directory level groups launchers by dataset. HRC-WHU currently has
five configured models and a UNet loss control:

```text
scripts/train/
  HRC_WHU/
    train_unet.sh
    train_unet_ce5_dice5.sh
    train_fmamba_cafbr.sh
    train_fmamba_cafbr_mask2former.sh
    train_lsmamba.sh
    train_lsmamba_mask2former.sh
  CloudSEN12_L1C/
    train_fmamba_cafbr.sh
    train_fmamba_cafbr_mask2former.sh
  CloudSEN12_L2A/
    train_fmamba_cafbr.sh
    train_fmamba_cafbr_mask2former.sh
```

Each launcher locates the repository root, uses
`/home/jzx/anaconda3/envs/qwen3/bin/python` directly, and forwards additional CLI
arguments to the trainer. It uses the configured GPU and output directory.
All launchers display tqdm progress. Add `--wandb` for cloud loss/metric curves;
see [experiment tracking](../../configs/experiment_tracking.md) for the unified
`results.json`, local figures, logging settings and split definitions.

```bash
bash scripts/train/HRC_WHU/train_unet.sh
bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh
bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh
bash scripts/train/HRC_WHU/train_lsmamba.sh
bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh
```

HRC-WHU defaults: 256x256 inputs, batch size 4, 40,000 iterations, validation every 2,000
iterations on all 30 test images; 4,000-step linear warmup followed by cosine decay, top-three mIoU checkpoints plus best/last.
Outputs are saved to `experiments/HRC_WHU/<Model>_warmup_cosine/`.
UNet uses CE; `train_unet_ce5_dice5.sh` runs a separate matched 5CE + 5Dice control; native CAFBR FMamba uses 5 x CE + 5 x Dice; Mask2Former uses its
classification/mask/Dice losses and auxiliary supervision.
Native LS-Mamba uses 5 x CE + 5 x source-form squared-denominator Dice.
See [LS-Mamba configuration](../../configs/lsmamba/README.md).

CloudSEN12 launchers use 512x512 RGB, physical batch 1 and accumulation 4
(effective batch 4), with 40,000 optimizer updates / 160,000 micro-batches.
Validation is every 2,000 optimizer updates; 10% warmup precedes cosine LR. Training
only records loss and updates gradients; full-train evaluation is disabled.
Periodic evaluation and best selection use the 975-image `test` split, followed
by final best testing on the same split. The `val` directory is unused.
CAFBR activates after half the training steps. See
[CloudSEN12 configuration and timing](../../configs/fmamba/cloudsen12_README.md).

Additional arguments, for example:

```bash
# Resume the same UNet run
bash scripts/train/HRC_WHU/train_unet.sh \
  --resume experiments/HRC_WHU/UNet_warmup_cosine/checkpoints/last.pth

# A separate smoke run
bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh \
  --smoke-test --work-dir experiments/HRC_WHU/FMamba_CAFBR_launcher_smoke
```

The launchers start training immediately. An interrupted run with only startup
files/training logs, no checkpoint and no validation/test results is automatically
preserved in a sibling `<Model>_incomplete_<timestamp>` directory before restarting.
Existing saved runs require `--resume` or a new `--work-dir`. Add launchers for other datasets after
their corresponding model/data configurations have been prepared.
