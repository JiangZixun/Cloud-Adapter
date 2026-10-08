# Training launchers

The first directory level groups launchers by dataset. HRC-WHU currently has
five configured models:

```text
scripts/train/
  HRC_WHU/
    train_unet.sh
    train_fmamba_cafbr.sh
    train_fmamba_cafbr_mask2former.sh
    train_lsmamba.sh
    train_lsmamba_mask2former.sh
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

Defaults: 256x256 inputs, batch size 4, 40,000 iterations, validation every 4,000
iterations on all 30 test images, top-three mIoU checkpoints plus best/last.
Outputs are saved to `experiments/HRC_WHU/<Model>/`.
UNet uses CE; native CAFBR FMamba uses 5 x CE + 5 x Dice; Mask2Former uses its
classification/mask/Dice losses and auxiliary supervision.
Native LS-Mamba uses 5 x CE + 5 x source-form squared-denominator Dice.
See [LS-Mamba configuration](../../configs/lsmamba/README.md).

Additional arguments, for example:

```bash
# Resume the same UNet run
bash scripts/train/HRC_WHU/train_unet.sh \
  --resume experiments/HRC_WHU/UNet/checkpoints/last.pth

# A separate smoke run
bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh \
  --smoke-test --work-dir experiments/HRC_WHU/FMamba_CAFBR_launcher_smoke
```

The launchers start training immediately. An interrupted run with only startup
files/training logs, no checkpoint and no validation/test results is automatically
preserved in a sibling `<Model>_incomplete_<timestamp>` directory before restarting.
Existing saved runs require `--resume` or a new `--work-dir`. Add launchers for other datasets after
their corresponding model/data configurations have been prepared.
