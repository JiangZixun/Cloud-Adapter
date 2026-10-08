# HRC-WHU UNet baseline

The trainer includes tqdm progress, unified train/validation/test JSON history,
local loss/metric curves and opt-in `--wandb` cloud logging. See
[experiment tracking](../experiment_tracking.md).

This is a classic four-level UNet trained from scratch, with RGB input, BatchNorm,
transposed-convolution upsampling, skip connections, and two output logits. It uses
its own UNet decoder rather than Mask2Former. This is a simple reproduction baseline,
not a claim of matching a particular published UNet recipe or paper score.

The standalone trainer needs Python, PyTorch, NumPy, and Pillow; MMSeg/MMCV/MMEngine
are not required. On this machine the selected `qwen3` environment supports the
RTX 5090 D; the default Python environment does not support this GPU.

Run from the repository root:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_unet.py
```

Defaults in `hrc_whu.json`: original 120 training images, all 30 test images used
for validation/checkpoint selection as explicitly requested, RGB ImageNet
normalization, 256x256, batch size 4, AdamW (lr 1e-4, weight decay .05), pixel
cross-entropy loss, 40,000 iterations, PolyLR power .9 without warmup, validation
every 4,000 iterations, seed 42. All UNet parameters are trained. Training uses
random crops, horizontal flips and brightness/contrast/saturation jitter. Palette
mask indices are preserved; classes are clear sky=0 and cloud=1, ignore index=255.
The split manifests elsewhere in the data root are deliberately not used.

The test split serves both checkpoint selection and final evaluation in this
requested setup; final scores therefore describe that protocol, not an independent
held-out evaluation.

Outputs:

```text
experiments/HRC_WHU/UNet/
  config.json                 # resolved training settings
  environment.json            # runtime, device, sample counts, split protocol
  data_manifest.json          # exact image/mask pairs
  train.log
  metrics.jsonl               # every validation, full precision, nested per-class metrics
  metrics.csv                 # every validation, flattened metrics
  validation/iter_XXXXXXX.json # separate record for every validation
  best_metrics.json           # automatic full-test evaluation of best.pth at the end
  checkpoints/
    top_k.json                # ranking by mIoU, descending; ties prefer earlier iteration
    iter_XXXXXXX.pth          # up to 3 highest-scoring validated checkpoints
    last.pth                  # most recent validation, always the final iteration at completion
    best.pth                  # copy of the highest-scoring checkpoint
```

Each checkpoint includes model, optimizer, scheduler, AMP scaler, iteration,
configuration, metrics and Torch RNG states. `last.pth` is saved at every validation,
not every training iteration. A validation is forced at the final iteration even
when it is not a multiple of the validation interval. Undefined per-class metrics
are JSON `null`, and macro means omit undefined classes, as in MMSeg's nanmean.
Metrics are percentages: aAcc, mIoU, mAcc, mDice, mFscore, mPrecision, mRecall.
Per-class IoU/Acc/Dice/Fscore/Precision/Recall and the confusion matrix are also
saved, along with validation loss and the original result's timing fields
`data_time`/`time` (seconds per batch, not comparable across model/environment).

Resume an interrupted run with the same configuration and directory:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_unet.py \
  --resume experiments/HRC_WHU/UNet/checkpoints/last.pth
```

To evaluate a checkpoint separately:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_unet.py \
  --evaluate experiments/HRC_WHU/UNet/checkpoints/best.pth
```

Smoke test (5 training iterations, first 8 training images, 16 base channels,
original 256x256 resolution, all 30 test images each validation). Output uses
`experiments/HRC_WHU/UNet_smoke/` to keep it separate from the full run:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_unet.py --smoke-test
/home/jzx/anaconda3/envs/qwen3/bin/python -m unittest discover -s tests -v
```

Interrupted runs without any checkpoint or validation/test result are preserved
in a sibling `<run-name>_incomplete_<timestamp>_<id>` directory before restarting.
Existing saved output directories are protected: use `--resume` or choose a new
`--work-dir experiments/HRC_WHU/UNet_<run-name>`. Available overrides include
`--max-iters`, `--val-interval`, `--batch-size`, `--num-workers`, `--device`,
`--data-root`, and `--amp`. When evaluating/resuming smoke checkpoints, include
`--smoke-test` to select the matching architecture. Changing `max_iters` changes
the PolyLR horizon; use the original horizon when resuming a full experiment.

Visualize representative training/test samples through the same dataset loader:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/visualize_hrc_whu.py
```

Figures and full-split label statistics are saved under
`experiments/HRC_WHU/UNet_data_preview/visualizations/`. Training panels compare
original RGB/mask/overlay with the loader's augmented RGB/mask/overlay. Test panels
show the loader's RGB/mask/overlay. Normalized tensors are denormalized for display;
gold indicates cloud label 1, black indicates clear sky label 0. Colors are display
choices and do not modify training masks.
