# CAFBR FMamba on HRC-WHU

For the four CloudSEN12 L1C/L2A native/Mask2Former experiments, see
[the CloudSEN12 guide](cloudsen12_README.md).

The trainer includes tqdm progress, unified train/validation/test JSON history,
local loss/metric curves and opt-in `--wandb` cloud logging. See
[experiment tracking](../experiment_tracking.md).

Use the existing Conda **qwen3** environment by absolute interpreter path:
`/home/jzx/anaconda3/envs/qwen3/bin/python`. No `conda activate` is required.
The machine has no environment literally named `qwen`; `qwen3` is the available
Qwen environment, with Python 3.12.7, Torch 2.10.0+cu128 and Transformers 4.51.3.
The real selective-scan CUDA extensions support the RTX 5090 D.

## Model provenance and adaptations

The CAFBR source is copied into `baselines/fmamba_cafbr/fmamba.py` from
`/mnt/data1/Qwen3/remote_cafbr_overlay/scripts/cloud_seg/models/hzy_fmamba/fmamba.py`.
The local selective-scan wrapper is copied from the source project's
`scripts/cloud_seg/ops/selective_scan_interface.py`. Source paths and SHA256
hashes are recorded in `baselines/fmamba_cafbr/provenance.json`.
The native forward computation is preserved, including Fourier/VSS encoder and
decoder blocks and all four CAFBR skip refiners. Local edits make the scan import
package-relative, make plotting optional, and expose the decoded feature pyramid.
No Qwen language model is instantiated. Both models train from scratch; no
multispectral/10-class checkpoints are loaded into this RGB/two-class experiment.

HRC-WHU uses RGB (3 channels). SCGM requires input channels to be divisible by
its group count, so the source experiment's 16-channel/4-group setting becomes
3-channel/3-group here. CAFBR branch settings are copied from the source resolved
configuration, including spectral branches at stages 4, 3, and 2.

Variants:

* **FMamba_CAFBR**: original FMamba decoder and final convolution;
  **5 x pixel CE + 5 x multiclass soft Dice**. Dice matches the source recipe:
  softmax probabilities, aggregate batch/spatial dimensions, mean over both
  clear-sky and cloud classes, epsilon 1e-5, and ignore pixels labeled 255.
  The weights are explicit in `hrc_whu_native.json`. Changing the loss requires
  a new run; resume rejects checkpoints with different loss settings.
* **FMamba_CAFBR_Mask2Former**: same full FMamba/CAFBR decoder, with its final
  classifier replaced by Transformers' official Mask2Former pixel decoder and
  masked-attention query decoder. The four decoded maps (64/128/256/512 channels)
  are average-pooled by 4 to provide strides 4/8/16/32, retaining all CAFBR paths.
  It uses 256-dimensional features, 100 queries, six deformable pixel-encoder
  layers, and nine attention decoder blocks plus an initial prediction (the HF
  `decoder_layers=10` convention). Official Hungarian matching, point-sampled
  mask/Dice losses, no-object weight .1, CE/mask/Dice weights 2/5/5, and auxiliary
  supervision are enabled. There is no approximate/custom substitute head.

The Mask2Former modules are initialized using the official Transformers method,
without reinitializing the source FMamba. Semantic inference combines query class
probabilities with query mask probabilities, following Mask2Former. Its `val_loss`
is pixel cross-entropy of the resulting semantic scores, while `train_loss` is the
weighted Hungarian-matched query loss. The HRC-WHU masks are fully annotated;
this target adapter rejects ignored pixels rather than treating them as background.
Native FMamba's `train_loss` is the weighted CE+Dice total; its `val_loss` remains
pixel CE for comparability with existing records. The Mask2Former loss settings
remain 2/5/5 with auxiliary supervision, matching the original Cloud-Adapter
loss weights; UNet continues to use pixel CE.

## Training and evaluation

Original splits: `img_dir/train` + `ann_dir/train` (120 images), and
`img_dir/test` + `ann_dir/test` (30 images). As requested, test is also used for
validation and checkpoint selection. Settings shared with UNet: RGB ImageNet
normalization, crop size 256, batch size 4, four workers, AdamW lr 1e-4 and decay
.05, 10% linear warmup followed by cosine decay, 40,000 iterations, validation every 2,000,
seed 42, FP32. No synthetic validation split or external split manifest is used.

Both variants set `cafbr_start_ratio: 0.5`: iterations 1–20,000 bypass all four
CAFBR refiners and freeze their parameters; iteration 20,001 enables them and
joint optimization with the backbone. The optimizer includes these parameters
from the start, so enabling them preserves the existing optimizer and LR schedule.
Validation uses the current phase. Every checkpoint records `cafbr_enabled`, and
final/standalone testing restores that checkpoint's phase, including a best
checkpoint selected before CAFBR activation. The tqdm display, JSON history and
optional W&B metrics expose this state. Resume derives the next phase from the
absolute iteration; changing the configured ratio requires a new run. Older
checkpoints without this setting retain their original always-enabled behavior
when tested. The midpoint follows `max_iters`, including smoke-test overrides.

```bash
# Native head
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_native.json

# Mask2Former head
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_mask2former.json

# Test either variant: its model config is loaded from the checkpoint
/home/jzx/anaconda3/envs/qwen3/bin/python tools/test_fmamba.py \
  experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/checkpoints/best.pth

# Resume the same experiment
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_native.json \
  --resume experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/checkpoints/last.pth
```

Outputs live at `experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/` and
`experiments/HRC_WHU/FMamba_CAFBR_Mask2Former_warmup_cosine/`. Each validation writes
`metrics.csv`, `metrics.jsonl`, and `validation/iter_XXXXXXX.json`, including aAcc,
mIoU, mAcc, mDice, mFscore, mPrecision, mRecall, per-class metrics, confusion
matrix, pixel validation loss and timing. Percentages are unrounded; undefined
values are JSON null. Checkpoints retain the highest three validation mIoUs,
plus `best.pth` and `last.pth`. Validation runs at the last iteration even if
not aligned with the interval. Training automatically reloads/evaluates best at
completion into `best_metrics.json`; the standalone test writes `test_metrics.json`.
Config, environment, exact sample paths, optimizer/scheduler/scaler and Torch RNG
are preserved. Existing output directories are protected against accidental reuse.

## GPU smoke tests and timing

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_native.json --smoke-test
/home/jzx/anaconda3/envs/qwen3/bin/python tools/train_fmamba.py \
  --config configs/fmamba/hrc_whu_mask2former.json --smoke-test
/home/jzx/anaconda3/envs/qwen3/bin/python -m unittest discover -s tests -v
/home/jzx/anaconda3/envs/qwen3/bin/python tools/benchmark_baseline.py \
  --config configs/fmamba/hrc_whu_native.json
```

Smoke tests use the full model architectures, original 256x256 images, batch 4,
five training steps from eight training images, and all 30 test images at every
validation. Outputs have `_smoke` appended to model names. They validate execution
and saving, not final model quality. The benchmark uses all 120 train images and
four workers, five warmup steps then 30 timed steps, a complete validation and
real optimizer-inclusive checkpoint I/O. Temporary timing checkpoints are deleted
automatically, with results saved to `FMamba_CAFBR_benchmark/timing_estimate.json`.

Measured before the native loss changed from CE to CE+Dice on this RTX 5090 D
with qwen3: native CAFBR FMamba has 63,767,255
parameters and averages 0.306 seconds per training step over 30 timed steps.
A complete 30-image validation takes about 0.84 seconds, and a model/optimizer
checkpoint is about 766 MB. The benchmark measured about 20.0 GiB peak allocated
GPU memory. For 40,000 iterations, ten validations and the final best evaluation,
the measured extrapolation is about 3 hours 24 minutes; allow roughly 3.5–4 hours
for a full run on an otherwise idle GPU. This is a short-run estimate, not a
completed full training run or a final quality result.

## HRC-WHU warmup and LR tuning

All HRC-WHU models now use 10% linear warmup followed by cosine decay,
with validation every 2,000 iterations. Completed
runs retain their saved configurations. The tuning controls keep 5CE+5Dice:

```bash
python tools/train_fmamba.py --config configs/fmamba/hrc_whu_native_warmup.json
```

The warmup config uses 10% of 40,000 steps (4,000), linearly increasing LR from
1% of the target LR, then applying cosine decay over the remaining steps. Optional
CLI overrides: `--lr`, `--warmup-ratio`, `--warmup-start-factor`. Use a new
`--work-dir` for each LR trial; schedule changes require a fresh run.
See [the updated tuning plan](../../docs/hrc_whu_cafbr_tuning_plan.md).

Completed PolyLR runs retain their saved configuration. To evaluate/resume an
older run, use its saved `config.json` rather than the new default config.
The new schedule requires a fresh run; the shared trainer rejects a scheduler
change during resume. `lr_schedule="cosine"`, `warmup_ratio=0.1`,
`warmup_start_factor=0.01`, `min_lr_ratio=0.0` are the current HRC defaults.
