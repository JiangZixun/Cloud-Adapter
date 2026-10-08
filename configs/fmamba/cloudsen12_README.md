# CAFBR FMamba on CloudSEN12 L1C / L2A

Four configurations provide native and official Mask2Former heads on each dataset:
`cloudsen12_l1c_native.json`, `cloudsen12_l1c_mask2former.json`,
`cloudsen12_l2a_native.json`, and `cloudsen12_l2a_mask2former.json`.

The PNG datasets come from `/mnt/data1/Dataset/Cloud-Adapter/cloudsen12_high_l1c`
and `cloudsen12_high_l2a`. Each has 8,490 train, 535 val, and 975 test images,
with matching masks. The entire mask collection was checked: labels are 0–3,
in order clear / thick cloud / thin cloud / cloud shadow. Images are RGB at
512x512; palette indices remain intact. The code consumes these RGB PNGs,
not the original multispectral Sentinel-2 products.

These runs use only `train` and `test`, as requested. The existing `val` directory
is unused. Periodic evaluation on all 975 test images supplies loss and all metrics
and selects top-three/best checkpoints; after training, best is loaded and evaluated
on the same test split again. Standalone evaluation also uses `test`. Histories,
data manifests, environment metadata and plots identify checkpoint selection and
final testing as separate roles on the same original test split.

Default settings: 512x512 input, physical batch size 1, gradient accumulation 4, effective batch size 4,
four workers,
FP32, 40,000 optimizer steps, AdamW peak lr 1e-4 and weight decay 0.05,
seed 42, and validation every 2,000 steps (20 evaluations in total). The first
10% (4,000 optimizer steps) linearly warm up from 1e-6 to 1e-4; the remaining
36,000 steps use cosine decay to zero. Config keys are `lr_schedule: cosine`,
`warmup_ratio: 0.1`, `warmup_start_factor: 0.01`, and `min_lr_ratio: 0.0`.
A checkpoint created with another LR schedule requires a new tuning run; resume
checks the saved schedule to prevent silently changing its optimization history.
Batch 1 accommodates the full model at native resolution on the 32 GB RTX 5090 D.
`grad_accum_steps: 4` averages four micro-batch gradients before each optimizer
update. `max_iters` and tqdm count optimizer updates, so 40,000 steps consume
160,000 physical batches/images (about 18.85 passes over the 8,490-image split).
Warmup, cosine LR, validation and CAFBR switching all count optimizer updates;
validation/test themselves continue to use physical batch 1. Loss logging records
the mean original objective, not the loss divided by four for backward.
JSON and W&B record effective batch, micro-batch counts and optimizer updates.
Resume uses `iteration * grad_accum_steps` for sampling and rejects a different
accumulation factor. Other configurations default to accumulation 1.
Gradient accumulation matches the reported batch count, but BatchNorm statistics
still use physical batch 1, so it is not identical to a physical batch of four. `train_loss_only: true` skips training prediction
conversion, confusion matrices and full-train evaluation. Training only computes
the objective, updates gradients and records window-averaged batch loss and LR.
The window boundaries and CAFBR state remain traceable in JSON and W&B.

All four CAFBR refiners are bypassed and frozen for iterations 1–20,000, then
enabled for joint optimization from iteration 20,001. Checkpoint inference restores
the saved phase. The native loss is **5 x CE + 5 x Dice**; Mask2Former retains
its official classification/mask/Dice weights **2/5/5**, Hungarian matching and
auxiliary supervision. The models initialize from scratch; HRC-WHU checkpoints
are not automatically loaded into these four-class experiments.

```bash
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_mask2former.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_mask2former.sh --wandb

/home/jzx/anaconda3/envs/qwen3/bin/python tools/test_fmamba.py \
  experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/best.pth

bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh \
  --resume experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/last.pth
```

Output paths are `experiments/CloudSEN12_L1C/<Model>/` and
`experiments/CloudSEN12_L2A/<Model>/`. All seven segmentation metrics, per-class
metrics, confusion matrices and configured objective losses are preserved.
`results.json`, `metrics.csv/jsonl`, per-validation JSON, local curves and optional
W&B provide the same traceability as the HRC-WHU experiments.
Training losses are in `results.json`'s `training` array; its `train` evaluation
array remains empty for these runs. Periodic test-set evaluation is in `validation`
with `data_split: test`, while final best evaluation is in `test`. W&B records
batch training loss as `train/loss` and `optimization/loss`; no training segmentation
metrics are calculated. Switching an existing run from the old independent-val
protocol requires a new work directory, preserving its original checkpoint ranking.

## Smoke tests and timing

Add `--smoke-test --work-dir <separate-directory>` to any launcher. CloudSEN12
smoke tests use eight train images and eight test images with the full 512x512
model; smoke limits are saved into their checkpoints so standalone testing remains
bounded. These limits are absent from normal training. Four GPU integration tests
verified both datasets/heads in temporary data roots containing only train/test,
CAFBR activation, loss-only training, test-based selection, top-3/best/last,
metrics, curves and standalone evaluation. Native L1C additionally verified resume. These checks include batch1 x accumulation4,
optimizer/scheduler update counts and 16 micro-batches per four-step smoke run.
The smoke directories and their checkpoints were deleted after testing.

The following benchmark measures the actual CAFBR-off and CAFBR-on stages, plus
evaluation on 64 real test images, and extrapolates the full 975-image test split.
It includes every periodic test-set evaluation, final best testing and
checkpoint I/O; recommended ranges add 10–25% for logging and system variation.

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python tools/benchmark_cloudsen.py \
  --config configs/fmamba/cloudsen12_l1c_native.json
```

Timing reports are saved at `experiments/<Dataset>/<Model>_benchmark_accum4/timing_estimate.json`
for accumulation 4, preserving previous accumulation-1 benchmark reports.
Temporary benchmark checkpoints are automatically removed. This is a throughput
estimate, not evidence of final model quality or completed full training.

The original throughput reports describe physical batch 1 without accumulation.
`timing_estimate_train_test.json` retains the previous train/test estimate for that
protocol; its saved config identifies accumulation 1 or an absent accumulation key.
For the current accumulation-4 protocol, `timing_estimate_accum4.json` scales only
the training portion by four and retains the 20 periodic test-set evaluations,
final best testing and checkpoint I/O. These are approximate projections; four
forward/backward passes share one optimizer update, so actual training need not
be exactly four times longer. Re-run the benchmark above for fresh measurements.

| Dataset | Head | Approximate full-run estimate | Suggested time budget |
| --- | --- | --- | --- |
| CloudSEN12 L1C | Native | 15.4 h | 17–19.2 h |
| CloudSEN12 L1C | Mask2Former | 17.5 h | 19.3–22 h |
| CloudSEN12 L2A | Native | 14.8 h | 16.3–18.5 h |
| CloudSEN12 L2A | Mask2Former | 16.9 h | 18.6–21.2 h |

These estimates include 40,000 optimizer updates (160,000 micro-batches),
all twenty periodic test-set evaluations, final best testing and checkpoint writes.
