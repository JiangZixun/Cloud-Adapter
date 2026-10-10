# CAFBR FMamba on CloudSEN12 L1C / L2A

Four configurations provide native and official Mask2Former heads on each dataset:
`cloudsen12_l1c_native.json`, `cloudsen12_l1c_mask2former.json`,
`cloudsen12_l2a_native.json`, and `cloudsen12_l2a_mask2former.json`.

On the A6000 host, native configurations read PNG datasets from
`/opt/data/private/Dataset/Cloud-Adapter/cloudsen12_high_l1c` and
`/opt/data/private/Dataset/Cloud-Adapter/cloudsen12_high_l2a`.
Images are RGB at 512x512, with palette mask labels 0–3 in order
clear / thick cloud / thin cloud / cloud shadow.

## A6000 native DDP training

The two native launchers use `/root/anaconda3/envs/qwen3/bin/python` and
`torch.distributed.run --standalone` with two processes by default, one per GPU.
This host has two RTX A6000 GPUs, Python 3.11.7, Torch 2.1.2+cu118, and
mamba-ssm 1.2.0.post1. All branch synchronization uses `A6000`, tracking
`origin/A6000`.

```bash
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh

# Each dataset uses both GPUs; run these jobs sequentially.
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh \
  --smoke-test --work-dir experiments/CloudSEN12_L1C/FMamba_CAFBR_ddp_smoke
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr.sh \
  --smoke-test --work-dir experiments/CloudSEN12_L2A/FMamba_CAFBR_ddp_smoke

bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh \
  --resume experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/last.pth

# Single GPU debugging with the same global batch size.
NPROC_PER_NODE=1 bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr.sh \
  --grad-accum-steps 4 --work-dir experiments/CloudSEN12_L1C/FMamba_CAFBR_single_gpu
```

`PYTHON`, `NPROC_PER_NODE`, and `CUDA_VISIBLE_DEVICES` can override the interpreter,
process count, and GPU selection. Native defaults are **batch 1 per GPU ×
accumulation 2 × 2 GPUs = global effective batch 4**. Global batch size is computed
from the actual process count. Sampling partitions a shared shuffled stream by
rank, including deterministic augmentation seeds and resume offsets. Accumulated
micro-batches use `no_sync()` until the final backward. Loss is averaged over both
ranks. BatchNorm remains local to each GPU; rank-zero buffers are saved.

At the CAFBR phase boundary, the DDP reducer is rebuilt so the newly unfrozen
refiners participate in gradient synchronization. Only rank zero writes histories,
curves, W&B, and unwrapped checkpoints, and evaluates the complete original test
split while the other rank waits. Saved checkpoints retain each rank's CPU/CUDA
RNG and reject a changed process count or accumulation setting on resume.
Smoke runs also hash every parameter after each update and fail if ranks diverge;
training history records `parameter_sync_sha256`.

These runs use only `train` and `test`, as requested. The existing `val` directory
is unused. Periodic evaluation on all 975 test images supplies loss and all metrics
and selects top-three/best checkpoints; after training, best is loaded and evaluated
on the same test split again. Standalone evaluation also uses `test`. Histories,
data manifests, environment metadata and plots identify checkpoint selection and
final testing as separate roles on the same original test split.

Native DDP settings: 512x512 input, physical batch size 1 per GPU, gradient accumulation 2, global effective batch size 4,
four workers per rank,
FP32, 40,000 optimizer steps, AdamW peak lr 1e-4 and weight decay 0.05,
seed 42, and validation every 2,000 steps (20 evaluations in total). The first
10% (4,000 optimizer steps) linearly warm up from 1e-6 to 1e-4; the remaining
36,000 steps use cosine decay to zero. Config keys are `lr_schedule: cosine`,
`warmup_ratio: 0.1`, `warmup_start_factor: 0.01`, and `min_lr_ratio: 0.0`.
A checkpoint created with another LR schedule requires a new tuning run; resume
checks the saved schedule to prevent silently changing its optimization history.
Native batch 1 accommodates the full model on each 48 GB RTX A6000.
`grad_accum_steps: 2` averages two micro-batches per rank before each optimizer
update; DDP averages gradients between ranks. Mask2Former remains single GPU
with accumulation 4 and its existing host-specific data configuration. `max_iters` and tqdm count optimizer updates, so 40,000 steps consume
160,000 global physical batches/images (about 18.85 passes over the 8,490-image split).
Warmup, cosine LR, validation and CAFBR switching all count optimizer updates;
validation/test themselves continue to use physical batch 1. Loss logging records
the mean original objective, not the loss divided by the accumulation factor for backward.
JSON and W&B record effective batch, micro-batch counts and optimizer updates.
Resume uses `iteration * grad_accum_steps` for sampling and rejects a different
accumulation factor. Other configurations default to accumulation 1.
Gradient accumulation and DDP match the reported global batch count, but BatchNorm statistics
still use physical batch 1 per GPU, so it is not identical to a physical batch of four. `train_loss_only: true` skips training prediction
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

/root/anaconda3/envs/qwen3/bin/python tools/test_fmamba.py \
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

## Smoke tests and historical single-GPU timing

Add `--smoke-test --work-dir <separate-directory>` to any launcher. CloudSEN12
smoke tests use eight train images and eight test images with the full 512x512
model; smoke limits are saved into their checkpoints so standalone testing remains
bounded. These limits are absent from normal training. Previous single-GPU integration tests
verified both datasets/heads in temporary data roots containing only train/test,
CAFBR activation, loss-only training, test-based selection, top-3/best/last,
metrics, curves and standalone evaluation. Native L1C additionally verified resume. Those historical checks include batch1 x accumulation4,
optimizer/scheduler update counts and 16 micro-batches per four-step smoke run.
The smoke directories and their checkpoints were deleted after testing.

The following single-GPU benchmark measures the actual CAFBR-off and CAFBR-on stages, plus
evaluation on 64 real test images, and extrapolates the full 975-image test split.
It includes every periodic test-set evaluation, final best testing and
checkpoint I/O; recommended ranges add 10–25% for logging and system variation.

```bash
/root/anaconda3/envs/qwen3/bin/python tools/benchmark_cloudsen.py \
  --config configs/fmamba/cloudsen12_l1c_native.json
```

Timing reports are saved at `experiments/<Dataset>/<Model>_benchmark_accum4/timing_estimate.json`
for accumulation 4, preserving previous accumulation-1 benchmark reports.
Temporary benchmark checkpoints are automatically removed. This is a throughput
estimate, not evidence of final model quality or completed full training.

The timing figures below are historical RTX 5090 D single-GPU measurements,
not A6000 DDP estimates.

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


## Five-downsample native FMamba variants

The `native_base16_down5` and `native_base32_down5` configurations support both
CloudSEN12 L1C and L2A. `base_channels` now controls the actual model width;
`fmamba.downsample_stages=5` selects a stride-2 first convolution followed by
four 2x2 max pools. The spatial pyramid matches UNetMobv2: 256, 128, 64, 32,
16 for a 512x512 input. Upsampling remains FMamba's learned transpose convolution,
not UNetMobv2's nearest-neighbor interpolation.

| Stage | Resolution | Base 16 channels | Base 32 channels |
|---|---:|---:|---:|
| Encoder 1 | 256x256 | 16 | 32 |
| Encoder 2 | 128x128 | 32 | 64 |
| Encoder 3 | 64x64 | 64 | 128 |
| Encoder 4 | 32x32 | 128 | 256 |
| Bottleneck | 16x16 | 256 | 512 |
| Decoder 4 | 32x32 | 128 | 256 |
| Decoder 3 | 64x64 | 64 | 128 |
| Decoder 2 | 128x128 | 32 | 64 |
| Decoder 1 | 256x256 | 16 | 32 |
| Final decoder (no skip) | 512x512 | 16 | 32 |
| Segmentation logits | 512x512 | 4 | 4 |

Both variants have five downsampling and five upsampling operations, four CAFBR
skip refiners, four encoder and five decoder FourierVSS blocks. The bottleneck
uses convolution only. CAFBR Fourier branches run at encoder stages 2/3/4;
Mamba state size remains 16. The existing CAFBR activation schedule, loss,
optimizer, LR schedule and effective batch size of four are retained.

Four-class parameter counts: base 16 = 4,199,547 (93.41% fewer than the original
63,768,409); base 32 = 16,260,539 (74.50% fewer). Parameter reduction does not
translate directly into the same runtime or activation-memory reduction.

`start_train.sh` runs base16 L1C/L2A, then base32 L1C/L2A sequentially using both
GPUs and `--wandb`. Each configuration has its own experiment directory.

```bash
bash start_train.sh

# Run an individual experiment:
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base32_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base32_down5.sh --wandb
```

The original base64/four-downsample configs remain available. Their checkpoint
keys, parameter order and seeded initialization are preserved, including legacy
AdamW resumes. New widths/depths require fresh training; use their own config
and checkpoint for evaluation/resume. Five-downsample variants use the native
head; the current Mask2Former pyramid requires four downsampling operations.

Architecture/gradient checks are in `tests/test_fmamba_scaling.py`; real-data
512x512 two-GPU smoke results are in `docs/fmamba_scaled_ddp_smoke.json`. Smoke
runs cover the CAFBR off/on transition, exact parameter synchronization after
every optimizer step, validation, checkpoint saving and best-checkpoint testing.
