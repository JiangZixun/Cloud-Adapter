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
legacy step-mode smoke tests use eight train images and eight test images with
the full 512x512 model. The six epoch-mode variants instead use 16 train images,
three smoke epochs and eight test images, retaining batch4 per GPU; smoke limits are saved into their checkpoints so standalone testing remains
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

The `native_base16_down5`, `native_base32_down5`, and `native_base64_down5`
configurations support both CloudSEN12 L1C and L2A. `base_channels` now controls the actual model width;
`fmamba.downsample_stages=5` selects a stride-2 first convolution followed by
four 2x2 max pools. The spatial pyramid matches UNetMobv2: 256, 128, 64, 32,
16 for a 512x512 input. Upsampling remains FMamba's learned transpose convolution,
not UNetMobv2's nearest-neighbor interpolation.

| Stage | Resolution | Base 16 channels | Base 32 channels | Base 64 channels |
|---|---:|---:|---:|---:|
| Encoder 1 | 256x256 | 16 | 32 | 64 |
| Encoder 2 | 128x128 | 32 | 64 | 128 |
| Encoder 3 | 64x64 | 64 | 128 | 256 |
| Encoder 4 | 32x32 | 128 | 256 | 512 |
| Bottleneck | 16x16 | 256 | 512 | 1024 |
| Decoder 4 | 32x32 | 128 | 256 | 512 |
| Decoder 3 | 64x64 | 64 | 128 | 256 |
| Decoder 2 | 128x128 | 32 | 64 | 128 |
| Decoder 1 | 256x256 | 16 | 32 | 64 |
| Final decoder (no skip) | 512x512 | 16 | 32 | 64 |
| Segmentation logits | 512x512 | 4 | 4 | 4 |

All three variants have five downsampling and five upsampling operations,
four CAFBR skip refiners, four encoder and five decoder FourierVSS blocks. The bottleneck
uses convolution only. CAFBR Fourier branches run at encoder stages 2/3/4;
Mamba state size remains 16. All six new configurations use four SCGM groups.
The input remains RGB: a learned 1x1 convolution maps 3 input channels to 4
feature channels before SCGM; each group outputs one channel, concatenated into
4 channels for the first encoder convolution. Newly projected SCGM groups use
SiLU at their outputs (`fmamba.scgm_output_activation="silu"`) to retain
negative responses smoothly; the old unprojected groups retain ReLU. This does not add a measured
spectral band. Divisible inputs (including the old RGB/3-group configs) use an
identity projection and retain their original checkpoint structure.

The existing CAFBR activation schedule, loss,
optimizer and LR schedule are retained. Each GPU now uses physical batch size
4 with `grad_accum_steps=1`; on two GPUs the effective batch size is 8.
The target is now 20 epochs with an evaluation after every epoch. Runtime
optimizer-update counts follow the dataset size and actual world size.

Four-class parameter counts: base 16 = 4,199,766 (93.41% fewer than the original
63,768,409); base 32 = 16,260,830 (74.50% fewer); base 64 = 64,057,038
(0.45% more than the original four-downsample base64). Parameter reduction does not
translate directly into the same runtime or activation-memory reduction.

`start_train.sh` runs base16 L1C/L2A, then base32 L1C/L2A, then base64 L1C/L2A
sequentially using both GPUs and `--wandb`. Each configuration has its own experiment directory ending in `_scgm4_silu_b4_e20`.
The four-group variants require fresh training; prior three-group checkpoints
are structurally incompatible with the RGB projection and new grouping.
The SiLU variant uses a fresh directory to keep earlier LeakyReLU experiments
separate. Configurations saved without `scgm_output_activation` retain the
previous activation behavior for checkpoint evaluation/resume.

```bash
bash start_train.sh

# Run an individual experiment:
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base16_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base32_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base32_down5.sh --wandb
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base64_down5.sh --wandb
bash scripts/train/CloudSEN12_L2A/train_fmamba_cafbr_base64_down5.sh --wandb
```

The original base64/four-downsample configs remain available. Their checkpoint
keys, parameter order and seeded initialization are preserved, including legacy
AdamW resumes. New widths/depths require fresh training; use their own config
and checkpoint for evaluation/resume. Five-downsample variants use the native
head; the current Mask2Former pyramid requires four downsampling operations.

Architecture/gradient checks are in `tests/test_fmamba_scaling.py`; real-data
512x512 two-GPU SiLU smoke results (base64 on both datasets) are in
`docs/fmamba_scgm4_silu_ddp_smoke.json`.
Earlier LeakyReLU four-group smoke results are preserved in
`docs/fmamba_scgm4_ddp_smoke.json`. Earlier three-group smoke results are preserved in `docs/fmamba_scaled_ddp_smoke.json`. Smoke
runs cover the CAFBR off/on transition, exact parameter synchronization after
every optimizer step, validation, checkpoint saving and best-checkpoint testing.


## Historical A6000 DDP batch4 timing: 40,000 updates (2026-10-10)

The preceding batch4/accumulation1 configuration used 40,000 updates.
The measurements below are retained as historical throughput evidence.
All six scaled SCGM4/SiLU variants were measured with batch4 per GPU and accumulation1
(effective batch8 across two RTX A6000 48GB GPUs). All six passed real-data DDP
training in both CAFBR phases, finite-loss checks, exact phase-end parameter
synchronization, validation and checkpoint save/reload. Training remains FP32
with 512x512 inputs and 40,000 optimizer updates.

| Base | Dataset | CAFBR off/on seconds per update | Peak allocated GiB/GPU | Measured full-run estimate (hours) | Suggested budget (hours) |
|---|---|---:|---:|---:|---:|
| 16 | CloudSEN12_L1C | 0.605 / 0.647 | 11.52 | 7.30–7.31 | 8.0–9.1 |
| 16 | CloudSEN12_L2A | 0.608 / 0.653 | 11.52 | 7.37–7.37 | 8.1–9.2 |
| 32 | CloudSEN12_L1C | 1.122 / 1.193 | 20.77 | 13.35–13.35 | 14.7–16.7 |
| 32 | CloudSEN12_L2A | 1.123 / 1.186 | 20.77 | 13.32–13.32 | 14.7–16.7 |
| 64 | CloudSEN12_L1C | 1.982 / 2.100 | 40.00 | 23.54–23.54 | 25.9–29.4 |
| 64 | CloudSEN12_L2A | 2.011 / 2.109 | 40.00 | 23.76–23.76 | 26.1–29.7 |

Measurements use 3 warmup updates and 12 timed updates per phase on each
configuration, excluding parameter hashing from update timings. Full-run
estimates include 20 full 975-image evaluations, one final best evaluation, and
worst-case new-best checkpoint saves. Evaluation is measured on 40 real images
and extrapolated by batch count. Suggested budgets include 10–25% for W&B,
system load and data variation. These are short-run estimates, not completed
training results. A separate formal-trainer base64 L2A smoke also passed
CAFBR off/on, every-update synchronization, five optimizer updates (ten total
rank batches), and final best-checkpoint testing with accumulation1.
Peak allocated memory differs from CUDA reserved memory;
base64 reserved about 43.1 GiB per card during training.

All six jobs sequentially: approximately 97.5–110.8 hours (4.1–4.6 days).

Results: `docs/fmamba_batch4_timing.json`. Formal training output directories
for those historical measurements ended in `_scgm4_silu_b4`. Current
20-epoch outputs end in `_scgm4_silu_b4_e20`.

```bash
/root/anaconda3/envs/qwen3/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 tools/benchmark_fmamba_ddp.py \
  --config configs/fmamba/cloudsen12_l1c_native_base64_down5.json
```


## L1C original 40,000-step model with RGB min-max normalization

`cloudsen12_l1c_native_minmax_40k.json` derives from the original L1C config.
It uses fixed uint8 RGB bounds: `(x - 0) / (255 - 0)`, implemented by
`mean=[0,0,0]` and `std=[255,255,255]` in the existing data pipeline.
Training augmentation runs before normalization; train and test use the same
fixed bounds. This does not independently stretch each image's observed range.

Only input normalization and the experiment output directory change. The model
retains base64, four downsampling stages, SCGM3/ReLU, 40,000 optimizer updates,
batch1/GPU with accumulation2, and test-based checkpoint selection every 2000
updates. AdamW, warmup/cosine and the CAFBR activation schedule are unchanged.
Outputs use `experiments/CloudSEN12_L1C/FMamba_CAFBR_minmax_40k`.
The launcher enables W&B by default:

```bash
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_minmax_40k.sh
```

This is a fresh normalization experiment, so train from initialization instead
of resuming the original standardized-input checkpoint. `start_train.sh` still
runs the previously selected L1C base16/base32 30-epoch experiments.
Real-data checks covered 16 train and 16 test RGB images with normalized values
in [0,1]. A two-GPU smoke passed five synchronized updates, both CAFBR phases,
validation/final-best testing and offline W&B logging.
Online W&B smoke also completed on `interactive49037v1`; this does not establish
successful startup on another host. The launcher keeps rank 0 stdout/stderr
directly connected to the terminal for in-place tqdm refreshes, saves rank 1
stdout/stderr and both ranks' Python exception records under
`experiments/ddp_logs/FMamba_CAFBR_minmax_40k` (override with `DDP_LOG_DIR`).
The FMamba entrypoint records child exceptions so torchrun can print the root
traceback instead of only `ChildFailedError`; an intentional invalid-step run
verified this reporting path.
It uses `--redirects 1:3` rather than `--tee 3`: torchrun's line-based tee relay
turns tqdm carriage-return updates into separate output lines.

## Current entrypoint: L1C base16/base32, 30 epochs

`start_train.sh` runs only L1C base16 and base32 sequentially with `--wandb`.
These two configs use `max_epochs=30`, batch4 per GPU, accumulation1, and
testing after every epoch. Their output directories end in `_b4_e30`.
On two GPUs each epoch has 1062 updates, for 31860 updates in total.
The existing 10% warmup now spans epochs 1-3; CAFBR is enabled from epoch 16
with the existing 50% activation schedule. Other variant configs retain 20 epochs.
All epoch-mode variants show a separate `Epoch n/total [Train]` tqdm bar counting
batches within the current epoch (1062 for two GPUs). It completes before the
`Epoch n/total [Test]` bar starts. Each epoch ends with a printed test loss,
aAcc, mIoU, mAcc, mDice, mFscore, mPrecision and mRecall summary.

## Previous 20-epoch configuration and timing estimates

All six L1C/L2A base16/base32/base64 launchers and `start_train.sh` now use
`max_epochs=20`, `val_interval_epochs=1`, batch4 per GPU and accumulation1.
The stored user configs no longer specify `max_iters` or `val_interval`.
The trainer derives internal optimizer counters from the actual dataset/world size:

| GPUs | Batch/GPU | Global batch | Updates/epoch | Updates for 20 epochs | Padding/epoch |
|---|---:|---:|---:|---:|---:|
| 2 | 4 | 8 | 1062 | 21240 | 6 |
| 1 | 4 | 4 | 2123 | 42460 | 2 |

Both datasets have 8490 training images. Each epoch has its own shuffled
permutation, covers every image, and repeats only the minimal number of images
in the final global batch to keep every rank at batch4. No batch crosses into
another epoch. Augmentation seeds and epoch sampling support deterministic resume.

Warmup covers the first two epochs (10%); cosine decay spans the remaining
training. CAFBR is disabled for epochs 1-10 and enabled from epoch 11.
Epoch-end evaluation uses all 975 images from the configured test split. Its
metrics are recorded in both validation (checkpoint selection) and test history;
this is one evaluation pass because both roles use the same split. Final best
checkpoint testing is retained. Checkpoint ranking, CSV/JSON, console, plots and
W&B expose epoch numbers; checkpoint filenames also retain internal step counts.
Existing step-mode configs/checkpoints retain their original behavior.

```bash
bash start_train.sh

# Override the epoch target, retaining batch4 and --wandb:
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh --max-epochs 20 --wandb

# Three miniature epochs on real data, including CAFBR off/on:
bash scripts/train/CloudSEN12_L1C/train_fmamba_cafbr_base16_down5.sh \
  --smoke-test --work-dir experiments/CloudSEN12_L1C/FMamba_CAFBR_base16_epoch_smoke
```

The following updated budgets extrapolate previously measured batch4 throughput
to 20 epochs. They include the same 20 full epoch-end tests, final best test,
checkpoint overhead and a 10-25% reserve; they are projections, not fresh
20-epoch timing measurements.

| Base | Dataset | Suggested hours for 20 epochs |
|---|---|---:|
| 16 | CloudSEN12_L1C | 4.4–5.1 |
| 16 | CloudSEN12_L2A | 4.5–5.1 |
| 32 | CloudSEN12_L1C | 8.0–9.1 |
| 32 | CloudSEN12_L2A | 8.0–9.1 |
| 64 | CloudSEN12_L1C | 14.2–16.1 |
| 64 | CloudSEN12_L2A | 14.3–16.3 |

All six sequential jobs: 53.5–60.9 hours (2.2–2.5 days).

Projection: `docs/fmamba_epoch_timing_projection.json`. Epoch/DDP smoke and
resume checks: `docs/fmamba_epoch_ddp_smoke.json`.
All six variants passed three-epoch real-data DDP smoke tests; 28 related
regression tests passed. Epoch-end resume restored the optimizer counters,
schedule and per-rank RNG; final test metrics matched. CUDA training is not
bitwise deterministic: the largest intermediate mIoU difference was 0.00173
percentage points. W&B epoch logging was verified in offline mode.
