# LS-Mamba on HRC-WHU

The architecture comes from Qwen3's
`external/ls_mamba_paper1/model/localvmunet2/Local_VMUNet2`, identified as
LS-Mamba by its reproducibility manifest. The local implementation lives in
`baselines/lsmamba/`; `provenance.json` records source paths and SHA256 hashes.
No runtime imports or checkpoints from Qwen3 are required.

Both variants preserve 16 hybrid encoder/decoder blocks: each combines spectral
Mamba with learned soft channel grouping, four-direction local window scans,
and four-direction global scans. Channels are 96/192/384/768, encoder and decoder
depths are 2/2/2/2, and drop-path is 0.2. RGB input and two output classes replace
the source experiment's 16-channel/10-class setting. Models initialize from scratch.

The spectral branch vendors the installed source environment's official Mamba-1
initialization and unfused full-sequence forward path, using PyTorch causal
depthwise Conv1d and the original CUDA selective-scan core plus SiLU gating.
This avoids requiring the old `mamba_ssm`/`causal_conv1d` binary wheels in qwen3.
Spatial branches retain their original CUDA core scan and local/global ordering.
The scan wrapper also accepts the spectral branch's ungrouped B/C tensors.

* `LSMamba`: original patch-expansion decoder and final classifier. The configured
  loss is 5 x unweighted pixel CE + 5 x multiclass soft Dice, using the source
  Dice's squared probability/target denominator, epsilon 1e-5 and both classes.
  Labels 255 are excluded from both terms. Qwen3's current `main.py` uses a custom
  attention-weighted CE for its active experiment; this HRC-WHU adaptation explicitly
  uses ordinary CE, rather than transferring its 10-class weighting scheme.
* `LSMamba_Mask2Former`: all four original decoded stages feed the shared official
  Transformers Mask2Former head at strides 4/8/16/32. The unused final patch expansion
  and classifier are removed. Matching, auxiliary supervision, and CE/mask/Dice
  weights 2/5/5 match the existing CAFBR Mask2Former variant.

The shared pipeline uses 120 training images and 30 test images; the test split
also supplies validation and checkpoint selection. Settings are 256x256 crops,
batch 4, AdamW lr 1e-4, weight decay 0.05, 10% linear warmup followed by cosine decay, 40,000 iterations,
validation every 2,000, seed 42, FP32 and four workers. This transfers the
architecture into the current experiment protocol; it does not reproduce the
original paper's dataset or reported scores. LS-Mamba has no CAFBR module or
delayed CAFBR schedule.

```bash
bash scripts/train/HRC_WHU/train_lsmamba.sh --wandb
bash scripts/train/HRC_WHU/train_lsmamba_mask2former.sh --wandb

/root/anaconda3/envs/qwen3/bin/python tools/test_lsmamba.py \
  experiments/HRC_WHU/LSMamba_warmup_cosine/checkpoints/best.pth

bash scripts/train/HRC_WHU/train_lsmamba.sh \
  --resume experiments/HRC_WHU/LSMamba_warmup_cosine/checkpoints/last.pth
```

Outputs are `experiments/HRC_WHU/LSMamba_warmup_cosine/` and
`experiments/HRC_WHU/LSMamba_Mask2Former_warmup_cosine/`. Both support tqdm, validation-loss
printing, optional W&B, complete `results.json` split histories, local loss/metric
plots, all seven existing segmentation metrics and per-class confusion records.
Checkpoints keep the top three mIoUs plus last and best; training ends by testing
best. Standalone testing and resume use the same shared pipeline.

To smoke-test either launcher, add `--smoke-test --work-dir <separate-directory>`.
The full LS-Mamba architectures remain enabled; training uses eight real images,
and each validation evaluates all 30 test images. Unit tests additionally check
spectral CUDA outputs/gradients against a direct recurrence, local scan/reverse
including padding, source-form Dice, and gradients in every hybrid branch.

Native-model timing measured on 2026-10-08 with qwen3 and the RTX 5090 D
(before rescaling the CE/Dice weights from 0.5/0.5 to 5/5):
10 warmup iterations followed by 60 timed iterations, using the full 120-image
training split, 256x256 crops, batch 4, four workers and FP32. Mean training step
time was 0.13681 s (median 0.13606 s); full train evaluation took 1.539 s and
30-image test/validation took 0.532 s. Extrapolating 40,000 steps, ten full
train/test evaluations, checkpoint writes and final best testing gives about
5,500 s (1 h 32 min). Allow 1 h 40 min to 2 h for logging, system load and
long-run variation. Peak allocated GPU memory was 8.06 GiB. This is a short-run
measurement, not a completed full training run. The small timing report is
saved at `experiments/HRC_WHU/LSMamba_benchmark/timing_estimate.json`; benchmark
checkpoint files are automatically removed.

Completed PolyLR runs retain their saved configuration. To evaluate/resume an
older run, use its saved `config.json` rather than the new default config.
The new schedule requires a fresh run; the shared trainer rejects a scheduler
change during resume. `lr_schedule="cosine"`, `warmup_ratio=0.1`,
`warmup_start_factor=0.01`, `min_lr_ratio=0.0` are the current HRC defaults.
