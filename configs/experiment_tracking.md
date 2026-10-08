# Progress, metrics and W&B

All three launchers under `scripts/train/HRC_WHU/` use the shared trainer with
tqdm overall iteration progress (including resume), current loss/LR and ETA.
Separate progress bars show train-split evaluation, validation and final testing.
Training loss formulas and checkpoint selection are unchanged.

## Cloud logging

W&B is opt-in. The selected qwen3 environment includes the SDK. Login once if
credentials are not already configured, then add `--wandb` to any launcher:

```bash
/home/jzx/anaconda3/envs/qwen3/bin/python -m wandb login
bash scripts/train/HRC_WHU/train_unet.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr.sh --wandb
bash scripts/train/HRC_WHU/train_fmamba_cafbr_mask2former.sh --wandb
```

Default project: `Cloud-Adapter`. Optional arguments: `--wandb-project`,
`--wandb-entity` (account/team), `--wandb-name`, and `--wandb-mode offline` for
local W&B testing. Normal `--wandb` uses online cloud logging. A saved W&B run ID
is reused when resuming. `results.json` contains the cloud run URL.
No API key is stored in experiment JSON or supplied by the launchers.

Cloud series include `optimization/loss`, `train/loss`, `validation/loss`,
`test/loss`, all seven aggregate metrics and per-class metrics. The horizontal
axis is `iteration`; the final test point uses the best checkpoint's iteration.
Local curves and a metrics-only JSON artifact are uploaded at completion.
Only model/dataset names and numerical hyperparameters enter the W&B config;
filesystem paths, code/environment/data manifests and checkpoints remain local.
SDK console capture, Git/code collection, environment metadata and requirements
upload are disabled. Compatible optional dependencies are listed in
`requirements-tracking.txt`.
The standalone `tools/test_fmamba.py` also accepts `--wandb`.

## Local traceability and visualization

Under `experiments/HRC_WHU/<Model>/`:

```text
results.json         # one JSON document containing all splits and trace metadata
curves/loss.png      # optimization, train evaluation, validation and final test loss
curves/metrics.png   # seven aggregate metric curves, including final test markers
```

`results.json` arrays:

* `training`: augmented optimization-batch windows, recorded every `log_interval`
  steps and at each validation. Contains objective loss, LR, batch range and
  aggregate/per-class metrics for those batches; weights evolve within a window.
* `train`: at each validation interval, evaluate every training sample in eval
  mode with augmentation disabled (in smoke runs, the configured 8-sample subset).
* `validation`: evaluate the complete 30-image original test split each interval,
  following the requested HRC-WHU protocol.
* `test`: evaluate the best checkpoint at completion; standalone tests append
  additional timestamped entries here.

Each evaluation records configured objective `loss`, pixel `ce_loss`, all seven
aggregate metrics, per-class metrics, confusion matrix and sample/timing data.
The legacy `val_loss` field remains pixel CE for compatibility. Evaluation
objective loss averages batch losses weighted by image count. Mask2Former
evaluation uses the same matched query/auxiliary loss as training; its point
sampling is seeded without changing the training RNG.

Trace metadata includes run ID, timestamps, per-session interpreter/config/code
SHA256 hashes, hashes of environment/config/data-manifest files, checkpoint
ranking/retention, and the exact best/test checkpoint path plus SHA256. Histories
survive resume. Previous histories from before this feature are not reconstructed;
recording starts when using the new trainer. The original CSV/JSONL records and
per-validation JSON files are retained.

Train and validation loss curves use the same configured objective, evaluated in
different modes/data. HRC-WHU validation and final testing use the same original
test images as requested; both roles are explicitly identified in the JSON.
