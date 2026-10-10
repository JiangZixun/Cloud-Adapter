#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
# L1C base32: RGB / 255, 30 epochs, batch4/GPU, test every epoch, W&B.
# Rank 0 stays on the terminal for in-place tqdm updates.
exec "${PYTHON:-/root/anaconda3/envs/qwen3/bin/python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --log-dir "${DDP_LOG_DIR:-experiments/ddp_logs/FMamba_CAFBR_base32_down5_minmax}" --redirects 1:3 \
  tools/train_fmamba.py \
  --wandb --config configs/fmamba/cloudsen12_l1c_native_base32_down5_minmax.json "$@"
