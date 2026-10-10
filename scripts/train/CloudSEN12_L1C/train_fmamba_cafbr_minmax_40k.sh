#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
# Original 40000-update L1C model; fixed RGB min-max normalization: x / 255.
exec "${PYTHON:-/root/anaconda3/envs/qwen3/bin/python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --log-dir "${DDP_LOG_DIR:-experiments/ddp_logs/FMamba_CAFBR_minmax_40k}" --tee 3 \
  tools/train_fmamba.py \
  --wandb --config configs/fmamba/cloudsen12_l1c_native_minmax_40k.json "$@"
