#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
# Configuration: 20 epochs, test after each epoch, batch4/GPU, no accumulation.
exec "${PYTHON:-/root/anaconda3/envs/qwen3/bin/python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-2}" tools/train_fmamba.py \
  --config configs/fmamba/cloudsen12_l2a_native_base64_down5.json "$@"
