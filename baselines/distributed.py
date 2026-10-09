"""Additional DDP correctness checks for real-model smoke runs."""

import hashlib

import torch.distributed as dist


def verify_parameter_sync(model):
    """Fail on any parameter divergence, including newly activated CAFBR weights."""
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    checksums = [None] * dist.get_world_size()
    dist.all_gather_object(checksums, digest.hexdigest())
    if len(set(checksums)) != 1:
        raise RuntimeError(f"DDP parameters diverged across ranks: {checksums}")
    return checksums[0]
