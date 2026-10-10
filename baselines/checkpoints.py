"""Atomic checkpoint writes and explicit top-k ranking by validation score."""

import json
import math
import os
import shutil
from pathlib import Path

import torch


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


class CheckpointManager:
    def __init__(self, directory, keep=3, metric="mIoU"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep, self.metric = keep, metric
        manifest = self.directory / "top_k.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if metadata["metric"] != metric or metadata["keep"] != keep:
                raise ValueError("Checkpoint ranking settings differ from the saved run")
            self.top = metadata["checkpoints"]
        else:
            self.top = []

    def update(self, state, metrics):
        save_checkpoint(self.directory / "last.pth", state)
        score = metrics[self.metric]
        if score is None or not math.isfinite(score):
            raise ValueError(f"Undefined checkpoint selection metric: {self.metric}")
        candidate = {"file": f"iter_{state['iteration']:07d}.pth",
                     "iteration": state["iteration"], "score": score}
        if "epoch" in state:
            candidate["epoch"] = state["epoch"]
        ranked = sorted(self.top + [candidate], key=lambda row: (-row["score"], row["iteration"]))[:self.keep]
        if candidate in ranked:
            save_checkpoint(self.directory / candidate["file"], state)
        old = self.top
        self.top = ranked
        if candidate == ranked[0]:
            temporary = self.directory / "best.tmp"
            shutil.copyfile(self.directory / candidate["file"], temporary)
            os.replace(temporary, self.directory / "best.pth")
        write_json(self.directory / "top_k.json", {
            "metric": self.metric, "rule": "greater", "keep": self.keep, "checkpoints": ranked
        })
        for row in old:
            if row not in ranked:
                (self.directory / row["file"]).unlink()
