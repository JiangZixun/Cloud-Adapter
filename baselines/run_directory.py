"""Preserve interrupted runs that never produced a resumable checkpoint."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def prepare_fresh_run(directory):
    directory = Path(directory)
    if not directory.exists() or not any(directory.iterdir()):
        return None
    error = FileExistsError(
        f"Run directory contains saved results: {directory}; "
        "use --resume with checkpoints/last.pth or a new --work-dir"
    )
    if directory.is_symlink():
        raise error
    recognized = {"config.json", "environment.json", "data_manifest.json", "train.log",
                  "results.json", "checkpoints", "validation", "wandb", "curves"}
    if not (directory / "config.json").is_file() or any(
        path.name not in recognized for path in directory.iterdir()
    ):
        raise error
    if any((directory / "checkpoints").rglob("*.pth")) or (directory / "checkpoints/top_k.json").exists():
        raise error
    if (directory / "validation").exists() and any((directory / "validation").iterdir()):
        raise error
    history = directory / "results.json"
    if history.exists():
        try:
            results = json.loads(history.read_text())
        except (ValueError, OSError):
            raise error
        if results.get("validation") or results.get("test"):
            raise error
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    archive = directory.with_name(f"{directory.name}_incomplete_{suffix}")
    directory.rename(archive)
    return archive
