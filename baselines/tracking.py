"""Traceable split histories, local figures, and opt-in W&B logging."""

import hashlib
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import write_json
from .metrics import METRIC_NAMES
from .data import dataset_splits


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_wandb_arguments(parser):
    parser.add_argument("--wandb", action="store_true", help="Upload loss/metric history to W&B")
    parser.add_argument("--wandb-project", default="Cloud-Adapter")
    parser.add_argument("--wandb-entity", help="W&B account or team; otherwise use the logged-in default")
    parser.add_argument("--wandb-name", help="W&B run display name")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online",
                        help="Online uploads to cloud; offline is available for local smoke tests")


class ExperimentHistory:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "results.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            splits = dataset_splits(config)
            self.data = dict(schema_version=1, run_id=uuid.uuid4().hex, created_at=timestamp(),
                             config=config, protocol=dict(train="original train split",
                             validation=f"original {splits['validation']} split", test=f"original {splits['test']} split; final best checkpoint",
                             selection_metric=config["selection_metric"],
                             train_loss_only=config.get("train_loss_only", False),
                             metric_unit="percent", evaluation_augmentation=False),
                             training=[], train=[], validation=[], test=[], sessions=[])
        self.data["sessions"].append(dict(started_at=timestamp(), executable=sys.executable, config=config))
        root = Path(__file__).resolve().parents[1]
        files = [*root.glob("baselines/**/*.py"), *root.glob("tools/*unet.py"),
                 *root.glob("tools/*fmamba.py"), *root.glob("tools/*lsmamba.py")]
        self.data["sessions"][-1]["code_sha256"] = {
            str(path.relative_to(root)): file_digest(path) for path in sorted(files)
        }
        self.save()

    def save(self):
        self.data["updated_at"] = timestamp()
        write_json(self.path, self.data)

    def record(self, split, iteration, metrics, **context):
        entry = dict(iteration=iteration, recorded_at=timestamp(), **metrics, **context)
        self.data[split].append(entry)
        self.save()
        return entry

    def attach_files(self):
        self.data["artifacts"] = {
            name: dict(path=name, sha256=file_digest(self.directory / name))
            for name in ("config.json", "environment.json", "data_manifest.json")
            if (self.directory / name).exists()
        }
        self.save()

    def render(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figures = self.directory / "curves"
        figures.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(10, 5))
        for split, label in [("training", "Training batches (optimized loss)"),
                             ("train", "Full train split (eval mode)"),
                             ("validation", f"Validation ({self.data['config'].get('validation_split', 'test')} split)")]:
            rows = self.data[split]
            if not rows:
                continue
            axis.plot([row["iteration"] for row in rows], [row["loss"] for row in rows], label=label)
        for index, row in enumerate(self.data["test"]):
            axis.scatter(row["iteration"], row["loss"], marker="*", s=120,
                         label="Test / best checkpoint" if index == 0 else None)
        axis.set(xlabel="Training iteration", ylabel="Loss", title="Loss history (configured training objective)")
        axis.grid(alpha=.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "loss.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for axis, key in zip(axes.flat, METRIC_NAMES):
            for split, label in [("train", "Train"), ("validation", f"Validation ({self.data['config'].get('validation_split', 'test')})")]:
                rows = self.data[split]
                if not rows:
                    continue
                axis.plot([row["iteration"] for row in rows],
                          [float("nan") if row[key] is None else row[key] for row in rows], marker="o", label=label)
            for index, row in enumerate(self.data["test"]):
                if row[key] is not None:
                    axis.scatter(row["iteration"], row[key], marker="*", s=100,
                                 label="Test / best" if index == 0 else None)
            axis.set(title=key, xlabel="Iteration", ylabel="Percent", ylim=(0, 100))
            axis.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8)
        axes.flat[-1].axis("off")
        fig.tight_layout()
        fig.savefig(figures / "metrics.png", dpi=160)
        plt.close(fig)


class WandbTracker:
    def __init__(self, args, history, resume=False):
        self.history, self.run = history, None
        if not args.wandb:
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install wandb using /home/jzx/anaconda3/envs/qwen3/bin/python -m pip install wandb") from error
        previous = history.data.get("wandb")
        run_id = previous["id"] if previous else uuid.uuid4().hex[:8]
        config = history.data["sessions"][-1]["config"]
        # Cloud authorization covers metrics/curves. Keep filesystem paths,
        # environment manifests and code provenance in the local JSON only.
        cloud_config = {key: config[key] for key in (
            "dataset", "model", "classes", "image_size", "base_channels", "batch_size",
            "grad_accum_steps", "effective_batch_size",
            "max_iters", "val_interval", "lr", "weight_decay", "poly_power", "lr_schedule", "min_lr_ratio",
            "warmup_ratio", "warmup_start_factor", "seed", "amp",
            "loss", "fmamba", "lsmamba", "mask2former", "cafbr_start_ratio", "validation_split", "test_split", "train_loss_only"
        ) if key in config}
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, id=run_id,
            name=args.wandb_name or f"{history.data['config']['dataset']}-{history.data['config']['model']}-{history.data['run_id'][:8]}",
            config=cloud_config,
            dir=str(history.directory), mode=args.wandb_mode,
            settings=wandb.Settings(init_timeout=45, disable_code=True, disable_git=True,
                                    console="off", x_disable_meta=True, x_save_requirements=False),
            resume="allow" if resume and previous and args.wandb_mode == "online" else None,
        )
        self.run.define_metric("iteration")
        for split in ("optimization", "train", "validation", "test"):
            self.run.define_metric(f"{split}/*", step_metric="iteration")
        history.data["wandb"] = dict(id=self.run.id, project=args.wandb_project,
                                     mode=args.wandb_mode, url=self.run.url if args.wandb_mode == "online" else None)
        history.save()

    def update_checkpoints(self, ranking, last_iteration):
        self.history.data["checkpoint_ranking"] = ranking
        retained = {row["iteration"] for row in ranking}
        filenames = {row["iteration"]: row["file"] for row in ranking}
        for split in ("train", "validation"):
            for row in self.history.data[split]:
                row["checkpoint_retained"] = row["iteration"] in retained or row["iteration"] == last_iteration
                if row["iteration"] in retained:
                    row["checkpoint"] = f"checkpoints/{filenames[row['iteration']]}"
                elif row["iteration"] == last_iteration:
                    row["checkpoint"] = "checkpoints/last.pth"
                else:
                    row["checkpoint"] = None
        self.history.save()

    def log(self, split, row):
        if self.run is None:
            return
        payload = {"iteration": row["iteration"]}
        for key in ("loss", "ce_loss", "lr", "samples", "batches", "optimizer_updates", "micro_batches_completed", "grad_accum_steps", "effective_batch_size", "cafbr_enabled", "cafbr_start_iteration", *METRIC_NAMES):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                payload[f"{split}/{key}"] = value
        for name, metrics in row.get("per_class", {}).items():
            for key, value in metrics.items():
                if value is not None:
                    payload[f"{split}/per_class/{name}/{key}"] = value
        self.run.log(payload)

    def finish(self):
        if self.run is None:
            return
        import wandb
        for filename in ("loss.png", "metrics.png"):
            path = self.history.directory / "curves" / filename
            if path.exists():
                self.run.log({f"curves/{filename[:-4]}": wandb.Image(str(path))})
        for key in METRIC_NAMES:
            if self.history.data["test"]:
                self.run.summary[f"best_test/{key}"] = self.history.data["test"][-1][key]
        artifact = wandb.Artifact(f"history-{self.history.data['run_id']}", type="metrics")
        fields = ("iteration", "recorded_at", "loss", "ce_loss", "lr", "samples", "batches",
                  "cafbr_enabled", "cafbr_start_iteration", "optimizer_updates",
                  "micro_batches_completed", "grad_accum_steps", "effective_batch_size",
                  "first_iteration", "per_class", "confusion_matrix", *METRIC_NAMES)
        cloud_history = {split: [{key: row[key] for key in fields if key in row}
                                for row in self.history.data[split]]
                         for split in ("training", "train", "validation", "test")}
        metrics_path = self.history.directory / "wandb_metrics.json"
        write_json(metrics_path, cloud_history)
        artifact.add_file(str(metrics_path), name="metrics.json")
        self.run.log_artifact(artifact)
        self.run.finish()
