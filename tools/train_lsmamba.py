"""Train LS-Mamba on HRC-WHU using the shared experiment pipeline."""

from pathlib import Path

from train_unet import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "configs/lsmamba/hrc_whu_native.json")
