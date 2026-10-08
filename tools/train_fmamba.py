"""HRC-WHU CAFBR FMamba training; select a head via its JSON configuration."""

from pathlib import Path

from train_unet import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "configs/fmamba/hrc_whu_native.json")
