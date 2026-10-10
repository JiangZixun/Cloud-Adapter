"""CAFBR FMamba training; select a dataset/head via its JSON configuration."""

from pathlib import Path

from torch.distributed.elastic.multiprocessing.errors import record

from train_unet import main



@record
def run():
    main(Path(__file__).resolve().parents[1] / "configs/fmamba/hrc_whu_native.json")


if __name__ == "__main__":
    run()
