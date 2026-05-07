from typing import List, Any

import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Timer
from omegaconf import DictConfig, OmegaConf, ListConfig, base
import torch
from collections import defaultdict
OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])

@hydra.main(version_base="1.3", config_path="./configs", config_name="eval")
def main(cfg: DictConfig):
    """
    Entry point for training the model.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: List[Logger] = [
        hydra.utils.instantiate(logger) for logger in cfg.get("loggers").values()
    ]

    # Log hyperparameters
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    # Initialize the model
    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    timer = Timer()
    callbacks.append(timer)

    # Initialize the trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )

    ckpt_path = cfg.get("ckpt_path")
    base_ckpt_path = cfg.get("base_ckpt_path")

    if ckpt_path is not None:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        if "lora_config" in checkpoint:
            if not base_ckpt_path:
                raise ValueError(
                    "LoRA evaluation requires `base_ckpt_path` to point to the full base checkpoint."
                )
            base_state_dict = torch.load(
                base_ckpt_path,
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
            model.load_state_dict(base_state_dict)
            model.enable_lora(ckpt_path)
            ckpt_path = None

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print("Time taken: ", timer.time_elapsed("test"))


if __name__ == "__main__":
    main()
