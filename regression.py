import torch
import numpy as np
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pytorch_lightning import Trainer

from models import RegressionEncoder

torch.set_float32_matmul_precision('high')

import datasets

import os
from pathlib import Path

import time

# Try to import lovely_tensors
try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ModuleNotFoundError:
    print("lovely_tensors not found, skipping monkey patching")
    # But not mandatory, pass if lovely tensor is not available
    pass


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base="1.3", config_path="config", config_name="regression")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    # Set Wandb logger
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.experiment_name)

    # Instantiate the dataset
    datamodule = instantiate(cfg.dataset)
    datamodule.setup()

    # Instantiate the model
    vision_model = instantiate(cfg.vision_model)

    # Create the model
    model = RegressionEncoder(vision_model=vision_model,
                              dropout=cfg.head_dropout,
                              head_learning_rate=cfg.head_learning_rate,
                              encoder_learning_rate=cfg.encoder_learning_rate,
                              freeze_encoder=cfg.freeze_encoder,
                              weight_decay=cfg.weight_decay,
                              scale=datamodule.std,
                              bias=datamodule.mean,
                              target_dim=datamodule.target_dim,
                              head_additional_dim=datamodule.head_input_dim,
                              )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )

    # LR monitor callback
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Early stopping callback
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=10)
    callbacks = [early_stopping, checkpoint_callback, lr_monitor]

    # Instantiate the trainer
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=cfg.accelerator,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    # Train the model
    trainer.fit(model,
                datamodule=datamodule,
                )

    # Evaluate the model
    trainer.test(model,
                 datamodule=datamodule,
                 ckpt_path="best",
                 )


if __name__ == "__main__":
    main()
