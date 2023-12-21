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

from models import ClassificationEncoder

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


@hydra.main(version_base="1.3", config_path="config", config_name="classification")
def main(cfg: DictConfig):
    # Seed everything
    seed_everything(cfg.seed)

    # Set Wandb logger
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.experiment_name)

    # Instantiate the dataset
    datamodule = instantiate(cfg.dataset)

    # Instantiate the model
    vision_model = instantiate(cfg.vision_model)

    # Create the classifier
    classifier = ClassificationEncoder(vision_model=vision_model,
                                       n_classes=datamodule.n_classes,
                                       pool_image=cfg.pool_image,
                                       dropout=cfg.classifier_dropout,
                                       classifier_learning_rate=cfg.classifier_learning_rate,
                                       encoder_learning_rate=cfg.encoder_learning_rate,
                                       freeze_encoder=cfg.freeze_encoder,
                                       weight_decay=cfg.weight_decay,
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
    )

    # Train the classifier
    trainer.fit(classifier,
                datamodule=datamodule,
                )

    # Evaluate the classifier
    trainer.test(classifier,
                 datamodule=datamodule,
                 ckpt_path="best",
                 )


if __name__ == "__main__":
    main()
