import numpy as np

import torch

from pytorch_lightning import LightningModule

from torchmetrics import R2Score, MeanAbsoluteError


# define the LightningModule
class RegressionEncoder(LightningModule):
    def __init__(self,
                 vision_model=None,
                 pool_image=False,
                 freeze_encoder=0,
                 dropout=0.2,
                 head_learning_rate=1e-4,
                 encoder_learning_rate=1e-6,
                 weight_decay=1e-6,
                 ):
        """
        Args:
        :param vision_model: The vision model to use as encoder
        :param pool_image: whether to pool the images or not if not performed by the encoder
        :param freeze_encoder: number of steps with the encoder frozen (-1 to freeze forever)
        :param head_learning_rate: learning rate for the linear layer
        :param encoder_learning_rate: learning rate for the encoder
        """
        super().__init__()

        assert vision_model is not None, "vision_model should be defined"

        self.vision_model = vision_model
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(vision_model.config.hidden_size, 1)

        # Use MSE loss for regression
        #self.loss = torch.nn.MSELoss()

        # Use MAE loss for regression
        self.loss = torch.nn.L1Loss()

        # R2 score metrics
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # MAD score metrics
        self.train_mad = MeanAbsoluteError()
        self.val_mad = MeanAbsoluteError()
        self.test_mad = MeanAbsoluteError()

        self.freeze_encoder = freeze_encoder

        self.save_hyperparameters(ignore=["vision_model"])

    def common_step(self, batch):
        images = batch["image"]
        targets = batch["target"]

        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        vision_outputs = self.vision_model(images)

        image_embeds = vision_outputs.last_hidden_state[:, 0, :]

        return image_embeds, targets

    def training_step(self, batch, batch_idx):
        image_embeds, targets = self.common_step(batch)

        image_embeds = self.dropout(image_embeds)

        logits = self.head(image_embeds)

        loss = self.loss(logits, targets.float())

        batch_size = targets.shape[0]

        self.log('train/loss', loss, batch_size=batch_size)

        # Add to metrics
        self.train_r2(logits, targets)
        self.train_mad(logits, targets)

        return loss

    def on_train_epoch_end(self):
        self.log('train/r2', self.train_r2.compute())
        self.log('train/mad', self.train_mad.compute())

    def validation_step(self, batch, batch_idx):
        image_embeds, targets = self.common_step(batch)

        logits = self.head(image_embeds)

        loss = self.loss(logits, targets.float())

        batch_size = targets.shape[0]

        self.log('val/loss', loss, batch_size=batch_size)

        # Add metrics
        self.val_r2(logits, targets)
        self.val_mad(logits, targets)

        return loss

    def on_validation_epoch_end(self):
        self.log('val/r2', self.val_r2.compute())
        self.log('val/mad', self.val_mad.compute())

    def test_step(self, batch, batch_idx):
        # At test time, always pool the images (evaluating the model on the whole study)
        image_embeds, targets = self.common_step(batch)

        logits = self.head(image_embeds)

        loss = self.loss(logits, targets.float())

        batch_size = targets.shape[0]

        self.log('test/loss', loss, batch_size=batch_size)

        # Add metrics
        self.test_r2(logits, targets)
        self.test_mad(logits, targets)

        return loss

    def on_test_epoch_end(self):
        self.log('test/r2', self.test_r2.compute())
        self.log('test/mad', self.test_mad.compute())

    def configure_optimizers(self):

        if self.freeze_encoder != 0:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.head_learning_rate,
                                          weight_decay=self.hparams.weight_decay,
                                          )
            self.vision_model.requires_grad_(False)
        else:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.encoder_learning_rate,
                                          weight_decay=self.hparams.weight_decay,
                                          )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=3,
                                                               verbose=True,
                                                               )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"}]

    # Unfreeze the encoder after a certain number of steps
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step == self.freeze_encoder:
            print("Unfreezing the encoder")
            self.vision_model.requires_grad_(True)
            self.freeze_encoder = 0

            # Set the learning rate to the encoder learning rate
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = self.hparams.encoder_learning_rate