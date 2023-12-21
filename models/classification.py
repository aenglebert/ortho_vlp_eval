import numpy as np

import torch

import torchmetrics

from pytorch_lightning import LightningModule

from sklearn.metrics import roc_auc_score


class BinaryAUROC:
    """
    Compute the auroc
    """
    def __init__(self):

        self.preds = []
        self.target = []

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        return self.update(preds, target)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric
        Keep track of the preds and target to compute the confidence interval at the end
        :param preds: predictions
        :param target: target
        :return:
        """

        self.preds.append(preds.cpu().detach())
        self.target.append(target.cpu().detach())

    def compute(self, keep=False):
        """
        Compute the metric
        :return:
        """
        # Concatenate all the predictions and targets
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        # Compute the auroc with confidence interval
        auroc = roc_auc_score(target, preds)

        if not keep:
            self.preds = []
            self.target = []

        return auroc


# define the LightningModule
class ClassificationEncoder(LightningModule):
    def __init__(self,
                 vision_model=None,
                 n_classes=1,
                 pool_image=False,
                 freeze_encoder=0,
                 dropout=0.2,
                 classifier_learning_rate=1e-4,
                 encoder_learning_rate=1e-6,
                 weight_decay=1e-6,
                 ):
        """
        Args:
        :param vision_model: The vision model to use as encoder
        :param n_classes: number of classes to predict
        :param pool_image: whether to pool the images or not if not performed by the encoder
        :param freeze_encoder: number of steps with the encoder frozen (-1 to freeze forever)
        :param classifier_learning_rate: learning rate for the linear layer
        :param encoder_learning_rate: learning rate for the encoder
        """
        super().__init__()

        assert vision_model is not None, "vision_model should be defined"

        self.vision_model = vision_model
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(vision_model.config.hidden_size, n_classes)
        self.pool_image = pool_image

        assert isinstance(n_classes, int), "n_classes should be an integer"
        assert n_classes >= 1, "n_classes should be equal or greater than 1"

        if n_classes == 1:
            # Use BCE loss
            self.loss = torch.nn.BCEWithLogitsLoss()
            task = "binary"
        else:
            # Use cross entropy loss
            self.loss = torch.nn.CrossEntropyLoss()
            task = "multiclass"

        self.train_acc = torchmetrics.Accuracy(task=task)
        self.val_acc = torchmetrics.Accuracy(task=task)
        self.test_acc = torchmetrics.Accuracy(task=task)
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        self.freeze_encoder = freeze_encoder

        self.save_hyperparameters(ignore=["vision_model"])

    def common_step(self, batch, pool_image=False):
        images = batch["images"]
        labels = batch["labels"].unsqueeze(-1)

        if "pooling_matrix" in batch.keys() and batch["pooling_matrix"] is not None:
            pooling_matrix = batch["pooling_matrix"]
        else:
            assert images.shape[0] == labels.shape[
                0], "labels and images should have the same batch size when no pooling_matrix is provided"
            pooling_matrix = torch.eye(images.shape[0], device=labels.device)

        if hasattr(self.vision_model.config, 'seq_model') and self.vision_model.config.seq_model:
            vision_outputs = self.vision_model(images,
                                               pooling_matrix=pooling_matrix,
                                               )
            image_embeds = vision_outputs.pooler_output

        else:
            vision_outputs = self.vision_model(images)

            image_embeds = vision_outputs.last_hidden_state[:, 0, :]

            if pool_image:
                image_embeds = pooling_matrix @ image_embeds / pooling_matrix.sum(dim=-1, keepdim=True)
            else:
                labels = labels @ pooling_matrix.type_as(labels)

        return image_embeds, labels

    def training_step(self, batch, batch_idx):
        image_embeds, labels = self.common_step(batch, pool_image=self.pool_image)

        image_embeds = self.dropout(image_embeds)

        logits = self.classifier(image_embeds)

        loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('train/loss', loss, batch_size=batch_size)

        self.train_acc(logits, labels)

        self.log('train/acc', self.test_acc, on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        image_embeds, labels = self.common_step(batch, pool_image=self.pool_image)

        logits = self.classifier(image_embeds)

        loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('val/loss', loss, batch_size=batch_size)
        self.val_acc(logits, labels)
        self.val_auroc(torch.sigmoid(logits), labels)

        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        self.log('val/auroc_epoch', auroc)
        self.log('val/acc_epoch', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        # At test time, always pool the images (evaluating the model on the whole study)
        image_embeds, labels = self.common_step(batch, pool_image=True)

        logits = self.classifier(image_embeds)

        loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('test/loss', loss, batch_size=batch_size)

        self.test_acc(logits, labels)

        self.test_auroc(logits, labels)

        return loss

    def on_test_epoch_end(self):
        auroc = self.test_auroc.compute()
        self.log('test/auroc_epoch', auroc)
        self.log('test/acc_epoch', self.test_acc.compute())

    def configure_optimizers(self):

        if self.freeze_encoder != 0:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.classifier_learning_rate,
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
