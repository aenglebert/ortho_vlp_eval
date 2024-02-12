import numpy as np

import torch

import torchmetrics
from torchmetrics.classification import AUROC, Accuracy

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


class MultiLabelClassifier(torch.nn.Module):
    def __init__(self,
                 input_size,
                 n_classes,
                 n_labels,
                 ):
        super().__init__()

        self.classifier_list = torch.nn.ModuleList([torch.nn.Linear(input_size, n_classes) for _ in range(n_labels)])

    def forward(self, x):
        # We assume that the input is a tensor of shape (batch_size, input_size)
        # We return a tensor of shape (batch_size, n_labels, n_classes)
        logits = torch.stack([classifier(x) for classifier in self.classifier_list], dim=1)
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return logits


# define the LightningModule
class ClassificationEncoder(LightningModule):
    def __init__(self,
                 vision_model=None,
                 n_classes=1,
                 n_labels=1,
                 pool_image=False,
                 freeze_encoder=0,
                 dropout=0.2,
                 classifier_learning_rate=1e-4,
                 encoder_learning_rate=1e-6,
                 weight_decay=1e-6,
                 pos_weight=None,
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

        if n_labels is None:
            n_labels = 1

        self.vision_model = vision_model
        self.dropout = torch.nn.Dropout(dropout)

        self.n_classes = n_classes
        self.n_labels = n_labels

        self.classifier = MultiLabelClassifier(self.vision_model.config.hidden_size,
                                               n_classes,
                                               n_labels,
                                               )

        self.pool_image = pool_image

        assert isinstance(n_classes, int), "n_classes should be an integer"
        assert n_classes >= 1, "n_classes should be equal or greater than 1"

        if n_classes == 1:
            # Use BCE loss
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.task = "binary"
        else:
            # Use cross entropy loss
            self.loss = torch.nn.CrossEntropyLoss()
            self.task = "multiclass"

        self.train_acc = torchmetrics.Accuracy(task=self.task, num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task=self.task, num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task=self.task, num_classes=n_classes)

        self.val_auroc = torchmetrics.AUROC(task=self.task, num_classes=n_classes)
        self.test_auroc = torchmetrics.AUROC(task=self.task, num_classes=n_classes)

        self.val_f1 = torchmetrics.F1Score(task=self.task, num_classes=n_classes)
        self.test_f1 = torchmetrics.F1Score(task=self.task, num_classes=n_classes)

        self.freeze_encoder = freeze_encoder

        self.save_hyperparameters(ignore=["vision_model"])

    def common_step(self, batch, pool_image=False):
        images = batch["images"]
        labels = batch["labels"]

        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)

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

        if self.task == "binary":
            loss = self.loss(logits, labels.float())
        else:
            logits = logits.view(-1, self.n_classes)
            labels = labels.view(-1).long()
            loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('train/loss', loss, batch_size=batch_size)

        self.train_acc(logits, labels)

        # Log acc on step and epoch, if multilabel, log for each class
        #if self.hparams.n_classes == 1:
        #    self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, batch_size=batch_size)
        #else:
        #    for i in range(self.hparams.n_classes):
        #        self.log(f'train/acc_{i}', self.train_acc[i], on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.train_acc.reset()

        # Log acc on step and epoch
        self.log('train/acc', acc, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        image_embeds, labels = self.common_step(batch, pool_image=self.pool_image)

        logits = self.classifier(image_embeds)

        if self.task == "binary":
            loss = self.loss(logits, labels.float())
        else:
            logits = logits.view(-1, self.n_classes)
            labels = labels.view(-1).long()
            loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('val/loss', loss, batch_size=batch_size)

        self.val_acc(logits, labels)
        if self.task == "binary":
            self.val_auroc(torch.sigmoid(logits), labels)
            self.val_f1(torch.sigmoid(logits), labels)
        else:
            self.val_auroc(torch.softmax(logits, dim=-1), labels)
            self.val_f1(torch.softmax(logits, dim=-1), labels)

        return loss

    def on_validation_epoch_end(self):
        # Log acc and auroc on step and epoch
        self.log('val/acc', self.val_acc.compute(), on_epoch=True)
        self.log('val/auroc', self.val_auroc.compute(), on_epoch=True)
        self.log('val/f1', self.val_f1.compute(), on_epoch=True)

        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        # At test time, always pool the images (evaluating the model on the whole study)
        image_embeds, labels = self.common_step(batch, pool_image=True)

        logits = self.classifier(image_embeds)

        if self.task == "binary":
            loss = self.loss(logits, labels.float())
        else:
            logits = logits.view(-1, self.n_classes)
            labels = labels.view(-1).long()
            loss = self.loss(logits, labels)

        batch_size = labels.shape[0]

        self.log('test/loss', loss, batch_size=batch_size)

        self.test_acc(logits, labels)

        if self.task == "binary":
            self.test_auroc(torch.sigmoid(logits), labels)
            self.test_f1(torch.sigmoid(logits), labels)
        else:
            self.test_auroc(torch.softmax(logits, dim=-1), labels)
            self.test_f1(torch.softmax(logits, dim=-1), labels)

        return loss

    def on_test_epoch_end(self):
        # Log acc and auroc after the test epoch
        self.log('test/acc', self.test_acc.compute())
        self.log('test/auroc', self.test_auroc.compute())
        self.log('test/f1', self.test_f1.compute())

        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_f1.reset()

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
