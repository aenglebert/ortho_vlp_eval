import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def post_process_object_detection(
        outputs, threshold=0.5, target_sizes=None
):
    out_logits, boxes = outputs.logits, outputs.pred_boxes

    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})

    return results


class Detection(LightningModule):
    def __init__(self,
                 detection_model,
                 freeze_encoder=0,
                 **kwargs,
                 ):
        """
        Args:
        :param detection_model: The object detection model
        """
        super().__init__()

        self.detection_model = detection_model
        self.freeze_encoder = freeze_encoder

        self.val_map = MeanAveragePrecision(
            box_format="xywh",
            class_metrics=True,
        )

        self.test_map = MeanAveragePrecision(
            box_format="xywh",
            class_metrics=True,
        )

        self.save_hyperparameters(ignore=["detection_model"])

    def training_step(self, batch, batch_idx):

        output = self.detection_model(**batch)

        for key in output.loss_dict:
            self.log("train/" + key, output.loss_dict[key], batch_size=self.hparams.batch_size)

        self.log("train/loss", output.loss, batch_size=self.hparams.batch_size)

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.detection_model(**batch)

        for key in output.loss_dict:
            self.log("val/" + key, output.loss_dict[key], batch_size=self.hparams.batch_size)

        self.log("val/loss", output.loss, batch_size=self.hparams.batch_size)

        preds = post_process_object_detection(output)
        target = [{"labels": item_dict["class_labels"], "boxes": item_dict["boxes"]} for item_dict in batch["labels"]]

        self.val_map.update(preds, target)

        return output.loss

    def on_validation_epoch_end(self):
        map_result = self.val_map.compute()

        for key in map_result:
            # when the the result is a tensor, we want to log it to wandb
            if map_result[key].numel() == 1:
                self.log('val/' + key, map_result[key], batch_size=self.hparams.batch_size)

    def test_step(self, batch, batch_idx):
        output = self.detection_model(**batch)

        for key in output.loss_dict:
            self.log("test/" + key, output.loss_dict[key], batch_size=self.hparams.batch_size)

        self.log("test/loss", output.loss, batch_size=self.hparams.batch_size)

        preds = post_process_object_detection(output)
        target = [{"labels": item_dict["class_labels"], "boxes": item_dict["boxes"]} for item_dict in batch["labels"]]

        self.test_map.update(preds, target)

        return output.loss

    def on_test_epoch_end(self):
        map_result = self.test_map.compute()

        for key in map_result:
            if map_result[key].numel() == 1:
                self.log('test/' + key, map_result[key])

    def configure_optimizers(self):

        if self.freeze_encoder != 0:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.hparams.detector_learning_rate,
                                          weight_decay=self.hparams.weight_decay,
                                          )
            self.detection_model.owlvit.requires_grad_(False)

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
            self.detection_model.owlvit.requires_grad_(True)
            self.freeze_encoder = 0

            # Set the learning rate to the encoder learning rate
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = self.hparams.encoder_learning_rate
