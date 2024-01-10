import torch
from torch import nn

from transformers import YolosConfig, YolosForObjectDetection
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def create_yolos_from_vit(model_checkpoint,
                          num_classes,
                          num_detection_tokens=100,
                          ):

    # Get config from ViT and change it to Yolos config
    conf = YolosConfig.from_pretrained(model_checkpoint)
    conf.image_size = [conf.image_size, conf.image_size]
    conf.num_labels = num_classes
    conf.num_detection_tokens = num_detection_tokens

    # Create Yolos model using the config
    yolos = YolosForObjectDetection(conf)

    # Load the weights from the ViT model
    state_dict = torch.load(model_checkpoint + "/pytorch_model.bin")

    # Change the keys to match the Yolos model
    for key in list(state_dict.keys()):
        state_dict["vit." + key] = state_dict[key]
        state_dict.pop(key)

    # Change the position embeddings to match the new number of detection tokens
    shape = list(state_dict["vit.embeddings.position_embeddings"].shape)
    shape[1] += conf.num_detection_tokens
    new_emb = torch.zeros(shape)
    num_position_embeddings = state_dict["vit.embeddings.position_embeddings"].shape[1]
    new_emb[:, :num_position_embeddings, :] = state_dict["vit.embeddings.position_embeddings"]
    state_dict["vit.embeddings.position_embeddings"] = new_emb

    yolos.load_state_dict(state_dict, strict=False)

    return yolos


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


class YOLOsDetector(LightningModule):
    def __init__(self,
                 detection_model,
                 **kwargs,
                 ):
        """
        Args:
        :param detection_model: The object detection model
        """
        super().__init__()

        self.detection_model = detection_model

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
