# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OWL-ViT model."""

import warnings
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Callable, List
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from transformers import PretrainedConfig
from models.modeling_vitxrs import ViTXRSModel, ViTXRSConfig
from models.modeling_proj_camembert import ProjCamembertModel, ProjCamembertConfig
from transformers.models.owlvit.configuration_owlvit import OwlViTConfig
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTOutput,
    OwlViTBoxPredictionHead,
    OwlViTClassPredictionHead,
    OwlViTForObjectDetection,
    OwlViTObjectDetectionOutput,
    OwlViTImageGuidedObjectDetectionOutput,
    owlvit_loss,
    box_iou,
    generalized_box_iou,
)
from transformers.models.detr.modeling_detr import (
    DetrHungarianMatcher,
    DetrLoss,
)

from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


@dataclass
class XRSOwlViTObjectDetectionOutput(ModelOutput):
    """
    Output type of :class:`~transformers.XRSOwlViTForImageGuidedObjectDetection`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the classification and the regression losses.
        loss_dict (`optional`, returned when ``labels`` is provided, ``Dict[str, torch.FloatTensor]``):
        image_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, embed_dim)`):
            Output of the vision model embedding layer.
        text_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, embed_dim)`):
            Output of the text model embedding layer.
        pred_boxes (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, 4)`):
            Predicted boxes coordinates (output of the regression head).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, num_classes)`):
            Classification logits (output of the classification head).
        class_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, embed_dim)`):
            Output of the classification head.
        text_model_output (:obj:`Tuple[torch.FloatTensor]`, `optional`, returned when ``return_text_model_output`` is passed, tuple of :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, embed_dim)`):
            Output of the text model.
        vision_model_output (:obj:`Tuple[torch.FloatTensor]`, `optional`, returned when ``return_vision_model_output`` is passed, tuple of :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_queries, embed_dim)`):
            Output of the vision model.
    """
    loss: Optional[Tensor] = None
    loss_dict: Optional[Dict[str, Tensor]] = None
    image_embeds: Optional[Tensor] = None
    text_embeds: Optional[Tensor] = None
    pred_boxes: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    class_embeds: Optional[Tensor] = None
    text_model_output: Optional[Tuple[Tensor]] = None
    vision_model_output: Optional[Tuple[Tensor]] = None


class XRSOwlViT(torch.nn.Module):
    def __init__(self,
                 vision_config: Optional[ViTXRSConfig] = None,
                 text_config: Optional[ProjCamembertConfig] = None,
                 vision_model: Optional[ViTXRSModel] = None,
                 text_model: Optional[ProjCamembertModel] = None,
                 ):
        super().__init__()

        # Check that a config or a model has been provided for each modality, and only one of them
        assert (vision_config is not None) or (vision_model is not None), "vision_config or vision_model should be defined"
        assert (text_config is not None) or (text_model is not None), "text_config or text_model should be defined"
        assert (vision_config is None) or (vision_model is None), "Only one of vision_config or vision_model should be defined"
        assert (text_config is None) or (text_model is None), "Only one of text_config or text_model should be defined"

        if vision_config is None:
            vision_config = vision_model.config

        if text_config is None:
            text_config = text_model.config

        self.vision_config = vision_config
        self.text_config = text_config

        if vision_model is None:
            self.vision_model = ViTXRSModel(config=vision_config)
        else:
            self.vision_model = vision_model

        if text_model is None:
            self.text_model = ProjCamembertModel(config=text_config)
        else:
            self.text_model = text_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "XRSOwlViT":
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        vision_model = ViTXRSModel.from_pretrained(pretrained_model_name_or_path / "vision_encoder", **kwargs)
        text_model = ProjCamembertModel.from_pretrained(pretrained_model_name_or_path / "text_encoder", **kwargs)

        return cls(vision_model=vision_model, text_model=text_model)

    def save_pretrained(self, save_directory: str):
        save_directory = Path(save_directory)
        self.vision_model.save_pretrained(save_directory / "vision_encoder")
        self.text_model.save_pretrained(save_directory / "text_encoder")

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Get embeddings for all text queries in all batch samples
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        pooled_output = text_output.last_hidden_state[:, 0]

        text_features = self.text_model.projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)
        image_features = vision_outputs.last_hidden_state[:, 0]
        image_features = self.vision_model.projection(image_features)

        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OwlViTOutput]:

        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)
        image_embeds = vision_outputs.last_hidden_state[:, 0]
        image_embeds = self.vision_model.projection(image_embeds)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        text_embeds = text_outputs.last_hidden_state[:, 0]
        text_embeds = self.text_model.projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits and set it on the correct device
        #logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t())
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class XRSOwlViTClassPredictionHead(OwlViTClassPredictionHead):
    def __init__(self, config: OwlViTConfig):
        super().__init__(config)

        out_dim = config.text_config.projection_size
        self.query_dim = config.vision_config.projection_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()


class XRSOwlViTBoxPredictionHead(OwlViTBoxPredictionHead):
    def __init__(self, config: OwlViTConfig, out_dim: int = 4):
        super().__init__(config)

        width = config.vision_config.projection_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)


class XRSOwlViTForObjectDetection(OwlViTForObjectDetection):
    def __init__(self,
                 owlvit_config: Optional[OwlViTConfig] = None,
                 text_model: Optional[ProjCamembertModel] = None,
                 vision_model: Optional[ViTXRSModel] = None,
                 ):

        assert text_model is not None or owlvit_config is not None, "You must specify either a text config or a text model"
        assert vision_model is not None or owlvit_config is not None, "You must specify either a vision config or a vision model"

        if owlvit_config is not None:
            assert owlvit_config.text_config.to_dict()["model_type"] != "camembert" or text_model is None, "You cannot specify both a text config and a text model"
            assert owlvit_config.vision_config.to_dict()["model_type"] != "vitxrs" or vision_model is None, "You cannot specify both a vision config and a vision model"

        if text_model is None:
            text_model = ProjCamembertModel(owlvit_config.text_config)

        text_config = text_model.config

        if vision_model is None:
            vision_model = ViTXRSModel(owlvit_config.vision_config)

        vision_config = vision_model.config

        if owlvit_config is None:
            owlvit_config = OwlViTConfig(projection_dim=vision_config.projection_size,
                                         vision_config=vision_config.to_dict(),
                                         text_config=text_config.to_dict())

        super().__init__(owlvit_config)

        self.owlvit = XRSOwlViT(text_model=text_model, vision_model=vision_model)
        self.class_head = XRSOwlViTClassPredictionHead(owlvit_config)
        self.box_head = XRSOwlViTBoxPredictionHead(owlvit_config)

        self.layer_norm = nn.LayerNorm(owlvit_config.vision_config.projection_size, eps=owlvit_config.vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()

        self.no_class_token = nn.Parameter(torch.zeros(owlvit_config.vision_config.projection_size))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):

        # Save the class head and box head
        super().save_pretrained(save_directory, **kwargs)

    def image_text_embedder(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # Encode text and image
        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Get image embeddings
        last_hidden_state = outputs.vision_model_output.last_hidden_state

        image_embeds = self.owlvit.vision_model.projection(last_hidden_state)
        #image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]

        return (text_embeds, image_embeds, outputs)

    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # Get OwlViTModel vision embeddings (same as CLIP)
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values, return_dict=True)

        # Apply post_layernorm to last_hidden_state, return non-projected output
        last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = self.owlvit.vision_model.projection(last_hidden_state)
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    def image_guided_detection(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> OwlViTImageGuidedObjectDetectionOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Compute feature maps for the input and query images
        query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
        feature_map, vision_outputs = self.image_embedder(
                                                          pixel_values=pixel_values,
                                                          output_attentions=output_attentions,
                                                          output_hidden_states=output_hidden_states,
                                                         )

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        # Get top class embedding and best box index for each query image in batch
        query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats, query_feature_map)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)

        # Predict object boxes
        target_pred_boxes = self.box_predictor(image_feats, feature_map)

        if not return_dict:
            output = (
                feature_map,
                query_feature_map,
                target_pred_boxes,
                query_pred_boxes,
                pred_logits,
                class_embeds,
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return OwlViTImageGuidedObjectDetectionOutput(
            image_embeds=feature_map,
            query_image_embeds=query_feature_map,
            target_pred_boxes=target_pred_boxes,
            query_pred_boxes=query_pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=None,
            vision_model_output=vision_outputs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[List[dict]] = None,
        return_dict: Optional[bool] = None,
    ) -> XRSOwlViTObjectDetectionOutput:
        """
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Embed images and text queries
        query_embeds, feature_map, outputs = self.image_text_embedder(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Text and vision model outputs
        text_outputs = outputs.text_model_output
        vision_outputs = outputs.vision_model_output

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // feature_map.shape[0]
        query_embeds = query_embeds.reshape(batch_size, max_text_queries, query_embeds.shape[-1])

        # concat no_class_token to query_embeds
        no_class_token = self.no_class_token.unsqueeze(0).repeat(query_embeds.shape[0], 1, 1)
        query_embeds = torch.cat((no_class_token, query_embeds), dim=1)

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        #input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = torch.ones(query_embeds.shape[:2], dtype=torch.bool, device=query_embeds.device)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=1,
                bbox_cost=5,
                giou_cost=2,
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=max_text_queries,
                eos_coef=0.1,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the loss
            outputs_loss = {}
            outputs_loss["logits"] = pred_logits
            outputs_loss["pred_boxes"] = pred_boxes

            loss_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1,
                           "loss_bbox": 5,
                           "loss_giou": 2,
                           }

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            output = (
                loss,
                loss_dict,
                pred_logits,
                pred_boxes,
                query_embeds,
                feature_map,
                class_embeds,
                text_outputs.to_tuple(),
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return XRSOwlViTObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
