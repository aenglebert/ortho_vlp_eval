from typing import Dict, List, Optional, Set, Tuple, Union

import math

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import ViTConfig, ViTModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput, ModelOutput
from transformers.configuration_utils import PretrainedConfig

from transformers.models.vit.modeling_vit import ViTAttention, ViTIntermediate, ViTOutput, ViTEmbeddings, ViTPooler
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings, ViTMAEDecoder, ViTMAEEncoder

AUTO_MAP = {
    "AutoModel": "modeling_vitxrs.ViTXRSModel",
}


class ViTXRSConfig(ViTConfig):

    model_type = "vitxrs"

    def __init__(self,
                 seq_model=False,
                 mask_ratio=0,
                 projection_size=512,
                 logit_scale_init_value=2.6592,
                 **kwargs):
        super().__init__(**kwargs)

        self.seq_model = seq_model
        self.mask_ratio = mask_ratio
        self.projection_size = projection_size
        self.logit_scale = logit_scale_init_value
        self.logit_bias = -10.0


class ViTXRSMAEConfig(ViTXRSConfig):

    model_type = "vitxrs_mae"

    def __init__(self,
                 encoder_config: PretrainedConfig = None,
                 decoder_config: PretrainedConfig = None,
                 decoder_num_attention_heads=16,
                 decoder_hidden_size=512,
                 decoder_num_hidden_layers=8,
                 decoder_intermediate_size=2048,
                 norm_pix_loss=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pix_loss = norm_pix_loss

    @classmethod
    def from_encoder_decoder_configs(cls,
                                     encoder_config: PretrainedConfig,
                                     decoder_config: PretrainedConfig,
                                     ):
        return cls(encoder_config=encoder_config, decoder_config=decoder_config)


@dataclass
class ViTXRSModelOutput(ModelOutput):
    """
    Class for ViTXRSModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, token_sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        grouped_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, images_sequence_lenght, token_sequence_length, hidden_size)`):
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ViTXRSMAEForPreTrainingOutput(ModelOutput):

    loss: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GatedAttentionPool1d(nn.Module):
    """
    Based on paper from Ilse, M., Tomczak, J., & Welling, M. (2018, July).
    Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    http://proceedings.mlr.press/v80/ilse18a.html
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # V and U are linear projections of the input embeddings
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, pooling_matrix):
        # x is Bimg, C

        # We compute the projections for the images embeddings
        Vx = self.V(x)
        Ux = self.U(x)

        z = torch.tanh(Vx) * torch.sigmoid(Ux)
        s = self.w(z).squeeze(-1)

        # s is still Bimg, we repeat to get (Btxt, Bimg)
        s = s.expand(*pooling_matrix.shape)

        # We mask using pooling_matrix
        mask = ((1 - pooling_matrix) * torch.finfo(pooling_matrix.dtype).min)
        s = s + mask

        # Attention weights using a softmax
        a = torch.softmax(s, dim=-1)

        # Weighted sum
        return a @ x


class ViTXRSEmbeddings(ViTEmbeddings):
    """
    Combination of ViTEmbeddings and ViTMAEEmbeddings
    """
    def __init__(self, config):
        super().__init__(config)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        From ViTMAEEmbeddings.random_masking

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
        noise=None,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        cls_tokens = embeddings[:, 0, :]
        embeddings = embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_tokens = cls_tokens.unsqueeze(1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings, mask, ids_restore


class ViTXRSModel(ViTModel):
    config_class = ViTXRSConfig

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTXRSEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pooler = GatedAttentionPool1d(config) if config.seq_model else None

        self.projection = nn.Linear(config.hidden_size, config.projection_size)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(self.config.logit_bias))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pooling_matrix: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ViTXRSModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, interpolate_pos_encoding=True)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if self.pooler is not None and pooling_matrix is not None:
            pooled_output = self.pooler(sequence_output[:, 0], pooling_matrix)
        else:
            pooled_output = None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return ViTXRSModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTXRSMAEForPreTraining(ViTXRSModel):

    config_class = ViTXRSMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: ViTXRSMAEConfig):
        super().__init__(config)

        self.decoder = ViTMAEDecoder(config, num_patches=self.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """

        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            pooling_matrix: Optional[torch.FloatTensor] = None,
            noise: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTXRSMAEForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTXRSMAEForPreTrainingOutput(
            loss=loss,
            pooler_output=outputs.pooler_output,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
