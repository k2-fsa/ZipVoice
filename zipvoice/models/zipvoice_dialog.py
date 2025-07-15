# Copyright    2025    Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from zipvoice.models.modules.zipformer_two_stream import TTSZipformerTwoStream
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.utils.common import condition_time_mask_suffix, make_pad_mask, pad_labels


class ZipVoiceDialog(ZipVoice):
    """The ZipVoice-Dialog model."""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
        spk_a_id: int = 360,
        spk_b_id: int = 361,
    ):
        """
        Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
            spk_a_id: ID of speaker A / [S1].
            spk_b_id: ID of speaker B / [S2].
        """
        super().__init__(
            fm_decoder_downsampling_factor=fm_decoder_downsampling_factor,
            fm_decoder_num_layers=fm_decoder_num_layers,
            fm_decoder_cnn_module_kernel=fm_decoder_cnn_module_kernel,
            fm_decoder_feedforward_dim=fm_decoder_feedforward_dim,
            fm_decoder_num_heads=fm_decoder_num_heads,
            fm_decoder_dim=fm_decoder_dim,
            text_encoder_num_layers=text_encoder_num_layers,
            text_encoder_feedforward_dim=text_encoder_feedforward_dim,
            text_encoder_cnn_module_kernel=text_encoder_cnn_module_kernel,
            text_encoder_num_heads=text_encoder_num_heads,
            text_encoder_dim=text_encoder_dim,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            query_head_dim=query_head_dim,
            value_head_dim=value_head_dim,
            pos_head_dim=pos_head_dim,
            pos_dim=pos_dim,
            feat_dim=feat_dim,
            vocab_size=vocab_size,
            pad_id=pad_id,
        )

        self.spk_a_id = spk_a_id
        self.spk_b_id = spk_b_id
        self.spk_embed = nn.Embedding(2, feat_dim)
        torch.nn.init.normal_(self.spk_embed.weight, mean=0, std=0.1)

    def extract_spk_indices(self, tensor):
        turn_mask = ((tensor == self.spk_a_id) | (tensor == self.spk_b_id)).long()
        turn_counts = turn_mask.cumsum(dim=1)
        spk_mask = turn_counts % 2
        spk_mask = torch.where(tensor == self.pad_id, -1, spk_mask)
        spk_a_indices = torch.where(spk_mask == 0)
        spk_b_indices = torch.where(spk_mask == 1)
        return spk_a_indices, spk_b_indices

    def forward_text_embed(
        self,
        tokens: List[List[int]],
    ):
        """
        Get the text embeddings.
        Args:
            tokens: a list of list of token ids.
        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        embed = self.embed(tokens_padded)  # (B, S, C)
        spk_a_indices, spk_b_indices = self.extract_spk_indices(tokens_padded)
        tokens_lens = torch.tensor(
            [len(token) for token in tokens], dtype=torch.int64, device=device
        )
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(
            x=embed, t=None, padding_mask=tokens_padding_mask
        )  # (B, S, C)
        embed[spk_a_indices] += self.spk_embed(torch.tensor(0, device=device)).to(
            embed.dtype
        )
        embed[spk_b_indices] += self.spk_embed(torch.tensor(1, device=device)).to(
            embed.dtype
        )
        return embed, tokens_lens

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass of the model for training.
        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.
        Returns:
            fm_loss: the flow-matching loss.
        """

        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition_mask = condition_time_mask_suffix(
            features_lens=features_lens,
            mask_percent=(0.5, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss


class ZipVoiceDialogStereo(ZipVoiceDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        required_params = {
            "feat_dim",
            "fm_decoder_downsampling_factor",
            "fm_decoder_num_layers",
            "fm_decoder_cnn_module_kernel",
            "fm_decoder_dim",
            "fm_decoder_feedforward_dim",
            "fm_decoder_num_heads",
            "query_head_dim",
            "pos_head_dim",
            "value_head_dim",
            "pos_dim",
            "time_embed_dim",
        }

        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        self.fm_decoder = TTSZipformerTwoStream(
            in_dim=(kwargs["feat_dim"] * 5, kwargs["feat_dim"] * 3),
            out_dim=(kwargs["feat_dim"] * 2, kwargs["feat_dim"]),
            downsampling_factor=kwargs["fm_decoder_downsampling_factor"],
            num_encoder_layers=kwargs["fm_decoder_num_layers"],
            cnn_module_kernel=kwargs["fm_decoder_cnn_module_kernel"],
            encoder_dim=kwargs["fm_decoder_dim"],
            feedforward_dim=kwargs["fm_decoder_feedforward_dim"],
            num_heads=kwargs["fm_decoder_num_heads"],
            query_head_dim=kwargs["query_head_dim"],
            pos_head_dim=kwargs["pos_head_dim"],
            value_head_dim=kwargs["value_head_dim"],
            pos_dim=kwargs["pos_dim"],
            use_time_embed=True,
            time_embed_dim=kwargs["time_embed_dim"],
        )

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
        se_weight: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass of the model for training.
        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.
            se_weight: the weight of the speaker exclusive loss.
        Returns:
            fm_loss: the flow-matching loss.
        """

        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition_mask = condition_time_mask_suffix(
            features_lens=features_lens,
            mask_percent=(0.5, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        if se_weight > 0:
            target = xt + vt * (1 - t)
            fbank_1 = target[:, :, : self.feat_dim]
            fbank_2 = target[:, :, self.feat_dim :]
            energy_loss = torch.mean(
                self.energy_based_loss(fbank_1, fbank_2, features)[loss_mask]
            )
            loss = fm_loss + energy_loss * se_weight
        else:
            loss = fm_loss

        return loss

    def energy_based_loss(self, fbank1, fbank2, gt_fbank):
        energy1 = self.energy(fbank1)
        energy2 = self.energy(fbank2)

        energy_thresholds = self.adaptive_threshold_from_gt(
            torch.cat(
                [
                    gt_fbank[:, :, : self.feat_dim],
                    gt_fbank[:, :, self.feat_dim :],
                ],
                dim=1,
            )
        )

        both_speaking = (
            (energy1 > energy_thresholds) & (energy2 > energy_thresholds)
        ).float()

        penalty = (
            both_speaking
            * (energy1 - energy_thresholds)
            * (energy2 - energy_thresholds)
        )
        return penalty

    def energy(self, fbank):
        return torch.mean(fbank, dim=-1)

    def adaptive_threshold_from_gt(self, gt_fbank, percentile=50):
        frame_energies = self.energy(gt_fbank)
        thresholds = torch.quantile(frame_energies, q=percentile / 100, dim=1)
        return thresholds.unsqueeze(1)
