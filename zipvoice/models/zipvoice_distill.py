# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
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

from zipvoice.models.modules.solver import DistillEulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.models.zipvoice import ZipVoice


class ZipVoiceDistill(ZipVoice):
    """ZipVoice-Distill model."""

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

        self.fm_decoder = TTSZipformer(
            in_dim=kwargs["feat_dim"] * 3,
            out_dim=kwargs["feat_dim"],
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
            use_guidance_scale_embed=True,
        )
        self.solver = DistillEulerSolver(self, func_name="forward_fm_decoder")

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        speech_condition_mask: torch.Tensor,
        t_start: float,
        t_end: float,
        num_step: int = 1,
        guidance_scale: torch.Tensor = None,
    ) -> torch.Tensor:

        return self.sample_intermediate(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            speech_condition_mask=speech_condition_mask,
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
