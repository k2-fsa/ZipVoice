#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
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

"""
This script exports a pre-trained ZipVoice or ZipVoice-Distill model from PyTorch to
ONNX. Required models will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:

export HF_ENDPOINT=https://hf-mirror.com

python3 zipvoice/onnx_export.py \
  --onnx-model-dir onnx_zipvoice \
  --model-name zipvoice

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.
"""


import argparse
import json
import os

from typing import Dict

import onnx
import safetensors.torch
import torch

from huggingface_hub import hf_hub_download
from model import get_distill_model, get_model
from onnxruntime.quantization import QuantType, quantize_dynamic
from scaling_converter import convert_scaled_to_non_scaled
from tokenizer import EmiliaTokenizer
from torch import Tensor, nn
from utils import AttributeDict


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--onnx-model-dir",
        type=str,
        default="exp",
        help="Dir to the exported models",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxTextModel(nn.Module):
    def __init__(self, model: nn.Module):
        """A wrapper for ZipVoice text encoder."""
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id

    def forward(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tensor:
        cat_tokens = torch.cat([prompt_tokens, tokens], dim=1)
        cat_tokens = nn.functional.pad(cat_tokens, (0, 1), value=self.pad_id)
        tokens_len = cat_tokens.shape[1] - 1
        padding_mask = (torch.arange(tokens_len + 1) == tokens_len).unsqueeze(0)

        embed = self.embed(cat_tokens)
        embed = self.text_encoder(x=embed, t=None, padding_mask=padding_mask)

        features_len = torch.ceil(
            (prompt_features_len / prompt_tokens.shape[1] * tokens_len / speed)
        ).to(dtype=torch.int64)

        token_dur = torch.div(
            features_len, tokens_len, rounding_mode="floor"
        ).to(dtype=torch.int64)

        text_condition = embed[:, :-1, :].unsqueeze(2).expand(-1, -1, token_dur, -1)
        text_condition = text_condition.reshape(embed.shape[0], -1, embed.shape[2])

        text_condition = torch.cat(
            [
                text_condition,
                embed[:, -1:, :].expand(-1, features_len - text_condition.shape[1], -1)
            ],
            dim=1
        )

        return text_condition


class OnnxFlowMatchingModel(nn.Module):
    def __init__(self, model: nn.Module):
        """A wrapper for ZipVoice flow-matching decoder."""
        super().__init__()
        self.distill = model.distill
        self.fm_decoder = model.fm_decoder
        self.model_func = getattr(model, "forward_fm_decoder")
        self.feat_dim = model.feat_dim

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                guidance_scale=guidance_scale,
            )
        else:
            x = x.repeat(2, 1, 1)
            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )
            m = (t <= 0.5).float()
            speech_condition = torch.cat(
                [speech_condition, speech_condition], dim=0
            )
            guidance_scale = guidance_scale * (m + 1)
            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
            ).chunk(2, dim=0)
            v = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return v


def export_text_model(
    model: OnnxTextModel,
    filename: str,
    opset_version: int = 11,
) -> None:
    """Export the text encoder model to ONNX format.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
    prompt_tokens = torch.tensor([[0, 1]], dtype=torch.int64)
    prompt_features_len = torch.tensor(10, dtype=torch.int64)
    speed = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(
        model,
        (tokens, prompt_tokens, prompt_features_len, speed)
    )

    torch.onnx.export(
        model,
        (tokens, prompt_tokens, prompt_features_len, speed),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["tokens", "prompt_tokens", "prompt_features_len", "speed"],
        output_names=["text_condition"],
        dynamic_axes={
            "tokens": {0: "N", 1: "T"},
            "prompt_tokens": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice text encoder",
    }
    print(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)

    print(f"Exported to {filename}")


def export_flow_matching_model(
    model: OnnxFlowMatchingModel,
    filename: str,
    opset_version: int = 11,
) -> None:
    """Export the flow matching model to ONNX format.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    feat_dim = model.feat_dim
    seq_len = 200
    t = torch.tensor(0.5, dtype=torch.float32)
    x = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    text_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    speech_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    guidance_scale = torch.tensor(1.0, dtype=torch.float32)

    model = torch.jit.trace(
        model,
        (t, x, text_condition, speech_condition, guidance_scale)
    )

    torch.onnx.export(
        model,
        (t, x, text_condition, speech_condition, guidance_scale),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["t", "x", "text_condition", "speech_condition", "guidance_scale"],
        output_names=["v"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "text_condition": {0: "N", 1: "T"},
            "speech_condition": {0: "N", 1: "T"},
            "v": {0: "N", 1: "T"},
        },
    )

    meta_data = {
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ZipVoice flow-matching decoder",
        "feat_dim": str(feat_dim),
    }
    print(f"meta_data: {meta_data}")
    add_meta_data(filename=filename, meta_data=meta_data)

    print(f"Exported to {filename}")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))

    model_config = hf_hub_download("zhu-han/ZipVoice", filename="model.json")
    with open(model_config, "r") as f:
        model_config = json.load(f)
        for key, value in model_config["model"].items():
            setattr(params, key, value)
        for key, value in model_config["feature"].items():
            setattr(params, key, value)

    token_file = hf_hub_download("zhu-han/ZipVoice", filename="tokens_emilia.txt")
    tokenizer = EmiliaTokenizer(token_file)

    params.vocab_size = tokenizer.vocab_size
    params.pad_id = tokenizer.pad_id

    if params.model_name == "zipvoice_distill":
        model = get_distill_model(params)
        model_ckpt = hf_hub_download(
            "zhu-han/ZipVoice",
            filename="exp_zipvoice_distill/model.safetensors",
        )
    else:
        model = get_model(params)
        model_ckpt = hf_hub_download(
            "zhu-han/ZipVoice", filename="exp_zipvoice/model.safetensors"
        )

    safetensors.torch.load_model(model, model_ckpt)

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    print("Exporting model")
    os.makedirs(params.onnx_model_dir, exist_ok=True)
    opset_version = 11

    text_model = OnnxTextModel(model=model)
    text_model_file = f"{params.onnx_model_dir}/text_model.onnx"
    export_text_model(
        model=text_model,
        filename=text_model_file,
        opset_version=opset_version,
    )

    flow_matching_model = OnnxFlowMatchingModel(model=model)
    flow_matching_model_file = f"{params.onnx_model_dir}/flow_matching_model.onnx"
    export_flow_matching_model(
        model=flow_matching_model,
        filename=flow_matching_model_file,
        opset_version=opset_version,
    )

    print("Generate int8 quantization models")

    text_model_int8_file = f"{params.onnx_model_dir}/text_model_int8.onnx"
    quantize_dynamic(
        model_input=text_model_file,
        model_output=text_model_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    flow_matching_int8_file = f"{params.onnx_model_dir}/flow_matching_model_int8.onnx"
    quantize_dynamic(
        model_input=flow_matching_model_file,
        model_output=flow_matching_int8_file,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    print("Done!")


if __name__ == "__main__":
    main()
