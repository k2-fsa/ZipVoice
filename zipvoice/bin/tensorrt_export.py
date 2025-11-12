#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
# Copyright         2025  Nvidia Corp.        (authors: Yuekai Zhang)
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
ONNX.

Usage:

python3 -m zipvoice.bin.tensorrt_export \
    --model-name zipvoice_distill \
    --model-dir models/zipvoice_distill \
    --checkpoint-name model.pt \
    --trt-engine-file-name fm_decoder.fp16.max_batch_4.plan \
    --tensorrt-model-dir models/zipvoice_distill_trt || exit 1

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.
"""


import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import math

import safetensors.torch
import torch
from torch import Tensor, nn

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled
from zipvoice.models.modules.zipformer import CompactRelPositionalEncoding

# Monkey-patching CompactRelPositionalEncoding.extend_pe
def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
    """Reset the positional encodings."""
    T = x.size(0) + left_context_len

    # if self.pe is not None:
    #     # self.pe contains both positive and negative parts
    #     # the length of self.pe is 2 * input_len - 1
    #     if self.pe.size(0) >= T * 2 - 1:
    #         self.pe = self.pe.to(dtype=x.dtype, device=x.device)
    #         return

    # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
    x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

    freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

    # `compression_length` this is arbitrary/heuristic, if it is larger we have more
    # resolution for small time offsets but less resolution for large time offsets.
    compression_length = self.embed_dim**0.5
    # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity
    # to infinity; but it does so more slowly than T for large absolute values of T.
    # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which is
    # important.
    x_compressed = (
        compression_length
        * x.sign()
        * ((x.abs() + compression_length).log() - math.log(compression_length))
    )

    # if self.length_factor == 1.0, then length_scale is chosen so that the
    # FFT can exactly separate points close to the origin (T == 0).  So this
    # part of the formulation is not really heuristic.
    # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
    length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

    # note for machine implementations: if atan is not available, we can use:
    #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
    #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 ,
    #  atan(x))
    x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

    cosines = (x_atan * freqs).cos()
    sines = (x_atan * freqs).sin()

    pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
    pe[:, 0::2] = cosines
    pe[:, 1::2] = sines
    pe[:, -1] = 1.0  # for bias.

    self.pe = pe.to(dtype=x.dtype)


CompactRelPositionalEncoding.extend_pe = extend_pe


def get_trt_kwargs_dynamic_batch(
    min_batch_size: int = 1,
    opt_batch_size: int = 2,
    max_batch_size: int = 4,
) -> Dict:
    """Get keyword arguments for TensorRT with dynamic batch size."""
    feat_dim = 300
    min_seq_len = 100
    opt_seq_len = 200
    max_seq_len = 3000
    min_shape = [(min_batch_size, min_seq_len, feat_dim), (min_batch_size,), (min_batch_size, min_seq_len), (min_batch_size,)]
    opt_shape = [(opt_batch_size, opt_seq_len, feat_dim), (opt_batch_size,), (opt_batch_size, opt_seq_len), (opt_batch_size,)]
    max_shape = [(max_batch_size, max_seq_len, feat_dim), (max_batch_size,), (max_batch_size, max_seq_len), (max_batch_size,)]
    input_names = ["x", "t", "padding_mask", "guidance_scale"]
    return {
        "min_shape": min_shape,
        "opt_shape": opt_shape,
        "max_shape": max_shape,
        "input_names": input_names,
    }


def convert_onnx_to_trt(
    trt_model: str, trt_kwargs: Dict, onnx_model: str, dtype: torch.dtype = torch.float16
):
    """
    Convert an ONNX model to a TensorRT engine.

    Args:
        trt_model (str): The path to save the TensorRT engine.
        trt_kwargs (Dict): Keyword arguments for TensorRT.
        onnx_model (str): The path to the ONNX model.
        dtype (torch.dtype, optional): The data type to use. Defaults to torch.float16.
    """
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    if dtype == torch.float16:
        tensor_dtype = trt.DataType.HALF
    elif dtype == torch.bfloat16:
        tensor_dtype = trt.DataType.BF16
    elif dtype == torch.float32:
        tensor_dtype = trt.DataType.FLOAT
    else:
        raise ValueError('invalid dtype {}'.format(dtype))
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tensorrt-model-dir",
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

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model.pt",
        help="The name of model checkpoint.",
    )

    parser.add_argument(
        "--trt-engine-file-name",
        type=str,
        default=None,
        help="The name of TensorRT engine file.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="The maximum batch size to use for TensorRT.",
    )

    return parser

def export_onnx_fm_decoder(
    model: torch.nn.Module,
    filename: str,
    opset_version: int = 18,
    distill: bool = False,
) -> None:
    """Export the flow matching decoder model to ONNX format.

    Args:
      model:
        The input model
      filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """


    feat_dim, seq_len = model.feat_dim, 200

    t = torch.tensor(0.5, dtype=torch.float32).unsqueeze(0)
    guidance_scale = torch.tensor(1.0, dtype=torch.float32).unsqueeze(0)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    x = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    text_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    speech_condition = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)
    xt= torch.cat([x, text_condition, speech_condition], dim=2)
    xt = xt.repeat(2, 1, 1)
    t = t.repeat(2)
    padding_mask = padding_mask.repeat(2, 1)
    guidance_scale = guidance_scale.repeat(2)

    inputs_tensors = [xt, t, padding_mask]
    input_names = ['x', 't', 'padding_mask']
    dynamic_axes = {
        'x': {0: 'N', 1: 'T'},
        't': {0: 'N'},
        'padding_mask': {0: 'N', 1: 'T'},
    }
    if distill:
        inputs_tensors.append(guidance_scale)
        input_names.append('guidance_scale')
        dynamic_axes['guidance_scale'] = {0: 'N'}
    estimator = model.fm_decoder
    estimator = torch.jit.trace(estimator, inputs_tensors)
    torch.onnx.export(
        estimator,
        inputs_tensors,
        filename,
        opset_version=opset_version,
        input_names=input_names,
        output_names=['v'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    logging.info(f"Exported to {filename}")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))

    params.model_dir = Path(params.model_dir)
    if not params.model_dir.is_dir():
        raise FileNotFoundError(f"{params.model_dir} does not exist")
    for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
        if not (params.model_dir / filename).is_file():
            raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
    model_ckpt = params.model_dir / params.checkpoint_name
    model_config = params.model_dir / "model.json"
    token_file = params.model_dir / "tokens.txt"

    logging.info(f"Loading model from {params.model_dir}")

    tokenizer = SimpleTokenizer(token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = False
    else:
        assert params.model_name == "zipvoice_distill"
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )
        distill = True

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    logging.info("Exporting model")
    tensorrt_model_dir = Path(params.tensorrt_model_dir)
    tensorrt_model_dir.mkdir(parents=True, exist_ok=True)
    opset_version = 18


    fm_decoder_onnx_file = tensorrt_model_dir / "fm_decoder.onnx"

    export_onnx_fm_decoder(
        model=model,
        filename=fm_decoder_onnx_file,
        opset_version=opset_version,
        distill=distill,
    )

    logging.info("Exported to TensorRT model")

    trt_engine_file = f'{str(tensorrt_model_dir)}/{params.trt_engine_file_name}'
    trt_kwargs = get_trt_kwargs_dynamic_batch(min_batch_size=1, opt_batch_size=2, max_batch_size=params.max_batch_size)
    convert_onnx_to_trt(trt_engine_file, trt_kwargs, fm_decoder_onnx_file, dtype=torch.float16)

    logging.info("Done!")


if __name__ == "__main__":

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    import tensorrt as trt
    main()
