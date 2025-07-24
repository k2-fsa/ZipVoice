#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation
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
Usage:
This script loads checkpoints and averages them.

python3 -m zipvoice.bin.generate_averaged_model  \
    --epoch 11 \
    --avg 4 \
    --model-name zipvoice \
    --exp-dir exp/zipvoice

It will generate a file `epoch-11-avg-14.pt` in the given `exp_dir`.
You can later load it by `torch.load("epoch-11-avg-4.pt")`.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_dialog import ZipVoiceDialog, ZipVoiceDialogStereo
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import (
    average_checkpoints_with_averaged_model,
    find_checkpoints,
)
from zipvoice.utils.common import AttributeDict


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=11,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=4,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' or --iter",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice",
        help="The experiment dir",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=[
            "zipvoice",
            "zipvoice_distill",
            "zipvoice_dialog",
            "zipvoice_dialog_stereo",
        ],
        help="The model type to be averaged. ",
    )

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.exp_dir = Path(params.exp_dir)

    with open(params.exp_dir / "model.json", "r") as f:
        model_config = json.load(f)

    # Any tokenizer can be used here.
    # Use SimpleTokenizer for simplicity.
    tokenizer = SimpleTokenizer(token_file=params.exp_dir / "tokens.txt")
    if params.model_name in ["zipvoice", "zipvoice_distill"]:
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        }
    elif params.model_name in ["zipvoice_dialog", "zipvoice_dialog_stereo"]:
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "spk_a_id": tokenizer.spk_a_id,
            "spk_b_id": tokenizer.spk_b_id,
        }

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    logging.info("Script started")

    params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    logging.info("About to create model")
    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_distill":
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_dialog":
        model = ZipVoiceDialog(
            **model_config["model"],
            **tokenizer_config,
        )
    elif params.model_name == "zipvoice_dialog_stereo":
        model = ZipVoiceDialogStereo(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        raise ValueError(f"Unknown model name: {params.model_name}")

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for" f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        logging.info(
            "Calculating the averaged model over iteration checkpoints"
            f" from {filename_start} (excluded) to {filename_end}"
        )
        model.to(params.device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=params.device,
            ),
            strict=True,
        )
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {params.epoch}"
        )
        model.to(params.device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=params.device,
            ),
            strict=True,
        )
    if params.iter > 0:
        filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
    else:
        filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"

    logging.info(f"Saving the averaged checkpoint to {filename}")
    torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
