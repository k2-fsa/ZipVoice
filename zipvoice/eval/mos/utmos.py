#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
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
Calculate UTMOS score with automatic Mean Opinion Score (MOS) prediction system
"""
import argparse
import logging
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from zipvoice.eval.models.utmos import UTMOS22Strong
from zipvoice.eval.utils import load_waveform


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate UTMOS score using UTMOS22Strong model."
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Path to the directory containing evaluated speech files.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of our evaluatioin model repository."
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models."
        "Will use 'tts_eval_models/mos/utmos22_strong_step7459_v1.pt'"
        " in this script",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    return parser


class UTMOSScore:
    """Predicting UTMOS score for each audio clip."""

    def __init__(self, model_path: str):
        """
        Initializes the UTMOS score evaluator with the specified model.

        Args:
            model_path (str): Path of the UTMOS model checkpoint.
        """
        self.sample_rate = 16000
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")

        # Initialize and load the model
        self.model = UTMOS22Strong()
        try:
            state_dict = torch.load(
                model_path, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_files(self, wav_paths: List[str]) -> List[float]:
        """
        Computes UTMOS scores for a list of audio files.

        Args:
            wav_paths (List[str]): List of paths to audio files.

        Returns:
            List[float]: List of UTMOS scores.
        """
        scores = []
        for wav_path in tqdm(wav_paths, desc="Scoring audio files"):
            # Load and preprocess waveform
            speech = load_waveform(wav_path, self.sample_rate, device=self.device)
            # Compute score
            score = self.model(speech.unsqueeze(0), self.sample_rate)
            scores.append(score.item())

        return scores

    def score_dir(self, dir_path: str, extension: str) -> float:
        """
        Computes the average UTMOS score for all files in a directory.

        Args:
            dir_path (str): Path to the directory containing audio files.

        Returns:
            float: Average UTMOS score for the directory.
        """
        logging.info(f"Calculating UTMOS score for {dir_path}")

        # Get list of wav files
        wav_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(extension)
        ]

        if not wav_files:
            raise ValueError(f"No audio files found in {dir_path}")

        # Compute scores
        scores = self.score_files(wav_files)

        return float(np.mean(scores))


if __name__ == "__main__":

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()

    # Validate input path
    if not os.path.isdir(args.wav_path):
        logging.error(f"Invalid directory: {args.wav_path}")
        exit(1)

    # Initialize evaluator
    model_path = os.path.join(args.model_dir, "mos/utmos22_strong_step7459_v1.pt")
    if not os.path.exists(model_path):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        exit(1)
    utmos_evaluator = UTMOSScore(model_path)

    # Compute UTMOS score
    score = utmos_evaluator.score_dir(args.wav_path, args.extension)
    print("-" * 50)
    logging.info(f"UTMOS score: {score:.2f}")
    print("-" * 50)
