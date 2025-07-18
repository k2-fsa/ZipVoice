#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu
#                                                   Wei Kang)
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
Computes speaker similarity (SIM-o) using a WavLM-based
    ECAPA-TDNN speaker verification model.
"""
import argparse
import logging
import os
import warnings
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
from zipvoice.eval.utils import load_waveform

warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate speaker similarity (SIM-o) score."
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Path to the directory containing evaluated speech files.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        required=True,
        help="Path to the file list that contains the correspondence between prompts "
        "and evaluated speech. Each line contains (audio_name, prompt_text_1, "
        "prompt_text_2, prompt_audio_1, prompt_audio_2, text) separated by tabs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of our evaluatioin model repository."
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models."
        "Will use 'tts_eval_models/speaker_similarity/wavlm_large_finetune.pth'"
        "and 'tts_eval_models/speaker_similarity/wavlm_large/' in this script",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    return parser


class SpeakerSimilarity:
    """
    Computes speaker similarity (SIM-o) using a WavLM-based
        ECAPA-TDNN speaker verification model.
    """

    def __init__(
        self,
        sv_model_path: str = "speaker_similarity/wavlm_large_finetune.pth",
        ssl_model_path: str = "speaker_similarity/wavlm_large/",
    ):
        """
        Initializes the speaker similarity evaluator with the specified models.

        Args:
            sv_model_path (str): Path of the wavlm-based ECAPA-TDNN model checkpoint.
            ssl_model_path (str): Path of the wavlm SSL model directory.
        """
        self.sample_rate = 16000
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")
        self.model = ECAPA_TDNN_WAVLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=self.sample_rate,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(
            sv_model_path, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embeddings(self, wav_paths: List[str]) -> List[torch.Tensor]:
        """
        Extracts speaker embeddings from a list of audio files.

        Args:
            wav_paths (List[str]): List of paths to audio files.

        Returns:
            List[torch.Tensor]: List of speaker embeddings.
        """
        embeddings = []
        for wav_path in tqdm(wav_paths, desc="Extracting speaker embeddings"):
            # Load and preprocess waveform
            speech = load_waveform(
                wav_path, self.sample_rate, device=self.device, max_seconds=120
            )
            # Extract embedding
            embedding = self.model([speech])
            embeddings.append(embedding)

        return embeddings

    def score(self, wav_path: str, extension: str, test_list: str) -> float:
        """
        Computes the Speaker Similarity (SIM-o) score between reference and
            evaluated speech.

        Args:
            wav_path (str): Path to the directory containing evaluated speech files.
            test_list (str): Path to the test list file mapping evaluated files
                to reference prompts.

        Returns:
            float: Average similarity score between reference and evaluated embeddings.
        """
        logging.info(f"Calculating Speaker Similarity (SIM-o) score for {wav_path}")
        # Read test pairs
        try:
            with open(test_list, "r", encoding="utf-8") as f:
                lines = [line.strip().split("\t") for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to read test list: {e}")
            raise

        if not lines:
            raise ValueError(f"Test list {test_list} is empty or malformed")
        # Parse test pairs
        prompt_wavs = []
        eval_wavs = []
        for line in lines:
            if len(line) != 4:
                raise ValueError(f"Invalid line: {line}")
            wav_name, prompt_text, prompt_wav, text = line
            eval_wav_path = os.path.join(wav_path, f"{wav_name}.{extension}")
            # Validate file existence
            if not os.path.exists(prompt_wav):
                raise FileNotFoundError(f"Prompt file not found: {prompt_wav}")
            if not os.path.exists(eval_wav_path):
                raise FileNotFoundError(f"Evaluated file not found: {eval_wav_path}")
            prompt_wavs.append(prompt_wav)
            eval_wavs.append(eval_wav_path)
        logging.info(f"Found {len(prompt_wavs)} valid test pairs")
        # Extract embeddings

        prompt_embeddings = self.get_embeddings(prompt_wavs)
        eval_embeddings = self.get_embeddings(eval_wavs)

        if len(prompt_embeddings) != len(eval_embeddings):
            raise RuntimeError(
                f"Mismatch: {len(prompt_embeddings)} prompt vs "
                f" {len(eval_embeddings)} eval embeddings"
            )

        # Calculate similarity scores
        scores = []
        for prompt_emb, eval_emb in zip(prompt_embeddings, eval_embeddings):
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                prompt_emb, eval_emb, dim=-1
            )
            scores.append(similarity.item())

        return float(np.mean(scores))


if __name__ == "__main__":

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()
    # Initialize evaluator
    sv_model_path = os.path.join(
        args.model_dir, "speaker_similarity/wavlm_large_finetune.pth"
    )
    ssl_model_path = os.path.join(args.model_dir, "speaker_similarity/wavlm_large/")
    if not os.path.exists(sv_model_path) or not os.path.exists(ssl_model_path):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        exit(1)
    sim_evaluator = SpeakerSimilarity(
        sv_model_path=sv_model_path, ssl_model_path=ssl_model_path
    )
    # Compute similarity score
    score = sim_evaluator.score(args.wav_path, args.extension, args.test_list)
    print("-" * 50)
    logging.info(f"SIM-o score: {score:.3f}")
    print("-" * 50)
