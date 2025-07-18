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
Computes concatenated maximum permutation speaker similarity (cpSIM) scores using:
- A WavLM-based ECAPA-TDNN model for speaker embedding extraction.
- A pyannote pipeline for speaker diarization (segmenting speakers).
"""
import argparse
import logging
import os
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pyannote.audio import Pipeline
from tqdm import tqdm

from zipvoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
from zipvoice.eval.utils import load_waveform

warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate concatenated maximum permutation speaker "
        "similarity (cpSIM) score."
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
        help="Path to the tsv file for speaker splitted prompts. "
        "Each line contains (audio_name, prompt_text_1, prompt_text_2, "
        "prompt_audio_1, prompt_audio_2, text) separated by tabs.",
    )

    parser.add_argument(
        "--test-list-merge",
        type=str,
        help="Path to the tsv file for merged dialogue prompts. "
        "Each line contains (audio_name, prompt_text_dialogue, "
        "prompt_audio_dialogue, text) separated by tabs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of our evaluatioin model repository."
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models."
        "Will use 'tts_eval_models/speaker_similarity/wavlm_large_finetune.pth'"
        ", 'tts_eval_models/speaker_similarity/wavlm_large/' and "
        "tts_eval_models/speaker_similarity/pyannote/ in this script",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )
    return parser


class CpSpeakerSimilarity:
    """
    Computes concatenated maximum permutation speaker similarity (cpSIM) scores using:
    - A WavLM-based ECAPA-TDNN model for speaker embedding extraction.
    - A pyannote pipeline for speaker diarization (segmenting speakers).
    """

    def __init__(
        self,
        sv_model_path: str = "speaker_similarity/wavlm_large_finetune.pth",
        ssl_model_path: str = "speaker_similarity/wavlm_large/",
        pyannote_model_path: str = "speaker_similarity/pyannote/",
    ):
        """
        Initializes the cpSIM evaluator with the specified models.

        Args:
            sv_model_path (str): Path of the wavlm-based ECAPA-TDNN model checkpoint.
            ssl_model_path (str): Path of the wavlm SSL model directory.
            pyannote_model_path (str): Path of the pyannote diarization model directory.
        """
        self.sample_rate = 16000
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info(f"Using device: {self.device}")

        # Initialize speaker verification model
        self.sv_model = ECAPA_TDNN_WAVLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=self.sample_rate,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(
            sv_model_path, map_location=lambda storage, loc: storage
        )
        self.sv_model.load_state_dict(state_dict["model"], strict=False)
        self.sv_model.to(self.device)
        self.sv_model.eval()

        # Initialize diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            os.path.join(pyannote_model_path, "pyannote_diarization_config.yaml")
        )
        self.diarization_pipeline.to(self.device)

    @torch.no_grad()
    def get_embeddings_with_diarization(
        self, audio_paths: List[str]
    ) -> List[List[torch.Tensor]]:
        """
        Extracts speaker embeddings from audio files
            with speaker diarization (for 2-speaker conversations).

        Args:
            audio_paths: List of paths to audio files (each containing 2 speakers).

        Returns:
            List of embedding pairs, where each pair is
                [embedding_speaker1, embedding_speaker2].
        """

        embeddings_list = []
        for audio_path in tqdm(
            audio_paths, desc="Extracting embeddings with diarization"
        ):
            # Load audio waveform
            speech = load_waveform(
                audio_path, self.sample_rate, device=self.device, max_seconds=120
            )

            # Perform speaker diarization (assumes 2 speakers)
            diarization = self.diarization_pipeline(
                {"waveform": speech.unsqueeze(0), "sample_rate": self.sample_rate},
                num_speakers=2,
            )

            # Collect speech chunks for each speaker
            speaker1_chunks = []
            speaker2_chunks = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_frame = int(turn.start * self.sample_rate)
                end_frame = int(turn.end * self.sample_rate)
                chunk = speech[start_frame:end_frame]

                if speaker == "SPEAKER_00":
                    speaker1_chunks.append(chunk)
                elif speaker == "SPEAKER_01":
                    speaker2_chunks.append(chunk)

            # Handle cases where diarization fails to detect 2 speakers
            if not (speaker1_chunks and speaker2_chunks):
                logging.debug(
                    f"Insufficient speaker chunks in {audio_path} "
                    f"using full audio for both speakers"
                )
                speaker1_speech = speech
                speaker2_speech = speech
            else:
                speaker1_speech = torch.cat(speaker1_chunks, dim=0)
                speaker2_speech = torch.cat(speaker2_chunks, dim=0)

            # Extract embeddings with no gradient computation
            try:
                emb_speaker1 = self.sv_model([speaker1_speech])
                emb_speaker2 = self.sv_model([speaker2_speech])
            except Exception as e:
                logging.debug(
                    f"Encountered an error {e} when extracting embeddings with "
                    f"segmented speech, will use full audio for both speakers."
                )
                emb_speaker1 = self.sv_model([speech])
                emb_speaker2 = self.sv_model([speech])

            embeddings_list.append([emb_speaker1, emb_speaker2])

        return embeddings_list

    @torch.no_grad()
    def get_embeddings_from_pairs(
        self, audio_pairs: List[Tuple[str, str]]
    ) -> List[List[torch.Tensor]]:
        """
        Extracts speaker embeddings from pairs of single-speaker audio files.

        Args:
            audio_pairs: List of tuples (path_speaker1, path_speaker2).

        Returns:
            List of embedding pairs, where each pair is
                [embedding_speaker1, embedding_speaker2].
        """
        embeddings_list = []
        for (path1, path2) in tqdm(
            audio_pairs, desc="Extracting embeddings from pairs"
        ):
            # Load audio for each speaker
            speech1 = load_waveform(path1, self.sample_rate, device=self.device)
            speech2 = load_waveform(path2, self.sample_rate, device=self.device)

            # Extract embeddings
            emb_speaker1 = self.sv_model([speech1])
            emb_speaker2 = self.sv_model([speech2])

            embeddings_list.append([emb_speaker1, emb_speaker2])

        return embeddings_list

    def score(
        self,
        wav_path: str,
        extension: str,
        test_list: str,
        prompt_mode: str,
    ) -> float:
        """
        Computes the cpSIM score by comparing embeddings of prompt and evaluated speech.

        Args:
            wav_path: Directory containing evaluated speech files.
            test_list: Path to test list file mapping evaluated files to prompts.
            prompt_mode: Either "merge" (2-speaker prompt) or "split"
                (two single-speaker prompts).

        Returns:
            Average cpSIM score across all test pairs.
        """
        logging.info(f"Calculating cpSIM score for {wav_path} (mode: {prompt_mode})")

        # Load and parse test list
        try:
            with open(test_list, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Failed to read test list {test_list}: {e}")
            raise

        if not lines:
            raise ValueError(f"Test list {test_list} is empty")

        # Collect valid prompt-eval audio pairs
        prompt_audios = []  # For "merge": [path]; for "split": [(path1, path2)]
        eval_audios = []

        for line_num, line in enumerate(lines, 1):
            parts = line.split("\t")
            if prompt_mode == "merge":
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 columns, got {len(parts)}")
                audio_name, prompt_text, prompt_audio, text = parts
                eval_audio_path = os.path.join(wav_path, f"{audio_name}.{extension}")
                prompt_audios.append(prompt_audio)

            elif prompt_mode == "split":
                if len(parts) != 6:
                    raise ValueError(f"Expected 6 columns, got {len(parts)}")
                (
                    audio_name,
                    prompt_text1,
                    prompt_text2,
                    prompt_audio_1,
                    prompt_audio_2,
                    text,
                ) = parts
                eval_audio_path = os.path.join(wav_path, f"{audio_name}.{extension}")
                prompt_audios.append((prompt_audio_1, prompt_audio_2))

            else:
                raise ValueError(f"Invalid prompt_mode: {prompt_mode}")

            # Validate file existence
            if not os.path.exists(eval_audio_path):
                raise FileNotFoundError(f"Evaluated file not found: {eval_audio_path}")

            if prompt_mode == "merge":
                if not os.path.exists(prompt_audio):
                    raise FileNotFoundError(
                        f"Prompt merge file not found: {prompt_audio}"
                    )
            else:
                if not (
                    os.path.exists(prompt_audio_1) and os.path.exists(prompt_audio_2)
                ):
                    raise FileNotFoundError(
                        f"One or more prompt files missing in {prompt_audio_1}, "
                        f"{prompt_audio_2}"
                    )

            eval_audios.append(eval_audio_path)

        if not prompt_audios or not eval_audios:
            raise ValueError(f"No valid prompt-eval pairs found in {test_list}")

        logging.info(f"Processing {len(prompt_audios)} valid test pairs")

        # Extract embeddings for prompts and evaluations
        if prompt_mode == "merge":
            prompt_embeddings = self.get_embeddings_with_diarization(prompt_audios)
        else:
            prompt_embeddings = self.get_embeddings_from_pairs(prompt_audios)

        eval_embeddings = self.get_embeddings_with_diarization(eval_audios)

        if len(prompt_embeddings) != len(eval_embeddings):
            raise RuntimeError(
                f"Mismatch: {len(prompt_embeddings)} prompt vs "
                f" {len(eval_embeddings)} eval embeddings"
            )

        # Calculate maximum permutation similarity scores
        scores = []
        for prompt_embs, eval_embs in zip(prompt_embeddings, eval_embeddings):
            # Prompt and eval each have 2 embeddings: [emb1, emb2]
            sim1 = F.cosine_similarity(
                prompt_embs[0], eval_embs[0], dim=-1
            ) + F.cosine_similarity(prompt_embs[1], eval_embs[1], dim=-1)
            sim2 = F.cosine_similarity(
                prompt_embs[0], eval_embs[1], dim=-1
            ) + F.cosine_similarity(prompt_embs[1], eval_embs[0], dim=-1)
            max_sim = torch.max(sim1, sim2).item() / 2  # Average the sum
            scores.append(max_sim)

        return float(np.mean(scores))


if __name__ == "__main__":

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()

    # Validate test list arguments
    if not (args.test_list or args.test_list_merge):
        raise ValueError("Either --test-list or --test-list-merge must be provided")
    if args.test_list and args.test_list_merge:
        raise ValueError(
            "Only one of --test-list-split or --test-list-merge can be provided"
        )
    # Determine mode and test list
    if args.test_list:
        prompt_mode = "split"
        test_list = args.test_list
    else:
        prompt_mode = "merge"
        test_list = args.test_list_merge

    # Initialize evaluator
    sv_model_path = os.path.join(
        args.model_dir, "speaker_similarity/wavlm_large_finetune.pth"
    )
    ssl_model_path = os.path.join(args.model_dir, "speaker_similarity/wavlm_large/")
    pyannote_model_path = os.path.join(args.model_dir, "speaker_similarity/pyannote/")
    if (
        not os.path.exists(sv_model_path)
        or not os.path.exists(ssl_model_path)
        or not os.path.exists(pyannote_model_path)
    ):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        exit(1)
    cp_sim = CpSpeakerSimilarity(
        sv_model_path=sv_model_path,
        ssl_model_path=ssl_model_path,
        pyannote_model_path=pyannote_model_path,
    )
    # Compute similarity score
    score = cp_sim.score(
        wav_path=args.wav_path,
        extension=args.extension,
        test_list=test_list,
        prompt_mode=prompt_mode,
    )
    print("-" * 50)
    logging.info(f"cpSIM score: {score:.3f}")
    print("-" * 50)
