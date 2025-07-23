#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu,
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
Computes word error rate (WER) with Hubert models for LibriSpeech test sets.
"""
import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
from jiwer import compute_measures
from tqdm import tqdm
from transformers import pipeline

from zipvoice.eval.utils import load_waveform


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes WER with Hubert models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        required=True,
        help="Path to the directory containing speech files.",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="wav",
        help="Extension of the speech files. Default: wav",
    )

    parser.add_argument(
        "--decode-path",
        type=str,
        default=None,
        help="Path to the output file where WER information will be saved. "
        "If not provided, results are only printed to console.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local path of our evaluatioin model repository."
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models."
        "Will use 'tts_eval_models/wer/hubert-large-ls960-ft/'"
        " in this script",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="transcript.tsv",
        help="path of the tsv file. Each line is in the format:"
        "(audio_name, text) separated by tabs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for decoding with the Hugging Face pipeline.",
    )
    return parser


def post_process(text: str) -> str:
    """
    Cleans and normalizes text for WER calculation.
    Args:
        text (str): The input text to be processed.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_one(hypothesis: str, truth: str) -> tuple:
    """
    Computes WER and related metrics for a single hypothesis-truth pair.

    Args:
        hypothesis (str): The transcribed text from the ASR model.
        truth (str): The ground truth transcript.

    Returns:
        tuple: A tuple containing:
            - truth (str): Post-processed ground truth text.
            - hypothesis (str): Post-processed hypothesis text.
            - wer (float): Word Error Rate.
            - substitutions (int): Number of substitutions.
            - deletions (int): Number of deletions.
            - insertions (int): Number of insertions.
            - word_num (int): Number of words in the post-processed ground truth.
    """
    truth_processed = post_process(truth)
    hypothesis_processed = post_process(hypothesis)

    measures = compute_measures(truth_processed, hypothesis_processed)
    word_num = len(truth_processed.split(" "))

    return (
        truth_processed,
        hypothesis_processed,
        measures["wer"],
        measures["substitutions"],
        measures["deletions"],
        measures["insertions"],
        word_num,
    )


class SpeechEvalDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading speech waveforms and their transcripts
    for evaluation.
    """

    def __init__(self, wav_path: str, test_list: str, extension: str = "wav"):
        """
        Initializes the dataset.

        Args:
            wav_path (str): Path to the directory containing speech files.
            test_list (str): Path to the TSV file with speech file names and
                transcripts.
        """
        super().__init__()
        self.wav_names = []
        self.wav_paths = []
        self.transcripts = []
        with Path(test_list).open("r", encoding="utf8") as f:
            meta = [item.split("\t") for item in f.read().rstrip().split("\n")]
        for item in meta:
            self.wav_names.append(item[0])
            self.wav_paths.append(Path(wav_path, item[0] + "." + extension))
            self.transcripts.append(item[-1])

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index: int):
        waveform = load_waveform(
            self.wav_paths[index],
            sample_rate=16000,
            return_numpy=True,
        )
        item = {
            "array": waveform,
            "sampling_rate": 16000,
            "reference": self.transcripts[index],
            "wav_name": self.wav_names[index],
        }
        return item


def main(test_list, wav_path, extension, model_dir, decode_path, batch_size, device):
    logging.info(f"Calculating WER for {wav_path}")
    model_path = os.path.join(model_dir, "wer/hubert-large-ls960-ft/")
    if not os.path.exists(model_path):
        logging.error(
            "Please download evaluation models from "
            "https://huggingface.co/k2-fsa/TTS_eval_models"
            " and pass this dir with --model-dir"
        )
        exit(1)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        tokenizer=model_path,
    )

    dataset = SpeechEvalDataset(wav_path, test_list, extension)

    transcription_results = tqdm(
        asr_pipeline(
            dataset,
            generate_kwargs={"language": "english", "task": "transcribe"},
            batch_size=batch_size,
        ),
        total=len(dataset),
    )

    # Initialize metrics for overall WER calculation
    wers = []
    inses = []
    deles = []
    subses = []
    word_nums = 0
    if decode_path:
        # Ensure the output directory exists
        decode_dir = os.path.dirname(decode_path)
        if decode_dir and not os.path.exists(decode_dir):
            os.makedirs(decode_dir)
        fout = open(decode_path, "w", encoding="utf8")
        logging.info(f"Saving detailed WER results to: {decode_path}")
        fout.write(
            "Name\tWER\tTruth\tHypothesis\tInsertions\tDeletions\tSubstitutions\n"
        )
    for out in transcription_results:
        wav_name = out["wav_name"][0]
        transcription = out["text"].strip()
        text_ref = out["reference"][0].strip()
        truth, hypo, wer, subs, dele, inse, word_num = process_one(
            transcription, text_ref
        )
        if decode_path:
            fout.write(f"{wav_name}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n")
        wers.append(float(wer))
        inses.append(float(inse))
        deles.append(float(dele))
        subses.append(float(subs))
        word_nums += word_num

    wer = round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / word_nums * 100, 2)
    inse = np.sum(inses)
    dele = np.sum(deles)
    subs = np.sum(subses)
    print("-" * 50)
    logging.info(f"WER = {wer}%")
    logging.info(
        f"Errors: {inse} insertions, {dele} deletions, {subs} substitutions, "
        f"over {word_nums} reference words"
    )
    print("-" * 50)
    if decode_path:
        fout.write(f"WER = {wer}%\n")
        fout.write(
            f"Errors: {inse} insertions, {dele} deletions, {subs} substitutions, "
            f"over {word_nums} reference words\n"
        )
        fout.flush()


if __name__ == "__main__":

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    main(
        args.test_list,
        args.wav_path,
        args.extension,
        args.model_dir,
        args.decode_path,
        args.batch_size,
        device,
    )
