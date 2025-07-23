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
Computes word error rate (WER) with Whisper-large-v3 for English and
Paraformer for Chinese. Intended to evaluate WERs on Seed-TTS test sets.
"""

import argparse
import logging
import os
import string

import numpy as np
import scipy
import soundfile as sf
import torch
import zhconv
from funasr import AutoModel
from jiwer import compute_measures
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from zhon.hanzi import punctuation


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes WER with Whisper and Paraformer models, "
        "following Seed-TTS.",
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
        help="Local path of evaluation models repository. "
        "Download from https://huggingface.co/k2-fsa/TTS_eval_models. "
        "This script expects 'tts_eval_models/wer/whisper-large-v3/' for English "
        "and 'tts_eval_models/wer/paraformer-zh/' for Chinese within this directory.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="test.tsv",
        help="path of the tsv file. Each line is in the format:"
        "(audio_name, prompt_text,prompt_audio, text) separated by tabs.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "en"],
        required=True,
        help="Language of the audio and transcripts for "
        "decoding ('zh' for Chinese or 'en' for English).",
    )
    return parser


def load_en_model(model_dir):
    model_path = os.path.join(model_dir, "wer/whisper-large-v3/")
    if not os.path.exists(model_path):
        logging.error(
            f"Error: Whisper model not found at {model_path}. "
            "Please download evaluation modelss from "
            "https://huggingface.co/k2-fsa/TTS_eval_models "
            "and pass this directory with --model-dir."
        )
        exit(1)
    logging.info(f"Loading Whisper model from: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return processor, model


def load_zh_model(model_dir):
    model_path = os.path.join(model_dir, "wer/paraformer-zh/")
    if not os.path.exists(model_path):
        logging.error(
            f"Error: Paraformer model not found at {model_path}. "
            "Please download evaluation modelss from "
            "https://huggingface.co/k2-fsa/TTS_eval_models "
            "and pass this directory with --model-dir."
        )
        exit(1)
    logging.info(f"Loading Paraformer model from: {model_path}")
    model = AutoModel(model=model_path, disable_update=True)
    return model


def post_process(text: str, lang: str) -> str:
    """
    Cleans and normalizes text for WER calculation.
    Args:
        text (str): The input text to be processed.
        lang (str): The language of the input text.

    Returns:
        str: The cleaned and normalized text.
    """
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        if x == "'":
            continue
        text = text.replace(x, "")

    text = text.replace("  ", " ")

    if lang == "zh":
        text = " ".join([x for x in text])
    elif lang == "en":
        text = text.lower()
    else:
        raise NotImplementedError
    return text


def process_one(hypothesis: str, truth: str, lang: str) -> tuple:
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
    truth_processed = post_process(truth, lang)
    hypothesis_processed = post_process(hypothesis, lang)

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


def main(test_list, wav_path, extension, model_path, decode_path, lang, device):
    logging.info(f"Calculating WER for {wav_path}")
    if lang == "en":
        processor, model = load_en_model(model_path)
        model.to(device)
    elif lang == "zh":
        model = load_zh_model(model_path)
    params = []
    for line in open(test_list).readlines():
        line = line.strip()
        items = line.split("\t")
        wav_name, text_ref = items[0], items[-1]
        file_path = os.path.join(wav_path, wav_name + "." + extension)
        assert os.path.exists(file_path), f"{file_path}"

        params.append((file_path, text_ref))
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
        fout = open(decode_path, "w")
    for wav_path, text_ref in tqdm(params):
        if lang == "en":
            wav, sr = sf.read(wav_path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(
                wav, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
        elif lang == "zh":
            res = model.generate(input=wav_path, batch_size_s=300, disable_pbar=True)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, "zh-cn")

        truth, hypo, wer, subs, dele, inse, word_num = process_one(
            transcription, text_ref, lang
        )
        if decode_path:
            fout.write(f"{wav_path}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n")
        wers.append(float(wer))
        inses.append(float(inse))
        deles.append(float(dele))
        subses.append(float(subs))
        word_nums += word_num

    wer_avg = round(np.mean(wers) * 100, 2)
    wer = round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / word_nums * 100, 2)
    inse = np.sum(inses)
    dele = np.sum(deles)
    subs = np.sum(subses)
    print("-" * 50)
    # The official evaluation codes of Seed-TTS uses the average of WERs
    # instead of the weighted average of WERs.
    logging.info(f"Seed-TTS WER: {wer_avg}%\n")
    logging.info(f"WER: {wer}%\n")
    logging.info(
        f"Errors: {inse} insertions, {dele} deletions, {subs} substitutions, "
        f"over {word_nums} reference words"
    )
    print("-" * 50)
    if decode_path:
        fout.write(f"SeedTTS WER: {wer_avg}%\n")
        fout.write(f"WER: {wer}%\n")
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
        args.lang,
        device,
    )
