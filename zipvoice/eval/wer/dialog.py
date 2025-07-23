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
Computes WER or cpWER for English dialogue speech with WhisperD
or compute WER for Chinese with Paraformer.
"""

import argparse
import logging
import os
import re
import string
from typing import List, Tuple

import numpy as np
import torch
import zhconv
from funasr import AutoModel
from jiwer import compute_measures
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    pipeline,
)
from zhon.hanzi import punctuation

from zipvoice.eval.utils import load_waveform


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes WER or cpWER for English dialogue speech"
        " with WhisperD or compute WER for Chinese with Paraformer.",
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
        "This script expects 'tts_eval_models/wer/whisper-d-v1a/' for English "
        "and 'tts_eval_models/wer/paraformer-zh/' for Chinese within this directory.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="test.tsv",
        help="Path to the tsv file for speaker splitted prompts. "
        "Each line contains (audio_name, prompt_text_1, prompt_text_2, "
        "prompt_audio_1, prompt_audio_2, text) separated by tabs.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "en"],
        required=True,
        help="Language of the audio and transcripts for "
        "decoding ('zh' for Chinese or 'en' for English).",
    )
    parser.add_argument(
        "--cpwer",
        action="store_true",
        help="whether to compute the cpWER",
    )
    return parser


def load_en_model(model_dir, device):
    model_path = os.path.join(model_dir, "wer/whisper-d-v1a/")
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
    tokenizer = WhisperTokenizer.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16
    )

    model.generation_config.suppress_tokens = None
    model.generation_config.forced_decoder_ids = None
    # Using pipline to handle long audios
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
    )
    return pipe


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
    text = re.sub(r"\[.*?\]|<.*?>|\(.*?\)", "", text)
    for x in punctuation_all:
        if x == "'":
            continue
        text = text.replace(x, "")
    text = re.sub(r"\s+", " ", text).strip()
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


def process_one_cpwer(hypothesis: str, truth: str, lang: str) -> tuple:
    """
    Computes cpWER and related metrics for a single hypothesis-truth pair.

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
    assert lang == "en"
    truths = split_dialogue(truth)
    hypotheses = split_dialogue(hypothesis)
    for i in range(2):
        truths[i] = post_process(truths[i], lang)
        hypotheses[i] = post_process(hypotheses[i], lang)

    measures_1 = compute_measures(
        f"{truths[0]} {truths[1]}", f"{hypotheses[0]} {hypotheses[1]}"
    )
    measures_2 = compute_measures(
        f"{truths[0]} {truths[1]}", f"{hypotheses[1]} {hypotheses[0]}"
    )
    truth = f"[S1] {truths[0]} [S2] {truths[1]}"
    if measures_1["wer"] < measures_2["wer"]:
        measures = measures_1
        hypothesis = f"[S1] {hypotheses[0]} [S2] {hypotheses[1]}"
    else:
        measures = measures_2
        hypothesis = f"[S1] {hypotheses[1]} [S2] {hypotheses[0]}"
    truth = re.sub(r"\s+", " ", truth)
    hypothesis = re.sub(r"\s+", " ", hypothesis)
    word_num = len(truth.split(" ")) - 2
    return (
        truth,
        hypothesis,
        measures["wer"],
        measures["substitutions"],
        measures["deletions"],
        measures["insertions"],
        word_num,
    )


def split_dialogue(text):
    segments = re.split(r"\[S[1-9]\]", text)
    segments = [segment.strip() for segment in segments]
    spk1_texts = " ".join(segments[::2])
    spk2_texts = " ".join(segments[1::2])
    return [spk1_texts, spk2_texts]


class SpeechEvalDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading speech waveforms and their transcripts
    for evaluation. Will only keep shorter-than-30s waveforms if in `cpwer` mode.
    """

    def __init__(
        self, wav_transcript_path_pair: List[Tuple[str, str]], cpwer: bool = False
    ):
        super().__init__()
        if cpwer:
            self.wav_transcript_path_pair = []
            for wav_path, transcript in wav_transcript_path_pair:
                waveform = load_waveform(
                    wav_path,
                    sample_rate=16000,
                )
                if len(waveform) / 16000 <= 30:
                    self.wav_transcript_path_pair.append((wav_path, transcript))
        else:
            self.wav_transcript_path_pair = wav_transcript_path_pair

    def __len__(self):
        return len(self.wav_transcript_path_pair)

    def __getitem__(self, index: int):
        waveform = load_waveform(
            self.wav_transcript_path_pair[index][0],
            sample_rate=16000,
            return_numpy=True,
        )
        item = {
            "array": waveform,
            "sampling_rate": 16000,
            "reference": self.wav_transcript_path_pair[index][1],
            "wav_path": self.wav_transcript_path_pair[index][0],
        }
        return item


def main(test_list, wav_dir, extension, model_dir, decode_path, lang, cpwer, device):
    logging.info(f"Calculating WER for {wav_dir} (cpwer={cpwer})")
    if lang == "en":
        model = load_en_model(model_dir, device=device)
    elif lang == "zh":
        model = load_zh_model(model_dir)
    params = []
    for line in open(test_list).readlines():
        line = line.strip()
        assert len(line.split("\t")) == 6
        items = line.split("\t")
        wav_name, text_ref = items[0], items[-1]
        file_path = os.path.join(wav_dir, wav_name + "." + extension)
        assert os.path.exists(file_path), f"{file_path}"
        params.append((file_path, text_ref))

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

    # Initialize metrics for overall WER calculation
    wers = []
    inses = []
    deles = []
    subses = []
    word_nums = 0
    if cpwer:
        cp_wers = []
        cp_inses = []
        cp_deles = []
        cp_subses = []
        cp_word_nums = 0
    if decode_path:
        fout = open(decode_path, "w")
    if lang == "zh":
        for wav_path, text_ref in tqdm(params):
            res = model.generate(input=wav_path, batch_size_s=300, disable_pbar=True)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, "zh-cn")

            truth, hypo, wer, subs, dele, inse, word_num = process_one(
                transcription, text_ref, lang
            )
            if decode_path:
                fout.write(
                    f"{wav_path}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n"
                )
            wers.append(float(wer))
            inses.append(float(inse))
            deles.append(float(dele))
            subses.append(float(subs))
            word_nums += word_num
    elif lang == "en":
        dataset = SpeechEvalDataset(params, cpwer)
        bar = tqdm(
            model(
                dataset,
                generate_kwargs={"language": lang, "task": "transcribe"},
                batch_size=16,
            ),
            total=len(dataset),
        )
        for out in bar:
            transcription = out["text"]
            text_ref = out["reference"][0]
            wav_path = out["wav_path"][0]
            if cpwer:
                (
                    cp_truth,
                    cp_hypo,
                    cp_wer,
                    cp_subs,
                    cp_dele,
                    cp_inse,
                    cp_word_num,
                ) = process_one_cpwer(transcription, text_ref, lang)
                if decode_path:
                    fout.write(
                        f"{wav_path}\t{cp_wer}\t{cp_truth}\t"
                        f"{cp_hypo}\t{cp_inse}\t{cp_dele}\t{cp_subs}\n"
                    )
                cp_wers.append(float(cp_wer))
                cp_inses.append(float(cp_inse))
                cp_deles.append(float(cp_dele))
                cp_subses.append(float(cp_subs))
                cp_word_nums += cp_word_num
            truth, hypo, wer, subs, dele, inse, word_num = process_one(
                transcription, text_ref, lang
            )
            if decode_path:
                fout.write(
                    f"{wav_path}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n"
                )
            wers.append(float(wer))
            inses.append(float(inse))
            deles.append(float(dele))
            subses.append(float(subs))
            word_nums += word_num
            if cpwer:
                assert (
                    word_num == cp_word_num
                ), f"{wav_path} has {word_num} words, but {cp_word_num} cp words"

    print("-" * 50)
    if cpwer:
        cp_wer = round(
            (np.sum(cp_subses) + np.sum(cp_deles) + np.sum(cp_inses))
            / cp_word_nums
            * 100,
            2,
        )
        cp_inse = np.sum(cp_inses)
        cp_dele = np.sum(cp_deles)
        cp_subs = np.sum(cp_subses)
        logging.info(f"cpWER = {cp_wer}%")
        logging.info(
            f"Errors: {cp_inse} insertions, {cp_dele} deletions, {cp_subs} "
            f"substitutions, over {cp_word_nums} reference words"
        )
        if decode_path:
            fout.write(f"cpWER = {cp_wer}%\n")
            fout.write(
                f"Errors: {cp_inse} insertions, {cp_dele} deletions, {cp_subs} "
                f"substitutions, over {cp_word_nums} reference words\n"
            )
    wer = round((np.sum(subses) + np.sum(deles) + np.sum(inses)) / word_nums * 100, 2)
    inse = np.sum(inses)
    dele = np.sum(deles)
    subs = np.sum(subses)

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
    if args.cpwer:
        assert args.lang == "en", "Only English is supported for cpWER"
    main(
        args.test_list,
        args.wav_path,
        args.extension,
        args.model_dir,
        args.decode_path,
        args.lang,
        args.cpwer,
        device,
    )
