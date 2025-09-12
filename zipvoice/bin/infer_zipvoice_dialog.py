#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
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
This script generates speech with our pre-trained ZipVoice-Dialog or
    ZipVoice-Dialog-Stereo models. If no local model is specified,
    Required files will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

python3 -m zipvoice.bin.infer_zipvoice_dialog \
    --model-name zipvoice_dialog \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice_dialog` or `zipvoice_dialog_stereo`,
    which generate mono and stereo dialogues, respectively.

Each line of `test.tsv` is in the format of merged conversation:
    '{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'
    or splited conversation:
    '{wav_name}\t{spk1_prompt_transcription}\t{spk2_prompt_transcription}
        \t{spk1_prompt_wav}\t{spk2_prompt_wav}\t{text}'
"""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice_dialog import ZipVoiceDialog, ZipVoiceDialogStereo
from zipvoice.tokenizer.tokenizer import DialogTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_dialog,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice_dialog": "zipvoice_dialog",
    "zipvoice_dialog_stereo": "zipvoice_dialog_stereo",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice_dialog",
        choices=["zipvoice_dialog", "zipvoice_dialog_stereo"],
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
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of merged conversation: "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}' "
        "or splited conversation: "
        "'{wav_name}\t{spk1_prompt_transcription}\t{spk2_prompt_transcription}"
        "\t{spk1_prompt_wav}\t{spk2_prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    parser.add_argument(
        "--silence-wav",
        type=str,
        default="assets/silence.wav",
        help="Path of the silence wav file, used in two-channel generation "
        "with single-channel prompts",
    )

    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of threads to use for PyTorch on CPU.",
    )

    parser.add_argument(
        "--raw-evaluation",
        type=str2bool,
        default=False,
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=100,
        help="Maximum duration (seconds) in a single batch, including "
        "durations of the prompt and generated wavs. You can reduce it "
        "if it causes CUDA OOM.",
    )

    parser.add_argument(
        "--remove-long-sil",
        type=str2bool,
        default=False,
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    )

    return parser


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def generate_sentence_raw_evaluation(
    save_path: str,
    prompt_text: str,
    prompt_wav: Union[str, List[str]],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function directly feed the prompt_text, prompt_wav and text to the model.
        It is not efficient and can have poor results for some inappropriate inputs.
        (e.g., prompt wav contains long silence, text to be generated is too long)
        This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (Union[str, List[str]]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        assert len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(
            loaded_prompt_wavs[i], sampling_rate=sampling_rate
        )
        if loaded_prompt_wavs[i].size(0) != 1:
            loaded_prompt_wavs[i] = loaded_prompt_wavs[i].mean(0, keepdim=True)

    if len(loaded_prompt_wavs) == 1:
        prompt_wav = loaded_prompt_wavs[0]
    else:
        prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: Union[str, List[str]],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function will do the following to improve the generation quality:
        1. chunk the text according to speaker-turn symbol [S1].
        2. process chunked texts in batches.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (Union[str, List[str]]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        assert len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(
            loaded_prompt_wavs[i], sampling_rate=sampling_rate
        )
        if loaded_prompt_wavs[i].size(0) != 1:
            loaded_prompt_wavs[i] = loaded_prompt_wavs[i].mean(0, keepdim=True)

    if len(loaded_prompt_wavs) == 1:
        prompt_wav = loaded_prompt_wavs[0]
    else:
        prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 40:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (prompt shorter than 10 "
            f"seconds is recommended)."
        )
    elif prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # chunk text so that each len(prompt wav + generated wav) is around 40 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((40 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_dialog(tokens_str, max_tokens=max_tokens)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Batchify chunked texts for faster processing
    tokens_batches, chunked_index = batchify_tokens(
        chunked_tokens, max_duration, prompt_duration, token_duration
    )

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()

    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)

        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full(
            (len(batch_tokens),), prompt_features.size(1), device=device
        )

        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append((pred_features, pred_features_lens))

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):

            wav = (
                vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
                .squeeze(1)
                .clamp(-1, 1)
            )
            # Adjust wav volume if necessary
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    indexed_chunked_wavs = [
        (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
    ]
    sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
    sequential_chunked_wavs = [
        sequential_indexed_chunked_wavs[i][1]
        for i in range(len(sequential_indexed_chunked_wavs))
    ]
    final_wav = cross_fade_concat(
        sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_sentence_stereo_raw_evaluation(
    save_path: str,
    prompt_text: str,
    prompt_wav: Union[str, List[str]],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: Optional[str] = None,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function directly feed the prompt_text, prompt_wav and text to the model.
        It is not efficient and can have poor results for some inappropriate inputs.
        (e.g., prompt wav contains long silence, text to be generated is too long)
        This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (Union[str, List[str]]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        silence_wav (str): Path of the silence wav file, used in two-channel
            generation with single-channel prompts
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        assert len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(
            loaded_prompt_wavs[i], sampling_rate=sampling_rate
        )

    if len(loaded_prompt_wavs) == 1:
        assert (
            loaded_prompt_wavs[0].size(0) == 2
        ), "Merged prompt wav must be stereo for stereo dialogue generation"
        prompt_wav = loaded_prompt_wavs[0]

    else:
        assert len(loaded_prompt_wavs) == 2
        if loaded_prompt_wavs[0].size(0) == 2:
            prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)
        else:
            assert loaded_prompt_wavs[0].size(0) == 1
            silence_wav, silence_sampling_rate = torchaudio.load(silence_wav)
            assert silence_sampling_rate == sampling_rate
            prompt_wav = silence_wav[
                :, : loaded_prompt_wavs[0].size(1) + loaded_prompt_wavs[1].size(1)
            ]
            prompt_wav[0, : loaded_prompt_wavs[0].size(1)] = loaded_prompt_wavs[0]
            prompt_wav[1, loaded_prompt_wavs[0].size(1) :] = loaded_prompt_wavs[1]

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    feat_dim = pred_features.size(1) // 2
    wav_left = vocoder.decode(pred_features[:, :feat_dim]).squeeze(1).clamp(-1, 1)
    wav_right = (
        vocoder.decode(pred_features[:, feat_dim : feat_dim * 2])
        .squeeze(1)
        .clamp(-1, 1)
    )

    wav = torch.cat([wav_left, wav_right], dim=0)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate_sentence_stereo(
    save_path: str,
    prompt_text: str,
    prompt_wav: Union[str, List[str]],
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: Optional[str] = None,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function will do the following to improve the generation quality:
        1. chunk the text according to speaker-turn symbol [S1].
        2. process chunked texts in batches.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (Union[str, List[str]]): Path to the prompt wav file, can be
            one or two wav files, which corresponding to a merged conversational
            speech or two seperate speaker's speech.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (DialogTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        silence_wav (str): Path of the silence wav file, used in two-channel
            generation with single-channel prompts
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and preprocess prompt wav
    if isinstance(prompt_wav, str):
        prompt_wav = [
            prompt_wav,
        ]
    else:
        assert len(prompt_wav) == 2 and isinstance(prompt_wav[0], str)

    loaded_prompt_wavs = prompt_wav
    for i in range(len(prompt_wav)):
        loaded_prompt_wavs[i] = load_prompt_wav(
            loaded_prompt_wavs[i], sampling_rate=sampling_rate
        )

    if len(loaded_prompt_wavs) == 1:
        assert (
            loaded_prompt_wavs[0].size(0) == 2
        ), "Merged prompt wav must be stereo for stereo dialogue generation"
        prompt_wav = loaded_prompt_wavs[0]

    else:
        assert len(loaded_prompt_wavs) == 2
        if loaded_prompt_wavs[0].size(0) == 2:
            prompt_wav = torch.cat(loaded_prompt_wavs, dim=1)
        else:
            assert loaded_prompt_wavs[0].size(0) == 1
            silence_wav, silence_sampling_rate = torchaudio.load(silence_wav)
            assert silence_sampling_rate == sampling_rate
            prompt_wav = silence_wav[
                :, : loaded_prompt_wavs[0].size(1) + loaded_prompt_wavs[1].size(1)
            ]
            prompt_wav[0, : loaded_prompt_wavs[0].size(1)] = loaded_prompt_wavs[0]
            prompt_wav[1, loaded_prompt_wavs[0].size(1) :] = loaded_prompt_wavs[1]

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 40:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (prompt shorter than 10 "
            f"seconds is recommended)."
        )
    elif prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    # Tokenize text (str tokens), punctuations will be preserved.
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    # chunk text so that each len(prompt wav + generated wav) is around 40 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((40 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_dialog(tokens_str, max_tokens=max_tokens)

    # Tokenize text (int tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    # Batchify chunked texts for faster processing
    tokens_batches, chunked_index = batchify_tokens(
        chunked_tokens, max_duration, prompt_duration, token_duration
    )

    # Start predicting features
    chunked_features = []
    start_t = dt.datetime.now()

    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)

        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full(
            (len(batch_tokens),), prompt_features.size(1), device=device
        )

        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        # Postprocess predicted features
        pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)
        chunked_features.append((pred_features, pred_features_lens))

    # Start vocoder processing
    chunked_wavs = []
    start_vocoder_t = dt.datetime.now()

    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):

            feat_dim = pred_features.size(1) // 2
            wav_left = (
                vocoder.decode(
                    pred_features[i][None, :feat_dim, : pred_features_lens[i]]
                )
                .squeeze(1)
                .clamp(-1, 1)
            )
            wav_right = (
                vocoder.decode(
                    pred_features[i][
                        None, feat_dim : feat_dim * 2, : pred_features_lens[i]
                    ]
                )
                .squeeze(1)
                .clamp(-1, 1)
            )
            wav = torch.cat([wav_left, wav_right], dim=0)

            # Adjust wav volume if necessary
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    # Finish model generation
    t = (dt.datetime.now() - start_t).total_seconds()

    # Merge chunked wavs
    indexed_chunked_wavs = [
        (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
    ]
    sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
    sequential_chunked_wavs = [
        sequential_indexed_chunked_wavs[i][1]
        for i in range(len(sequential_indexed_chunked_wavs))
    ]
    final_wav = cross_fade_concat(
        sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate
    )
    final_wav = remove_silence(
        final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0
    )

    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_list(
    model_name: str,
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: DialogTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.5,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    silence_wav: Optional[str] = None,
    raw_evaluation: bool = False,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        items = line.strip().split("\t")
        if len(items) == 6:
            (
                wav_name,
                prompt_text_1,
                prompt_text_2,
                prompt_wav_1,
                prompt_wav_2,
                text,
            ) = items
            prompt_text = f"[S1]{prompt_text_1}[S2]{prompt_text_2}"
            prompt_wav = [prompt_wav_1, prompt_wav_2]
        elif len(items) == 4:
            wav_name, prompt_text, prompt_wav, text = items
        else:
            raise ValueError(f"Invalid line: {line}")
        assert text.startswith("[S1]")

        save_path = f"{res_dir}/{wav_name}.wav"

        common_params = {
            "save_path": save_path,
            "prompt_text": prompt_text,
            "prompt_wav": prompt_wav,
            "text": text,
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "device": device,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "target_rms": target_rms,
            "feat_scale": feat_scale,
            "sampling_rate": sampling_rate,
        }

        if model_name == "zipvoice_dialog":
            if raw_evaluation:
                metrics = generate_sentence_raw_evaluation(**common_params)
            else:
                metrics = generate_sentence(
                    **common_params,
                    max_duration=max_duration,
                    remove_long_sil=remove_long_sil,
                )
        else:
            assert model_name == "zipvoice_dialog_stereo"
            if raw_evaluation:
                metrics = generate_sentence_stereo_raw_evaluation(
                    **common_params,
                    silence_wav=silence_wav,
                )
            else:
                metrics = generate_sentence_stereo(
                    **common_params,
                    silence_wav=silence_wav,
                    max_duration=max_duration,
                    remove_long_sil=remove_long_sil,
                )
        logging.info(f"[Sentence: {i}] Saved to: {save_path}")
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    logging.info(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    logging.info(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    logging.info(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.num_thread)
    torch.set_num_interop_threads(args.num_thread)

    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    assert (
        params.test_list is not None
    ), "For inference, please provide prompts and text with '--test-list'"

    if params.model_dir is not None:
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")
        for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
        logging.info(
            f"Using {params.model_name} in local model dir {params.model_dir}, "
            f"checkpoint {params.checkpoint_name}"
        )
    else:
        logging.info(f"Using pretrained {params.model_name} model from the Huggingface")
        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt"
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )

        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )

    tokenizer = DialogTokenizer(token_file=token_file)

    tokenizer_config = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "spk_a_id": tokenizer.spk_a_id,
        "spk_b_id": tokenizer.spk_b_id,
    }

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice_dialog":
        model = ZipVoiceDialog(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        assert params.model_name == "zipvoice_dialog_stereo"
        model = ZipVoiceDialogStereo(
            **model_config["model"],
            **tokenizer_config,
        )

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        params.device = torch.device("mps")
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    model = model.to(params.device)
    model.eval()

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(params.device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        if params.model_name == "zipvoice_dialog":
            num_channels = 1
        else:
            assert params.model_name == "zipvoice_dialog_stereo"
            num_channels = 2
        feature_extractor = VocosFbank(num_channels=num_channels)
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    os.makedirs(params.res_dir, exist_ok=True)
    generate_list(
        model_name=params.model_name,
        res_dir=params.res_dir,
        test_list=params.test_list,
        model=model,
        vocoder=vocoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=params.device,
        num_step=params.num_step,
        guidance_scale=params.guidance_scale,
        speed=params.speed,
        t_shift=params.t_shift,
        target_rms=params.target_rms,
        feat_scale=params.feat_scale,
        sampling_rate=params.sampling_rate,
        silence_wav=params.silence_wav,
        raw_evaluation=params.raw_evaluation,
        max_duration=params.max_duration,
        remove_long_sil=params.remove_long_sil,
    )
    logging.info("Done")


if __name__ == "__main__":

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
