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
This script generates speech with our pre-trained ZipVoice or
    ZipVoice-Distill models. If no local model is specified,
    Required files will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.
"""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
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
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
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
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
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
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
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

    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
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
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
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
        1. chunk the text according to punctuations.
        2. process chunked texts in batches.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
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

    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
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

    # chunk text so that each len(prompt wav + generated wav) is around 25 seconds.
    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        len(prompt_tokens_str) * speed
    )
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

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


def generate_list(
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
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
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
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

        if raw_evaluation:
            metrics = generate_sentence_raw_evaluation(**common_params)
        else:
            metrics = generate_sentence(
                **common_params,
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

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

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

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        assert params.model_name == "zipvoice_distill"
        model = ZipVoiceDistill(
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
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    if params.test_list:
        res_dir = params.res_dir
        os.makedirs(res_dir, exist_ok=True)
        generate_list(
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
            raw_evaluation=params.raw_evaluation,
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
        )
    else:
        assert (
            not params.raw_evaluation
        ), "Raw evaluation is only valid with --test-list"
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
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
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
        )
        logging.info(f"Saved to: {params.res_wav_path}")
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
