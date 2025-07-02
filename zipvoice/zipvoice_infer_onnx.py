# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu,
#                                                       Zengwei Yao)
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
This script generates speech with our pre-trained ZipVoice or ZipVoice-Distill
    ONNX models. Required models will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:

export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 zipvoice/zipvoice_infer_onnx.py \
    --onnx-int8 False \
    --model-name "zipvoice" \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:
python3 zipvoice/zipvoice_infer_onnx.py \
    --onnx-int8 False \
    --model-name "zipvoice" \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.

Set `--onnx-int8 True` to use int8 quantizated ONNX model.
"""


import argparse
import datetime as dt
import json
import os

from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from feature import TorchAudioFbank, TorchAudioFbankConfig
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from solver import get_time_steps
from tokenizer import EmiliaTokenizer
from torch import Tensor, nn
from utils import AttributeDict, str2bool
from zipvoice_infer import get_vocoder


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--onnx-int8",
        type=str2bool,
        default=False,
        help="Whether to use the int8 model",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
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
        help="Target speech normalization rms value",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    return parser


class OnnxModel:
    def __init__(
        self,
        text_model_path: str,
        flow_matching_model_path: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_text_model(text_model_path)
        self.init_flow_matching_model(flow_matching_model_path)

    def init_text_model(self, model_path: str):
        self.text_model = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_flow_matching_model(self, model_path: str):
        self.flow_matching_model = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.flow_matching_model.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])
        print(f"feat_dim={self.feat_dim}")

    def run_text_model(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.text_model.run(
            [
                self.text_model.get_outputs()[0].name,
            ],
            {
                self.text_model.get_inputs()[0].name: tokens.numpy(),
                self.text_model.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_model.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_model.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_flow_matching_model(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        out = self.flow_matching_model.run(
            [
                self.flow_matching_model.get_outputs()[0].name,
            ],
            {
                self.flow_matching_model.get_inputs()[0].name: t.numpy(),
                self.flow_matching_model.get_inputs()[1].name: x.numpy(),
                self.flow_matching_model.get_inputs()[2].name: text_condition.numpy(),
                self.flow_matching_model.get_inputs()[3].name: speech_condition.numpy(),
                self.flow_matching_model.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: Tensor,
    speed: float = 1.0,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    """
    Generate acoustic features, given text tokens, prompts feature and prompt
    transcription's text tokens.

    Args:
        tokens: a list of list of text tokens.
        prompt_tokens: a list of list of prompt tokens.
        prompt_features: the prompt feature with the shape
            (batch_size, seq_len, feat_dim).
        speed : speed control.
        t_shift: time shift.
        guidance_scale: the guidance scale for classifier-free guidance.
        num_step: the number of steps to use in the ODE solver.
    """
    # Run text encoder
    assert len(tokens) == len(prompt_tokens) == 1
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    text_condition = model.run_text_model(
        tokens, prompt_tokens, prompt_features_len, speed
    )

    batch_size, num_frames, _ = text_condition.shape
    assert batch_size == 1
    feat_dim = model.feat_dim

    # Run flow matching model
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )  # (B, T, F)
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        v = model.run_flow_matching_model(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    x = x[:, prompt_features_len.item():, :]
    return x


# Copied from zipvoice/zipvoice_infer.py, but call an external sample function
def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: TorchAudioFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """
    Generate waveform of a text based on a given prompt
        waveform and its transcription.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (TorchAudioFbank): The feature extractor used to
            extract acoustic features.
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
    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Load and preprocess prompt wav
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    )

    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    pred_features = sample(
        model=model,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        speed=speed,
        t_shift=t_shift,
        guidance_scale=guidance_scale,
        num_step=num_step,
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


def generate(
    res_dir: str,
    test_list: str,
    model: OnnxModel,
    vocoder: nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: TorchAudioFbank,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
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
        metrics = generate_sentence(
            save_path=save_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
        )
        print(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    print(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    print(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    print(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))

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
            print(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    model_config = hf_hub_download("zhu-han/ZipVoice", filename="model.json")

    with open(model_config, "r") as f:
        model_config = json.load(f)
        for key, value in model_config["model"].items():
            setattr(params, key, value)
        for key, value in model_config["feature"].items():
            setattr(params, key, value)

    token_file = hf_hub_download("zhu-han/ZipVoice", filename="tokens_emilia.txt")

    tokenizer = EmiliaTokenizer(token_file)

    params.vocab_size = tokenizer.vocab_size
    params.pad_id = tokenizer.pad_id
    fix_random_seed(params.seed)

    if params.model_name == "zipvoice_distill":
        dirname = "exp_zipvoice_distill"
    else:
        dirname = "exp_zipvoice"

    if not params.onnx_int8:
        text_model_path = f"{dirname}/text_model.onnx"
        flow_matching_model_path = f"{dirname}/flow_matching_model.onnx"
    else:
        text_model_path = f"{dirname}/text_model_int8.onnx"
        flow_matching_model_path = f"{dirname}/flow_matching_model_int8.onnx"

    text_model_path = hf_hub_download(
        "zhu-han/ZipVoice", filename=text_model_path
    )
    flow_matching_model_path = hf_hub_download(
        "zhu-han/ZipVoice", filename=flow_matching_model_path
    )

    model = OnnxModel(text_model_path, flow_matching_model_path)

    vocoder = get_vocoder()
    vocoder.eval()

    config = TorchAudioFbankConfig(
        sampling_rate=params.sampling_rate,
        n_mels=params.feat_dim,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
    )
    feature_extractor = TorchAudioFbank(config)

    if params.test_list:
        os.makedirs(params.res_dir, exist_ok=True)
        generate(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
        )
    else:
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
        )
    print("Done")


if __name__ == "__main__":
    main()
