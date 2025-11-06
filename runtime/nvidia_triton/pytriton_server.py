import argparse
import json
import logging
import os

import numpy as np
import torch
import torchaudio
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from vocos import Vocos
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm
from zipvoice.utils.tensorrt import load_trt

LOGGER = logging.getLogger("zipvoice.pytriton_server")


class ZipVoiceModel:
    def __init__(
        self,
        model_dir,
        model_name,
        trt_engine_path,
        reference_audio_sample_rate,
        use_speaker_cache=False,
        prompt_text=None,
        prompt_audio=None,
        device="cuda",
    ):
        self.device = torch.device(device)

        self.model_dir = model_dir
        self.model_name = model_name

        token_file = os.path.join(self.model_dir, "tokens.txt")
        self.tokenizer = EmiliaTokenizer(token_file=token_file)

        model_config_path = os.path.join(self.model_dir, "model.json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        tokenizer_config = {"vocab_size": self.tokenizer.vocab_size, "pad_id": self.tokenizer.pad_id}

        if self.model_name == "zipvoice":
            self.model = ZipVoice(**model_config["model"], **tokenizer_config)
            self.num_step = 16
            self.guidance_scale = 1.0
        else:
            self.model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)
            self.num_step = 4
            self.guidance_scale = 3.0

        model_ckpt = os.path.join(self.model_dir, "model.pt")
        load_checkpoint(filename=model_ckpt, model=self.model, strict=True)

        self.model = self.model.to(self.device)
        self.model.eval()
        load_trt(self.model, trt_engine_path)

        self.feature_extractor = VocosFbank()
        self.sampling_rate = model_config["feature"]["sampling_rate"]

        self.reference_sample_rate = int(reference_audio_sample_rate)
        if self.reference_sample_rate != self.sampling_rate:
            self.resampler = torchaudio.transforms.Resample(
                self.reference_sample_rate, self.sampling_rate
            )  # .to(
            # self.device
            # )
        else:
            self.resampler = None

        self.target_rms = 0.1
        self.feat_scale = 0.1
        self.speed = 1.0
        self.t_shift = 0.5
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()

        self.speaker_info_dict = {}
        if use_speaker_cache:
            if not prompt_text or not prompt_audio:
                raise ValueError("prompt_text and prompt_audio must be provided when use_speaker_cache is True.")

            prompt_wav, sr = torchaudio.load(prompt_audio)
            if prompt_wav.shape[0] > 1:  # stereo to mono
                prompt_wav = torch.mean(prompt_wav, dim=0, keepdim=True)

            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                prompt_wav = resampler(prompt_wav)

            prompt_wav = prompt_wav.squeeze(0)
            prompt_wav, prompt_rms = rms_norm(prompt_wav, self.target_rms)

            prompt_features = self.feature_extractor.extract(
                prompt_wav.unsqueeze(0), sampling_rate=self.sampling_rate
            ).to(self.device)
            prompt_features = prompt_features * self.feat_scale
            prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])
            self.speaker_info_dict["default"] = (prompt_tokens, prompt_features, prompt_rms)

    @batch
    def __call__(self, reference_text, target_text, reference_wav=None, reference_wav_len=None):
        if (reference_wav is None) != (reference_wav_len is None):
            raise ValueError("`reference_wav` and `reference_wav_len` must be provided together or not at all.")

        batch_size = len(target_text)
        prompt_wavs_list = []
        prompt_rms_list = []
        prompt_texts_list = [t[0].decode("utf-8") for t in reference_text]
        target_texts_list = [t[0].decode("utf-8") for t in target_text]

        if reference_wav is not None:
            for i in range(batch_size):
                prompt_wav = torch.from_numpy(reference_wav[i])  # .to(self.device)
                wav_len = reference_wav_len[i][0]
                prompt_wav = prompt_wav[:wav_len].unsqueeze(0)

                if self.resampler:
                    prompt_wav = self.resampler(prompt_wav)

                prompt_wav = prompt_wav.squeeze(0)
                prompt_wav, prompt_rms = rms_norm(prompt_wav, self.target_rms)
                prompt_rms_list.append(prompt_rms)
                prompt_wavs_list.append(prompt_wav)
        else:
            prompt_rms_list = [self.target_rms] * batch_size

        prompt_features_list = []
        for prompt_wav in prompt_wavs_list:
            prompt_features = self.feature_extractor.extract(
                prompt_wav.unsqueeze(0), sampling_rate=self.sampling_rate
            ).to(self.device)
            prompt_features_list.append(prompt_features.squeeze(0))

        prompt_features_lens = torch.tensor([pf.size(0) for pf in prompt_features_list], device=self.device)
        prompt_features = torch.nn.utils.rnn.pad_sequence(
            prompt_features_list, batch_first=True, padding_value=0.0
        )
        prompt_features = prompt_features * self.feat_scale

        tokens = self.tokenizer.texts_to_token_ids(target_texts_list)
        prompt_tokens = self.tokenizer.texts_to_token_ids(prompt_texts_list)

        with torch.inference_mode():
            (
                pred_features,
                pred_features_lens,
                _,
                _,
            ) = self.model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=self.speed,
                t_shift=self.t_shift,
                duration="predict",
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
            )

        pred_features = pred_features.permute(0, 2, 1) / self.feat_scale

        wav_out_list = []
        for i in range(batch_size):
            feat = pred_features[i][None, :, : pred_features_lens[i]]
            wav = self.vocoder.decode(feat).squeeze(1).clamp(-1, 1)
            if prompt_rms_list[i] < self.target_rms:
                wav = wav * prompt_rms_list[i] / self.target_rms
            wav_out_list.append(wav.cpu().numpy().squeeze())

        max_len = max(len(w) for w in wav_out_list)
        padded_wavs = np.full((len(wav_out_list), max_len), -1.0, dtype=np.float32)
        for i, wav in enumerate(wav_out_list):
            padded_wavs[i, : len(wav)] = wav

        return {"waveform": padded_wavs}

    @batch
    def generate_with_speaker_cache(self, target_text):
        batch_size = len(target_text)
        target_texts_list = [t[0].decode("utf-8") for t in target_text]

        prompt_tokens, prompt_features, prompt_rms = self.speaker_info_dict["default"]

        # Batchify
        prompt_tokens = prompt_tokens * batch_size
        prompt_features = prompt_features.repeat(batch_size, 1, 1)
        prompt_features_lens = torch.full((batch_size,), prompt_features.size(1), device=self.device)
        prompt_rms_list = [prompt_rms] * batch_size

        tokens = self.tokenizer.texts_to_token_ids(target_texts_list)

        with torch.inference_mode():
            (
                pred_features,
                pred_features_lens,
                _,
                _,
            ) = self.model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=self.speed,
                t_shift=self.t_shift,
                duration="predict",
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
            )

        pred_features = pred_features.permute(0, 2, 1) / self.feat_scale

        wav_out_list = []
        for i in range(batch_size):
            feat = pred_features[i][None, :, : pred_features_lens[i]]
            wav = self.vocoder.decode(feat).squeeze(1).clamp(-1, 1)
            if prompt_rms_list[i] < self.target_rms:
                wav = wav * prompt_rms_list[i] / self.target_rms
            wav_out_list.append(wav.cpu().numpy().squeeze())

        max_len = max(len(w) for w in wav_out_list)
        padded_wavs = np.full((len(wav_out_list), max_len), -1.0, dtype=np.float32)
        for i, wav in enumerate(wav_out_list):
            padded_wavs[i, : len(wav)] = wav

        return {"waveform": padded_wavs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--model_name", type=str, required=True, choices=["zipvoice", "zipvoice_distill"])
    parser.add_argument("--trt_engine_path", type=str, required=True, help="Path to the TensorRT engine.")
    parser.add_argument("--reference_audio_sample_rate", type=int, default=16000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--use_speaker_cache",
        action="store_true",
        help="Use spk2info cache for reference audio.",
    )
    parser.add_argument("--prompt_text", type=str, default=None, help="Prompt text for speaker cache.")
    parser.add_argument("--prompt_audio", type=str, default=None, help="Prompt audio for speaker cache.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    model = ZipVoiceModel(
        model_dir=args.model_dir,
        model_name=args.model_name,
        trt_engine_path=args.trt_engine_path,
        reference_audio_sample_rate=args.reference_audio_sample_rate,
        use_speaker_cache=args.use_speaker_cache,
        prompt_text=args.prompt_text,
        prompt_audio=args.prompt_audio,
    )

    config = TritonConfig(http_port=args.port, grpc_port=args.port + 1, metrics_port=args.port + 2, log_verbose=0)
    with Triton(config=config) as triton:
        if args.use_speaker_cache:
            triton.bind(
                model_name="zipvoice",
                infer_func=model.generate_with_speaker_cache,
                inputs=[
                    Tensor(name="target_text", dtype=np.object_, shape=(1,)),
                ],
                outputs=[
                    Tensor(name="waveform", dtype=np.float32, shape=(-1, -1)),
                ],
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=10000),
                ),
            )
        else:
            triton.bind(
                model_name="zipvoice",
                infer_func=model,
                inputs=[
                    Tensor(name="reference_text", dtype=np.object_, shape=(1,)),
                    Tensor(name="target_text", dtype=np.object_, shape=(1,)),
                    Tensor(name="reference_wav", dtype=np.float32, shape=(-1,), optional=True),
                    Tensor(name="reference_wav_len", dtype=np.int32, shape=(1,), optional=True),
                ],
                outputs=[
                    Tensor(name="waveform", dtype=np.float32, shape=(-1, -1)),
                ],
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=10000),
                ),
            )
        LOGGER.info(f"Serving {args.model_name} model")
        triton.serve()


if __name__ == "__main__":
    main()