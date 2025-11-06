# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import os

import torch
import torchaudio
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm
from zipvoice.utils.tensorrt import load_trt

from vocos import Vocos

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda")

        parameters = json.loads(args["model_config"])["parameters"]
        for key, value in parameters.items():
            parameters[key] = value["string_value"]

        self.model_dir = parameters["model_dir"]
        self.model_name = parameters["model_name"]

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
        load_trt(self.model, parameters["trt_engine_path"])

        self.feature_extractor = VocosFbank()
        self.sampling_rate = model_config["feature"]["sampling_rate"]

        self.reference_sample_rate = int(parameters["reference_audio_sample_rate"])
        if self.reference_sample_rate != self.sampling_rate:
            self.resampler = torchaudio.transforms.Resample(self.reference_sample_rate, self.sampling_rate)
        else:
            self.resampler = None

        self.target_rms = 0.1
        self.feat_scale = 0.1
        self.speed = 1.0
        self.t_shift = 0.5
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device).eval()

    def execute(self, requests):
        prompt_wavs_list = []
        prompt_texts_list = []
        target_texts_list = []
        prompt_rms_list = []

        for request in requests:
            reference_wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            reference_wav_len_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()[0][0].decode("utf-8")
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()[0][0].decode("utf-8")

            prompt_wav = from_dlpack(reference_wav_tensor.to_dlpack())
            wav_len = from_dlpack(reference_wav_len_tensor.to_dlpack()).item()
            prompt_wav = prompt_wav[:, :wav_len].squeeze(0)

            if self.resampler:
                prompt_wav = self.resampler(prompt_wav)

            prompt_wav, prompt_rms = rms_norm(prompt_wav, self.target_rms)
            prompt_rms_list.append(prompt_rms)

            prompt_wavs_list.append(prompt_wav)
            prompt_texts_list.append(reference_text)
            target_texts_list.append(target_text)

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

        responses = []
        for i in range(len(requests)):
            feat = pred_features[i][None, :, : pred_features_lens[i]]
            wav = self.vocoder.decode(feat).squeeze(1).clamp(-1, 1)
            if prompt_rms_list[i] < self.target_rms:
                wav = wav * prompt_rms_list[i] / self.target_rms

            audio = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(wav.contiguous()))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio])
            responses.append(inference_response)
        return responses
