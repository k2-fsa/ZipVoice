# Copyright 2025 Xiaomi Corp. (authors: Fangjun Kuang)
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import kaldi_native_fbank as knf


def compute_features(samples):
    stft_config = knf.StftConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        window_type="hann",
    )
    knf_stft = knf.Stft(stft_config)
    stft_result = knf_stft(samples.tolist())
    real = np.array(stft_result.real, dtype=np.float32).reshape(
        stft_result.num_frames, -1
    )
    imag = np.array(stft_result.imag, dtype=np.float32).reshape(
        stft_result.num_frames, -1
    )

    mag = np.sqrt(real * real + imag * imag).astype(np.float32)

    mel_opts = knf.MelBanksOptions()
    mel_opts.num_bins = 100
    mel_opts.low_freq = 0
    mel_opts.high_freq = 24000 // 2
    mel_opts.is_librosa = True
    mel_opts.norm = ""
    mel_opts.use_slaney_mel_scale = False

    frame_opts = knf.FrameExtractionOptions()
    frame_opts.samp_freq = 24000
    #  frame_opts.frame_length_ms = 1024 * 1000 / 24000
    #  frame_opts.frame_shift_ms = 256 * 1000 / 24000

    mel_filters = knf.MelBanks(mel_opts, frame_opts)
    mel_features = np.zeros((mag.shape[0], 100))
    for i in range(mag.shape[0]):
        mel_features[i] = mel_filters.compute(mag[i])
    mel_features = np.log(mel_features + 1e-10)
    return mel_features


def get_timestamps(num_steps, t_shift=1):
    steps = np.linspace(0, 1, num_steps + 1)
    if t_shift != 1:
        steps = t_shift * steps / (1 + (t_shift - 1) * steps)

    return steps.tolist()


def trim_leading_silence_energy(samples, frame_size=2048, hop=512, energy_thresh=0.5):
    energies = [
        np.sum(np.abs(samples[i : i + frame_size]) ** 2)
        for i in range(0, len(samples) - frame_size, hop)
    ]
    #  print(energies)
    # First frame whose energy exceeds threshold
    frame_index = next((i for i, e in enumerate(energies) if e > energy_thresh), 0)
    frame_index = max(frame_index - 3, 0)
    start_sample = frame_index * hop
    return samples[start_sample:]
