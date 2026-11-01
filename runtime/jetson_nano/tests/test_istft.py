# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import kaldi_native_fbank as knf
from engine import DEFAULT_LIBRARY, GpuIstft


def check(library):
    cpu = knf.IStft(
        knf.StftConfig(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window_type="hann",
            center=True,
            pad_mode="reflect",
            normalized=False,
        )
    )
    for frames in [5, 456]:
        random = np.random.RandomState(frames)
        mag = np.abs(random.randn(513, frames)).astype(np.float32)
        phase = random.uniform(-np.pi, np.pi, (513, frames)).astype(np.float32)
        cosine, sine = np.cos(phase), np.sin(phase)
        expected = np.asarray(
            cpu(
                knf.StftResult(
                    real=(mag * cosine).T.reshape(-1).tolist(),
                    imag=(mag * sine).T.reshape(-1).tolist(),
                    num_frames=frames,
                )
            ),
            np.float32,
        )
        with GpuIstft(frames, library) as gpu:
            actual = gpu(mag, cosine, sine).copy()
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        assert (
            np.linalg.norm(actual.astype(np.float64) - expected)
            / np.linalg.norm(expected)
            < 1e-5
        )
        assert np.max(np.abs(actual - expected)) < 1e-4
        print("ISTFT passed: frames=%d" % frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    check(parser.parse_args().library)
