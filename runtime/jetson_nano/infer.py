#!/usr/bin/env python3
# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape Nano synthesis example using existing encoder/vocoder exports.

The lightweight lexicon frontend is intended for the Chinese benchmark. For a
different frontend, pass explicit token IDs instead. This does not replace the
upstream tokenizer, dynamic-length inference, or model downloading code.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from engine import DEFAULT_LIBRARY, Engine, GpuIstft
from features import compute_features, get_timestamps, trim_leading_silence_energy
from fixed_condition import encode_fixed


def token_ids(text, tokens_file, lexicon_file):
    tokens = {}
    for line in tokens_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            tokens[" " if len(fields) == 1 else fields[0]] = int(fields[-1])
    lexicon = {}
    for line in lexicon_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            lexicon[fields[0]] = fields[1:]
    text = text.translate(str.maketrans("，。！？", ",.!?"))
    if not text.endswith("."):
        text += "."
    result = []
    for char in text:
        if char not in lexicon and char not in tokens and char not in ",.!? ":
            raise ValueError(
                "Character absent from benchmark lexicon: "
                + repr(char)
                + "; use --token-ids for another frontend"
            )
        result.extend(
            tokens[phone] for phone in lexicon.get(char, [char]) if phone in tokens
        )
    if not result:
        raise ValueError("Empty token sequence")
    return np.array([result], dtype=np.int64)


def run(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = output_dir / "timings.jsonl"
    if results.exists():
        raise FileExistsError("Use a new output directory to avoid mixing timing runs")

    def emit(row):
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.token_ids:
        ids = json.loads(args.token_ids.read_text())
        tokens = np.array([ids["text"]], dtype=np.int64)
        prompt_ids = np.array([ids["prompt"]], dtype=np.int64)
        if tokens.size == 0 or prompt_ids.size == 0:
            raise ValueError("Token sequences must not be empty")
    else:
        if not args.tokens or not args.lexicon or not args.text or not args.prompt_text:
            raise ValueError("Provide text/prompt-text/tokens/lexicon or --token-ids")
        tokens = token_ids(args.text, args.tokens, args.lexicon)
        prompt_ids = token_ids(args.prompt_text, args.tokens, args.lexicon)
    reference, sample_rate = sf.read(str(args.prompt_wav), dtype="float32")
    if sample_rate != 24000 or reference.ndim != 1 or reference.size == 0:
        raise ValueError("Expected a nonempty mono 24 kHz prompt")
    rms = float(np.sqrt(np.mean(reference * reference)))
    normalized = reference * (0.1 / rms) if 0 < rms < 0.1 else reference
    features = (compute_features(normalized) * 0.1).astype(np.float32)
    prompt_frames = features.shape[0]
    if not 0 < prompt_frames < 1024:
        raise ValueError("Prompt must leave room within the static 1024-frame engine")
    speech = np.pad(features, ((0, 1024 - prompt_frames), (0, 0)))[None]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    started = time.perf_counter()
    encoder = ort.InferenceSession(
        str(args.encoder), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    emit(dict(event="encoder_loaded", seconds=time.perf_counter() - started))
    names = [value.name for value in encoder.get_inputs()]
    speed = args.speed
    if args.fit_frames:
        _, speed, seconds, history = encode_fixed(
            encoder, tokens, prompt_ids, prompt_frames, 1024
        )
        emit(
            dict(
                event="fixed_shape_speed_fit",
                seconds=seconds,
                speed=speed,
                history=history,
            )
        )
    encoder_feeds = {
        names[0]: tokens,
        names[1]: prompt_ids,
        names[2]: np.array(prompt_frames, np.int64),
        names[3]: np.array(speed, np.float32),
    }
    steps = get_timestamps(4, t_shift=0.5)
    with Engine(args.decoder, args.library) as decoder, Engine(
        args.vocoder, args.library
    ) as vocoder, GpuIstft(1024 - prompt_frames, args.library) as istft:

        def synth(seed, warmup):
            start = time.perf_counter()
            condition = encoder.run(None, encoder_feeds)[0]
            if condition.shape != (1, 1024, 100):
                raise ValueError(
                    "Encoder produced %s; choose --fit-frames or a compatible speed"
                    % (condition.shape,)
                )
            initial = (
                np.random.RandomState(seed).randn(*condition.shape).astype(np.float32)
            )
            feeds = dict(
                t=np.array([steps[0]], np.float32),
                guidance_scale=np.array([1.0], np.float32),
                x=initial,
                text_condition=condition,
                speech_condition=speech,
            )
            start_decoder = time.perf_counter()
            state = decoder.run4(feeds, steps)
            decoder_seconds = time.perf_counter() - start_decoder
            if not np.isfinite(state).all():
                raise RuntimeError("Non-finite decoder state")
            mel = np.ascontiguousarray(
                (state[:, prompt_frames:] / 0.1).transpose(0, 2, 1), dtype=np.float32
            )
            # TRT6 implicit-batch bindings omit the leading batch-one axis.
            spectra = vocoder.run({"mels": mel[0]})
            samples = trim_leading_silence_energy(
                istft(spectra["mag"], spectra["x"], spectra["y"])
            )
            elapsed = time.perf_counter() - start
            if not len(samples) or not np.isfinite(samples).all():
                raise RuntimeError("Invalid output audio")
            duration = len(samples) / 24000
            stem = "warmup" if warmup else "seed%d" % seed
            sf.write(
                str(output_dir / (stem + ".wav")), samples, 24000, subtype="PCM_16"
            )
            if args.save_mel:
                np.save(str(output_dir / (stem + ".npy")), mel)
            if args.save_feeds and warmup:
                np.savez(str(output_dir / "decoder_inputs.npz"), **feeds)
            emit(
                dict(
                    event="synthesis",
                    warmup=warmup,
                    seed=seed,
                    steps=4,
                    total_frames=1024,
                    prompt_frames=prompt_frames,
                    speed=speed,
                    cached_prompt_features=True,
                    elapsed_seconds=elapsed,
                    audio_seconds=duration,
                    rtf=elapsed / duration,
                    decoder_seconds=decoder_seconds,
                )
            )

        synth(args.seed - 1, True)
        for i in range(args.repeats):
            synth(args.seed + i, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["encoder", "decoder", "vocoder", "prompt-wav", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--token-ids", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--prompt-text")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--fit-frames",
        action="store_true",
        help="Explicitly adapt speed to fit this static engine; excluded from steady timing",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--save-mel", action="store_true")
    parser.add_argument(
        "--save-feeds",
        action="store_true",
        help="Save real warmup decoder inputs for host FP32 reference generation",
    )
    run(parser.parse_args())
