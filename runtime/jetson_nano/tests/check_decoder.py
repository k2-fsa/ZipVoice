#!/usr/bin/env python3
# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Check decoder/reference parity and exact four-step GPU residency.

--inputs is an NPZ containing t, x, text_condition, speech_condition and
guidance_scale. --reference contains FP32 outputs with keys seed_timeindex,
e.g. 1_0 ... 3_3. No model weights or prompt tensors are shipped in this PR.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from engine import Engine, DEFAULT_LIBRARY


def main(args):
    feeds = dict(np.load(args.inputs, allow_pickle=False))
    reference = np.load(args.reference, allow_pickle=False)
    steps = [0.0, 1 / 7, 1 / 3, 0.6, 1.0]
    rows = []
    with Engine(args.engine, args.library) as engine:
        for seed in [1, 2, 3]:
            initial = (
                np.random.RandomState(seed).randn(*feeds["x"].shape).astype(np.float32)
            )
            for index, t in enumerate(steps[:-1]):
                output = engine.run(
                    dict(feeds, x=initial, t=np.array([t], np.float32))
                )["v"].copy()
                expected = reference["%d_%d" % (seed, index)].astype(np.float64)
                if (
                    expected.shape != output.shape
                    or not np.isfinite(expected).all()
                    or np.linalg.norm(expected) == 0
                ):
                    raise ValueError("Invalid FP32 reference shape or values")
                relative = float(
                    np.linalg.norm(output.astype(np.float64) - expected)
                    / np.linalg.norm(expected)
                )
                row = dict(
                    seed=seed,
                    time=t,
                    finite=bool(np.isfinite(output).all()),
                    relative_rmse=relative,
                )
                print(json.dumps(row), flush=True)
                rows.append(row)
                if not row["finite"] or relative > args.max_relative_rmse:
                    raise RuntimeError("Decoder FP32 parity failed")
            host = initial.copy()
            for i in range(4):
                velocity = engine.run(
                    dict(feeds, x=host, t=np.array([steps[i]], np.float32))
                )["v"]
                host = host + velocity * np.float32(steps[i + 1] - steps[i])
            resident = engine.run4(dict(feeds, x=initial), steps)
            if not np.array_equal(host, resident):
                raise RuntimeError("Resident integration differs from the host loop")
    Path(args.output).write_text(json.dumps(dict(pass_all=True, rows=rows), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-relative-rmse", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("decoder_check.json"))
    main(parser.parse_args())
