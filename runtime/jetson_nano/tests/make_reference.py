#!/usr/bin/env python3
# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Create the 3-seed, 4-time FP32 decoder reference on a host CPU.

Use the source-weight-preserving decoder_fixed_fp32.onnx produced by the
converter, and decoder_inputs.npz from infer.py --save-feeds. Reference
generation is outside the synthesis timing boundary.
"""

import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main(args):
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    # Some ORT versions optimize static Concat/Reshape shapes differently.
    # Keep this oracle in the exported graph's operator order.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    inputs = dict(np.load(args.inputs, allow_pickle=False))
    outputs = {}
    for seed in [1, 2, 3]:
        state = np.random.RandomState(seed).randn(*inputs["x"].shape).astype(np.float32)
        for index, time in enumerate([0.0, 1 / 7, 1 / 3, 0.6]):
            output = session.run(
                None, dict(inputs, x=state, t=np.array([time], np.float32))
            )[0]
            if not np.isfinite(output).all():
                raise RuntimeError("Non-finite FP32 reference")
            outputs["%d_%d" % (seed, index)] = output
    np.savez(str(args.output), **outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    main(parser.parse_args())
