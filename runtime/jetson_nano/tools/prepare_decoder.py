#!/usr/bin/env python3
# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a static ZipVoice-Distill ONNX decoder for TRT6 implicit batch.

This is intentionally a host-side, fixed-1024-frame conversion. It preserves
the values of source FP16 weights while doing graph/reference math in FP32.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from graph_utils import promote, reorder
from fold import run as fold
from lower import run as lower
from implicit import run as implicit


def fix_softmax(model, shapes):
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "Softmax":
            nodes.append(node)
            continue
        shape = shapes[node.input[0]]
        axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
        if axis not in (-1, len(shape)):
            raise ValueError("Only last-axis attention Softmax is supported")
        prefix = node.output[0] + "__nano_softmax"
        names = []
        for suffix, dims in [("flat", [1, -1, shape[-1]]), ("restore", [1] + shape)]:
            name = prefix + "_" + suffix
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(dims, np.int64), name)
            )
            model.graph.input.append(
                helper.make_tensor_value_info(name, onnx.TensorProto.INT64, [len(dims)])
            )
            names.append(name)
        before, after = prefix + "_in", prefix + "_out"
        nodes.extend(
            [
                helper.make_node("Reshape", [node.input[0], names[0]], [before]),
                helper.make_node("Softmax", [before], [after], axis=-1),
                helper.make_node("Reshape", [after, names[1]], list(node.output)),
            ]
        )
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def prepare(source, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    expected = {"t", "x", "text_condition", "speech_condition", "guidance_scale"}
    if {v.name for v in model.graph.input} != expected:
        raise ValueError("Expected the five-input ZipVoice-Distill ONNX decoder")
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if value.name in {"t", "guidance_scale"}:
            del dims[:]
            dims.add().dim_value = 1
        else:
            if len(dims) != 3 or dims[2].dim_value != 100:
                raise ValueError("Expected [batch, frames, 100] inputs")
            for dim, size in zip(dims, [1, 1024, 100]):
                dim.ClearField("dim_param")
                dim.dim_value = size
    reorder(model.graph)
    promote(model.graph)
    onnx.checker.check_model(model)
    onnx.save(model, str(root / "decoder_fixed_fp32.onnx"))
    del model
    gc.collect()
    for operation in [fold, lower, implicit]:
        operation(root)
        gc.collect()
    model = onnx.load(str(root / "decoder_trt6_implicit.onnx"))
    shapes = json.loads((root / "static_shapes.json").read_text())
    count = fix_softmax(model, shapes)
    if count != 16:
        raise ValueError("Expected 16 attention Softmax nodes; model layout changed")
    onnx.checker.check_model(model, full_check=True)
    target = root / "decoder_trt6.onnx"
    onnx.save(model, str(target))
    print(json.dumps({"prepared": str(target), "softmax_rewrites": count}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input, args.output_dir)
