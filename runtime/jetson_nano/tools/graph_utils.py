# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import onnx
from onnx import numpy_helper


def captures(g):
    made = (
        {o for n in g.node for o in n.output}
        | {x.name for x in g.initializer}
        | {x.name for x in g.input}
    )
    used = {i for n in g.node for i in n.input if i}
    for n in g.node:
        for a in n.attribute:
            if a.HasField("g"):
                used |= captures(a.g)
    return used - made


def reorder(g):
    pending = list(g.node)
    produced = {o for n in pending for o in n.output}
    known = {v.name for v in g.input} | {v.name for v in g.initializer}
    ordered = []
    while pending:
        remaining = []
        for n in pending:
            deps = {x for x in n.input if x}
            for a in n.attribute:
                if a.HasField("g"):
                    deps |= captures(a.g)
            if not ((deps & produced) - known):
                ordered.append(n)
                known.update(n.output)
            else:
                remaining.append(n)
        if len(remaining) == len(pending):
            raise RuntimeError(
                "Cannot topologically sort " + str([n.name for n in remaining[:5]])
            )
        pending = remaining
    del g.node[:]
    g.node.extend(ordered)
    for n in g.node:
        for a in n.attribute:
            if a.HasField("g"):
                reorder(a.g)


def promote(g):
    for i, t in enumerate(g.initializer):
        if t.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(t).astype(np.float32)
            g.initializer[i].CopyFrom(numpy_helper.from_array(arr, t.name))
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for n in g.node:
        for a in n.attribute:
            if a.HasField("g"):
                promote(a.g)
            if a.HasField("t") and a.t.data_type == onnx.TensorProto.FLOAT16:
                a.t.CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(a.t).astype(np.float32), a.t.name
                    )
                )
            if (
                n.op_type == "Cast"
                and a.name == "to"
                and a.i == onnx.TensorProto.FLOAT16
            ):
                a.i = onnx.TensorProto.FLOAT
