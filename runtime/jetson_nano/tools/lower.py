# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
import collections
import hashlib
import json
import math
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from graph_utils import reorder


def run(root):
    m = onnx.load(str(root / "decoder_static_folded.onnx"))
    shapes = json.loads((root / "static_shapes.json").read_text())
    initial = {t.name: t for t in m.graph.initializer}
    byout = {o: n for n in m.graph.node for o in n.output}
    new = []
    extra = []
    counts = collections.Counter()
    index_cache = {}

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def resolve(name):
        while name in byout and (
            byout[name].op_type == "Identity"
            or (
                byout[name].op_type == "Cast"
                and attrs(byout[name]).get("to") == onnx.TensorProto.FLOAT
            )
        ):
            name = byout[name].input[0]
        return name

    def array(name):
        return numpy_helper.to_array(initial[name])

    def constant(name, value):
        t = numpy_helper.from_array(np.asarray(value), name)
        extra.append(t)
        initial[name] = t
        return name

    for n in m.graph.node:
        a = attrs(n)
        if n.op_type == "Cast" and a.get("to") == onnx.TensorProto.FLOAT:
            new.append(
                helper.make_node(
                    "Identity",
                    [n.input[0]],
                    list(n.output),
                    name=n.name + "__float_identity",
                )
            )
            counts["float_cast_removed"] += 1
            continue
        if len(n.output) == 1 and n.output[0] in shapes and 0 in shapes[n.output[0]]:
            constant(n.output[0], np.empty(shapes[n.output[0]], dtype=np.float32))
            counts["empty_tensor_folded"] += 1
            continue
        if n.op_type == "Concat":
            inputs = [x for x in n.input if x not in shapes or 0 not in shapes[x]]
            if len(inputs) == 1:
                new.append(
                    helper.make_node(
                        "Identity", inputs, list(n.output), name=n.name + "__no_empty"
                    )
                )
                counts["empty_concat_removed"] += 1
                continue
            if not inputs:
                raise RuntimeError("Unexpected all-empty concat " + n.name)
            new.append(
                helper.make_node(
                    "Concat", inputs, list(n.output), axis=a["axis"], name=n.name
                )
            )
            continue
        if n.op_type == "Where":
            eq = byout[resolve(n.input[0])]
            log = byout[resolve(n.input[2])]
            plus = byout[resolve(log.input[0])]
            exp = next(
                byout[resolve(x)]
                for x in plus.input
                if resolve(x) in byout and byout[resolve(x)].op_type == "Exp"
            )
            if (
                eq.op_type != "Equal"
                or log.op_type != "Log"
                or plus.op_type != "Add"
                or resolve(exp.input[0]) != resolve(n.input[1])
            ):
                raise RuntimeError("Unrecognized overflow guard " + n.name)
            other = next(
                resolve(x) for x in eq.input if resolve(x) != resolve(n.input[2])
            )
            if not np.isposinf(array(other)).all():
                raise RuntimeError("Guard constant is not +inf")
            new.append(
                helper.make_node(
                    "Softplus",
                    [n.input[1]],
                    list(n.output),
                    name=n.name + "__stable_softplus",
                )
            )
            counts["stable_softplus"] += 1
            continue
        if n.op_type == "GatherElements":
            data_shape = tuple(shapes[n.input[0]])
            indices = array(n.input[1])
            axis = a.get("axis", 0) % len(data_shape)
            key = (data_shape, axis, hashlib.sha256(indices.tobytes()).hexdigest())
            if key not in index_cache:
                offsets = np.arange(np.prod(data_shape), dtype=np.int32).reshape(
                    data_shape
                )
                flat = np.take_along_axis(offsets, indices, axis=axis).reshape(-1)
                idx = constant("__trt6_flat_indices_" + str(len(index_cache)), flat)
                index_cache[key] = idx
                del offsets, flat
            flatten = constant(n.name + "__flat_shape", np.array([-1], dtype=np.int64))
            output_shape = constant(
                n.name + "__output_shape", np.array(indices.shape, dtype=np.int64)
            )
            data_flat = n.output[0] + "__data_flat"
            picked = n.output[0] + "__picked"
            new.extend(
                [
                    helper.make_node("Reshape", [n.input[0], flatten], [data_flat]),
                    helper.make_node(
                        "Gather", [data_flat, index_cache[key]], [picked], axis=0
                    ),
                    helper.make_node("Reshape", [picked, output_shape], list(n.output)),
                ]
            )
            counts["gather_lowered"] += 1
            continue
        if n.op_type in ("Expand", "Tile"):
            s0 = tuple(shapes[n.input[0]])
            s1 = tuple(shapes[n.output[0]])
            current = n.input[0]
            if len(s0) < len(s1):
                s0 = (1,) * (len(s1) - len(s0)) + s0
                target = constant(
                    n.name + "__padded_shape", np.array(s0, dtype=np.int64)
                )
                expanded = n.output[0] + "__padded"
                new.append(helper.make_node("Reshape", [current, target], [expanded]))
                current = expanded
            for axis, (before, after) in enumerate(zip(s0, s1)):
                if before == after:
                    continue
                repeat = after // before
                if (
                    (n.op_type == "Expand" and before != 1)
                    or after % before
                    or repeat > 16
                ):
                    raise RuntimeError(
                        "Unsupported broadcast " + n.name + " " + str((s0, s1))
                    )
                expanded = n.output[0] + "__concat_" + str(axis)
                new.append(
                    helper.make_node(
                        "Concat", [current] * repeat, [expanded], axis=axis
                    )
                )
                current = expanded
            new.append(
                helper.make_node(
                    "Identity", [current], list(n.output), name=n.name + "__identity"
                )
            )
            counts["broadcast_lowered"] += 1
            continue
        if n.op_type == "Unsqueeze":
            shape = constant(
                n.name + "__shape", np.array(shapes[n.output[0]], dtype=np.int64)
            )
            new.append(
                helper.make_node(
                    "Reshape",
                    [n.input[0], shape],
                    list(n.output),
                    name=n.name + "__reshape",
                )
            )
            counts["unsqueeze_reshape"] += 1
            continue
        if n.op_type == "ReduceSum" and len(n.input) > 1:
            axes = array(n.input[1]).reshape(-1).tolist()
            new.append(
                helper.make_node(
                    "ReduceSum",
                    [n.input[0]],
                    list(n.output),
                    axes=axes,
                    keepdims=a.get("keepdims", 1),
                    name=n.name,
                )
            )
            counts["reduce_sum_legacy"] += 1
            continue
        new.append(n)
    needed = {v.name for v in m.graph.output}
    kept = []
    for n in reversed(new):
        if any(o in needed for o in n.output):
            kept.append(n)
            needed.update(n.input)
    kept.reverse()
    del m.graph.node[:]
    m.graph.node.extend(kept)
    weights = [t for t in list(m.graph.initializer) + extra if t.name in needed]
    del m.graph.initializer[:]
    m.graph.initializer.extend(weights)
    for imp in m.opset_import:
        if imp.domain == "":
            imp.version = 11
    m.ir_version = 3
    for t in m.graph.initializer:
        m.graph.input.append(
            helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
        )
    reorder(m.graph)
    onnx.checker.check_model(m)
    onnx.save(m, str(root / "decoder_trt6.onnx"))
    print(
        json.dumps(
            {
                "event": "legacy_lowered",
                "changes": dict(counts),
                "shared_index_tables": len(index_cache),
                "nodes": len(m.graph.node),
                "ops": dict(collections.Counter(n.op_type for n in m.graph.node)),
            }
        ),
        flush=True,
    )
