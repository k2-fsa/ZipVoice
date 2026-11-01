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
    m = onnx.load(str(root / "decoder_trt6.onnx"))
    shapes = json.loads((root / "static_shapes.json").read_text())
    initial = {t.name: t for t in m.graph.initializer}
    real_inputs = [v for v in m.graph.input if v.name not in initial]
    new = []
    extra = []
    counts = collections.Counter()
    padding_cache = {}

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def array(name):
        return numpy_helper.to_array(initial[name])

    def const(name, value):
        t = numpy_helper.from_array(np.asarray(value), name)
        extra.append(t)
        initial[name] = t
        return name

    def padded(name, rank):
        key = (name, rank)
        if key in padding_cache:
            return padding_cache[key]
        value = array(name)
        if value.ndim >= rank:
            return name
        result = const(
            name + "__rank_" + str(rank),
            value.reshape((1,) * (rank - value.ndim) + value.shape),
        )
        padding_cache[key] = result
        return result

    def copy(n):
        c = onnx.NodeProto()
        c.CopyFrom(n)
        return c

    for n in m.graph.node:
        a = attrs(n)
        c = copy(n)
        if n.op_type == "MatMul":
            if (
                n.input[1] in initial
                and len(initial[n.input[1]].dims) == 2
                and n.input[0] in shapes
                and len(shapes[n.input[0]]) > 2
            ):
                si = shapes[n.input[0]]
                so = shapes[n.output[0]]
                pre_shape = const(
                    n.name + "__dense_input_shape",
                    np.array([1, int(np.prod(si[:-1])), si[-1]], dtype=np.int64),
                )
                post_shape = const(
                    n.name + "__dense_output_shape", np.array([1] + so, dtype=np.int64)
                )
                pre = n.output[0] + "__dense_input"
                post = n.output[0] + "__dense_output"
                new.extend(
                    [
                        helper.make_node("Reshape", [n.input[0], pre_shape], [pre]),
                        helper.make_node(
                            "MatMul",
                            [pre, n.input[1]],
                            [post],
                            name=n.name + "__dense_gemm",
                        ),
                        helper.make_node("Reshape", [post, post_shape], list(n.output)),
                    ]
                )
                counts["dense_gemm_flattened"] += 1
                continue
            dynamic = [x for x in n.input if x not in initial]
            rank = max(len(shapes[x]) for x in dynamic)
            for i, name in enumerate(n.input):
                if name in initial:
                    c.input[i] = padded(name, rank)
            counts["matmul_broadcast"] += 1
        elif n.op_type in ("Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"):
            rank = len(shapes[n.output[0]]) + 1
            for i, name in enumerate(n.input):
                if name in initial:
                    c.input[i] = padded(name, rank)
            counts["elementwise_broadcast"] += 1
        elif n.op_type == "Transpose":
            perm = a.get("perm", list(reversed(range(len(shapes[n.input[0]])))))
            del c.attribute[:]
            c.attribute.append(
                helper.make_attribute("perm", [0] + [i + 1 for i in perm])
            )
            counts["transpose"] += 1
        elif n.op_type == "Reshape":
            target = shapes.get(n.output[0])
            if target is None:
                target = array(n.input[1]).reshape(-1).tolist()
                source_shape = shapes[n.input[0]]
                target = [
                    source_shape[i] if d == 0 else int(d) for i, d in enumerate(target)
                ]
                if -1 in target:
                    target[target.index(-1)] = int(
                        np.prod(source_shape) // (-np.prod(target))
                    )
            c.input[1] = const(
                n.name + "__batch_shape_" + n.output[0],
                np.array([1] + list(target), dtype=np.int64),
            )
            counts["reshape"] += 1
        elif n.op_type == "Slice":
            axes = (
                array(n.input[3]).reshape(-1)
                if len(n.input) > 3 and n.input[3]
                else np.arange(array(n.input[1]).size)
            )
            starts = array(n.input[1]).reshape(-1)
            ends = array(n.input[2]).reshape(-1)
            steps = (
                array(n.input[4]).reshape(-1)
                if len(n.input) > 4 and n.input[4]
                else [1] * len(starts)
            )
            source_shape = shapes[n.input[0]]
            current = n.input[0]
            for j, (start, end, axis, step) in enumerate(
                zip(starts, ends, axes, steps)
            ):
                axis = int(axis) % len(source_shape)
                begin, stop, stride = slice(int(start), int(end), int(step)).indices(
                    source_shape[axis]
                )
                index = const(
                    n.name + "__slice_indices_" + str(j),
                    np.arange(begin, stop, stride, dtype=np.int32),
                )
                output = (
                    n.output[0]
                    if j == len(starts) - 1
                    else n.output[0] + "__axis_" + str(j)
                )
                new.append(
                    helper.make_node(
                        "Gather",
                        [current, index],
                        [output],
                        axis=axis + 1,
                        name=n.name + "__gather_" + str(j),
                    )
                )
                current = output
            counts["slice_gather"] += 1
            continue
        elif n.op_type in ("Concat", "Gather", "Softmax"):
            axis = a.get("axis", -1 if n.op_type == "Softmax" else 0)
            if axis >= 0:
                axis += 1
            for at in c.attribute:
                if at.name == "axis":
                    at.i = axis
                    break
            else:
                c.attribute.append(helper.make_attribute("axis", axis))
            counts["axis"] += 1
        elif n.op_type in ("ReduceMean", "ReduceSum"):
            axes = a.get("axes", list(range(len(shapes[n.input[0]]))))
            axes = [x + 1 if x >= 0 else x for x in axes]
            found = False
            for at in c.attribute:
                if at.name == "axes":
                    del at.ints[:]
                    at.ints.extend(axes)
                    found = True
            if not found:
                c.attribute.append(helper.make_attribute("axes", axes))
            counts["reduction"] += 1
        elif n.op_type == "Gemm":
            weights = array(n.input[1])
            if a.get("transB", 0):
                weights = weights.T
            if a.get("transA", 0):
                raise RuntimeError("Unexpected transA")
            weights = weights * a.get("alpha", 1.0)
            rank = len(shapes[n.input[0]])
            weight_name = const(
                n.name + "__weight",
                weights.reshape((1,) * max(0, rank - weights.ndim) + weights.shape),
            )
            mm_out = n.output[0] + "__matmul"
            new.append(
                helper.make_node(
                    "MatMul",
                    [n.input[0], weight_name],
                    [mm_out],
                    name=n.name + "__matmul",
                )
            )
            if len(n.input) > 2:
                if n.input[2] in initial:
                    value = array(n.input[2]) * a.get("beta", 1.0)
                    rank = len(shapes[n.output[0]]) + 1
                    bias = const(
                        n.name + "__bias",
                        value.reshape((1,) * max(0, rank - value.ndim) + value.shape),
                    )
                else:
                    bias = n.input[2]
                    if a.get("beta", 1.0) != 1.0:
                        scale = const(
                            n.name + "__beta", np.array(a["beta"], dtype=np.float32)
                        )
                        scaled = bias + "__scaled"
                        new.append(helper.make_node("Mul", [bias, scale], [scaled]))
                        bias = scaled
                new.append(
                    helper.make_node(
                        "Add", [mm_out, bias], list(n.output), name=n.name + "__bias"
                    )
                )
            else:
                new.append(helper.make_node("Identity", [mm_out], list(n.output)))
            counts["gemm"] += 1
            continue
        elif n.op_type == "Conv":
            si = shapes[n.input[0]]
            so = shapes[n.output[0]]
            if len(si) != 3 or si[0] != 1:
                raise RuntimeError("Unexpected Conv dimensions " + str(si))
            pre_shape = const(
                n.name + "__conv_input_shape",
                np.array([1, si[1], 1, si[2]], dtype=np.int64),
            )
            post_shape = const(
                n.name + "__conv_output_shape", np.array([1] + so, dtype=np.int64)
            )
            pre = n.input[0] + "__conv2d_" + n.name
            post = n.output[0] + "__conv2d"
            new.append(helper.make_node("Reshape", [n.input[0], pre_shape], [pre]))
            c.input[0] = pre
            w = array(n.input[1])
            c.input[1] = const(
                n.name + "__conv_weight",
                w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]),
            )
            old_output = list(c.output)
            c.output[0] = post
            ca = dict(a)
            ca["kernel_shape"] = [1, w.shape[2]]
            ca["strides"] = [1] + a.get("strides", [1])
            ca["dilations"] = [1] + a.get("dilations", [1])
            pads = a.get("pads", [0, 0])
            ca["pads"] = [0, pads[0], 0, pads[1]]
            del c.attribute[:]
            for name, value in ca.items():
                c.attribute.append(helper.make_attribute(name, value))
            new.append(c)
            new.append(helper.make_node("Reshape", [post, post_shape], old_output))
            counts["conv2d"] += 1
            continue
        new.append(c)
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
    del m.graph.input[:]
    for v in real_inputs:
        dimensions = [d.dim_value for d in v.type.tensor_type.shape.dim]
        m.graph.input.append(
            helper.make_tensor_value_info(
                v.name, v.type.tensor_type.elem_type, [1] + dimensions
            )
        )
    for t in weights:
        m.graph.input.append(
            helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
        )
    outputs = []
    for v in m.graph.output:
        dims = shapes.get(v.name)
        if dims is None:
            raise RuntimeError("Unknown output shape " + v.name)
        outputs.append(
            helper.make_tensor_value_info(
                v.name, v.type.tensor_type.elem_type, [1] + dims
            )
        )
    del m.graph.output[:]
    m.graph.output.extend(outputs)
    del m.graph.value_info[:]
    reorder(m.graph)
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, str(root / "decoder_trt6_implicit.onnx"))
    print(
        json.dumps(
            {
                "event": "implicit_batch_lowered",
                "changes": dict(counts),
                "nodes": len(m.graph.node),
                "outputs": [
                    (v.name, [d.dim_value for d in v.type.tensor_type.shape.dim])
                    for v in m.graph.output
                ],
            }
        ),
        flush=True,
    )
