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
    m = onnx.load(str(root / "decoder_fixed_fp32.onnx"))
    reorder(m.graph)
    constants = {t.name: t for t in m.graph.initializer}
    arrays = {}
    shapes = {t.name: tuple(t.dims) for t in m.graph.initializer}
    for v in m.graph.input:
        dims = v.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") for d in dims):
            shapes[v.name] = tuple(d.dim_value for d in dims)
    new_initializers = []

    def arr(name):
        if name in arrays:
            return arrays[name]
        a = numpy_helper.to_array(constants[name])
        if a.size < 200000:
            arrays[name] = a
        return a

    def add_const(name, a):
        a = np.asarray(a)
        t = numpy_helper.from_array(a, name)
        constants[name] = t
        new_initializers.append(t)
        shapes[name] = a.shape
        if a.size < 200000:
            arrays[name] = a

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def shape_node(n):
        op = n.op_type
        a = attrs(n)
        ins = list(n.input)
        ss = [shapes.get(x) for x in ins]
        s = ss[0] if ss else None
        try:
            if op in (
                "Identity",
                "Cast",
                "Exp",
                "Log",
                "Tanh",
                "Sigmoid",
                "Sin",
                "Cos",
                "Abs",
                "Neg",
                "Softmax",
                "Sqrt",
            ):
                return s
            if op in (
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Pow",
                "Max",
                "Min",
                "Equal",
                "Where",
                "GreaterOrEqual",
                "Less",
                "Greater",
            ):
                if all(x is not None for x in ss):
                    return np.broadcast_shapes(*ss)
            if op == "MatMul" and ss[0] is not None and ss[1] is not None:
                p, q = ss[:2]
                if len(p) == 1 and len(q) == 1:
                    return ()
                if len(p) == 1:
                    return q[:-2] + (q[-1],)
                if len(q) == 1:
                    return p[:-1]
                return np.broadcast_shapes(p[:-2], q[:-2]) + (p[-2], q[-1])
            if op == "Gemm" and ss[0] is not None and ss[1] is not None:
                return (
                    ss[0][1] if a.get("transA", 0) else ss[0][0],
                    ss[1][0] if a.get("transB", 0) else ss[1][1],
                )
            if op == "Shape" and s is not None:
                return (len(s),)
            if op == "Transpose" and s is not None:
                return tuple(s[i] for i in a.get("perm", list(reversed(range(len(s))))))
            if op == "Unsqueeze" and s is not None:
                axes = arr(ins[1]).tolist() if len(ins) > 1 else a["axes"]
                out = list(s)
                rank = len(s) + len(axes)
                for axis in sorted(x % rank for x in axes):
                    out.insert(axis, 1)
                return tuple(out)
            if op == "Squeeze" and s is not None:
                axes = (
                    arr(ins[1]).tolist()
                    if len(ins) > 1
                    else a.get("axes", [i for i, d in enumerate(s) if d == 1])
                )
                return tuple(
                    d for i, d in enumerate(s) if i not in {x % len(s) for x in axes}
                )
            if op == "Reshape" and s is not None and ins[1] in constants:
                out = [int(x) for x in arr(ins[1]).reshape(-1)]
                out = [s[i] if x == 0 else x for i, x in enumerate(out)]
                if -1 in out:
                    out[out.index(-1)] = int(np.prod(s) // (-np.prod(out)))
                return tuple(out)
            if op == "Concat" and all(x is not None for x in ss):
                axis = a["axis"] % len(ss[0])
                out = list(ss[0])
                out[axis] = sum(x[axis] for x in ss)
                return tuple(out)
            if op == "Gather" and ss[0] is not None and ss[1] is not None:
                axis = a.get("axis", 0) % len(ss[0])
                return ss[0][:axis] + ss[1] + ss[0][axis + 1 :]
            if op == "GatherElements" and ss[1] is not None:
                return ss[1]
            if (
                op == "Slice"
                and s is not None
                and all(x in constants for x in ins[1:] if x)
            ):
                starts = arr(ins[1]).reshape(-1)
                ends = arr(ins[2]).reshape(-1)
                axes = (
                    arr(ins[3]).reshape(-1)
                    if len(ins) > 3 and ins[3]
                    else range(len(starts))
                )
                steps = (
                    arr(ins[4]).reshape(-1)
                    if len(ins) > 4 and ins[4]
                    else [1] * len(starts)
                )
                out = list(s)
                for start, end, axis, step in zip(starts, ends, axes, steps):
                    axis = int(axis) % len(s)
                    out[axis] = len(
                        range(*slice(int(start), int(end), int(step)).indices(s[axis]))
                    )
                return tuple(out)
            if op in ("ReduceMean", "ReduceSum") and s is not None:
                axes = (
                    arr(ins[1]).reshape(-1).tolist()
                    if len(ins) > 1 and ins[1]
                    else a.get("axes", list(range(len(s))))
                )
                axes = {x % len(s) for x in axes}
                return (
                    tuple(1 if i in axes else d for i, d in enumerate(s))
                    if a.get("keepdims", 1)
                    else tuple(d for i, d in enumerate(s) if i not in axes)
                )
            if op == "Expand" and s is not None and ins[1] in constants:
                return np.broadcast_shapes(s, tuple(arr(ins[1]).reshape(-1)))
            if op == "Tile" and s is not None and ins[1] in constants:
                return tuple(d * int(r) for d, r in zip(s, arr(ins[1]).reshape(-1)))
            if op == "Conv" and s is not None and ss[1] is not None:
                strides = a.get("strides", [1] * (len(s) - 2))
                dil = a.get("dilations", [1] * (len(s) - 2))
                pads = a.get("pads", [0] * (2 * (len(s) - 2)))
                return (
                    s[:1]
                    + (ss[1][0],)
                    + tuple(
                        (
                            s[i + 2]
                            + pads[i]
                            + pads[i + len(strides)]
                            - dil[i] * (ss[1][i + 2] - 1)
                            - 1
                        )
                        // strides[i]
                        + 1
                        for i in range(len(strides))
                    )
                )
        except (KeyError, ValueError, IndexError, ZeroDivisionError):
            return None
        return None

    pending = list(m.graph.node)
    kept = []
    folded = 0
    inlined = 0
    while pending:
        n = pending.pop(0)
        a = attrs(n)
        if n.op_type == "Constant":
            outputs = ReferenceEvaluator(n, opsets={"": 13}).run(None, {})
            for name, value in zip(n.output, outputs):
                add_const(name, value)
            folded += 1
            continue
        if n.op_type == "Shape" and n.input[0] in shapes:
            add_const(n.output[0], np.asarray(shapes[n.input[0]], dtype=np.int64))
            folded += 1
            continue
        if n.op_type == "If" and n.input[0] in constants:
            g = a["then_branch"] if bool(arr(n.input[0])) else a["else_branch"]
            prefix = n.output[0] + "__static__"
            local = {name: prefix + name for child in g.node for name in child.output}
            local.update({t.name: prefix + t.name for t in g.initializer})
            children = []
            for t in g.initializer:
                t2 = onnx.TensorProto()
                t2.CopyFrom(t)
                t2.name = local[t.name]
                constants[t2.name] = t2
                new_initializers.append(t2)
                shapes[t2.name] = tuple(t2.dims)
            for child in g.node:
                c = onnx.NodeProto()
                c.CopyFrom(child)
                for i, name in enumerate(c.input):
                    c.input[i] = local.get(name, name)
                for i, name in enumerate(c.output):
                    c.output[i] = local[name]
                children.append(c)
            for output, go in zip(n.output, g.output):
                children.append(
                    helper.make_node(
                        "Identity", [local.get(go.name, go.name)], [output]
                    )
                )
            pending = children + pending
            inlined += 1
            continue
        if all(x in constants for x in n.input if x) and n.op_type != "If":
            try:
                feed = {x: arr(x) for x in n.input if x}
                if sum(v.nbytes for v in feed.values()) < 64 * 1024 * 1024:
                    if n.op_type == "ScatterND":
                        # ONNX 1.14 indexes an ndarray as advanced rows here; ONNX requires coordinate tuples.
                        output = np.array(feed[n.input[0]], copy=True)
                        indices = feed[n.input[1]]
                        updates = feed[n.input[2]]
                        if a.get("reduction", "none") not in ("none", b"none"):
                            raise RuntimeError("Unexpected ScatterND reduction")
                        for ix in np.ndindex(indices.shape[:-1]):
                            output[tuple(indices[ix])] = updates[ix]
                        outputs = [output]
                    else:
                        outputs = ReferenceEvaluator(n, opsets={"": 13}).run(None, feed)
                    if sum(v.nbytes for v in outputs) < 64 * 1024 * 1024:
                        for name, value in zip(n.output, outputs):
                            add_const(name, value)
                        folded += 1
                        continue
            except Exception as e:
                print(
                    json.dumps(
                        {
                            "constant_eval_failed": n.op_type,
                            "name": n.name,
                            "reason": str(e)[:160],
                        }
                    ),
                    flush=True,
                )
        s = shape_node(n)
        if s is not None and len(n.output) == 1:
            shapes[n.output[0]] = tuple(int(x) for x in s)
        kept.append(n)
    del m.graph.node[:]
    m.graph.node.extend(kept)
    m.graph.initializer.extend(new_initializers)
    used = {x for n in m.graph.node for x in n.input} | {x.name for x in m.graph.output}
    remaining = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(remaining)
    del m.graph.value_info[:]
    reorder(m.graph)
    onnx.checker.check_model(m)
    onnx.save(m, str(root / "decoder_static_folded.onnx"))
    print(
        json.dumps(
            {
                "event": "static_folded",
                "folded": folded,
                "inlined": inlined,
                "nodes": len(m.graph.node),
                "ops": dict(collections.Counter(n.op_type for n in m.graph.node)),
                "shapes_known": len(shapes),
            }
        ),
        flush=True,
    )
    (root / "static_shapes.json").write_text(json.dumps(shapes))
