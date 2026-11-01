# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
"""Host-only regressions for legacy ONNX semantic pitfalls."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from prepare_decoder import fix_softmax
from fold import run as fold


def evaluate(model, inputs):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    ).run(None, inputs)[0]


class GraphTests(unittest.TestCase):
    def test_softmax_preserves_each_query_row(self):
        shape = [1, 4, 1, 8, 8]
        graph = helper.make_graph(
            [helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1)],
            "attention",
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, shape)],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=7
        )
        self.assertEqual(fix_softmax(model, {"scores": shape[1:]}), 1)
        values = np.random.RandomState(5).normal(0, 8, shape).astype(np.float32)
        actual = evaluate(model, {"scores": values})
        expected = np.exp(values - values.max(axis=-1, keepdims=True))
        expected /= expected.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=1e-6)

    def test_scatter_folding_uses_coordinate_tuples(self):
        values = np.zeros((3, 5), np.float32)
        indices = np.array([[0, 2], [2, 4]], np.int64)
        updates = np.array([1.0, 2.0], np.float32)
        graph = helper.make_graph(
            [helper.make_node("ScatterND", ["data", "indices", "updates"], ["out"])],
            "positions",
            [],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 5])],
            [
                numpy_helper.from_array(array, name)
                for array, name in [
                    (values, "data"),
                    (indices, "indices"),
                    (updates, "updates"),
                ]
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx.save(model, str(root / "decoder_fixed_fp32.onnx"))
            fold(root)
            actual = evaluate(onnx.load(str(root / "decoder_static_folded.onnx")), {})
        expected = values.copy()
        expected[0, 2] = 1.0
        expected[2, 4] = 2.0
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
