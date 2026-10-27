"""Exercise standalone static quantization without an MTC SDK or device."""

import numpy as np
import pytest

from edge_model_bench.converter.static_onnx import OnnxStaticQuantConfig, OnnxStaticQuantizer

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("method,chunk", [("MinMax", None), ("Entropy", 1)])
def test_static_quantized_model_runs_on_cpu(tmp_path, method, chunk):
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "linear",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(np.array([[0.5, -0.5], [1, 2]], dtype=np.float32), "weight")],
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8),
        model_path,
    )
    paths = []
    for index in range(4):
        path = tmp_path / f"calibration_{index}.npy"
        np.save(path, np.array([[0.25, 0.25 * (index + 1)]], dtype=np.float32))
        paths.append(str(path))
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("\n".join(paths))
    target = tmp_path / "quantized.onnx"
    OnnxStaticQuantizer(
        OnnxStaticQuantConfig(
            input_model=model_path,
            output_model=target,
            calibration_list=calibration,
            calibrate_method=method,
            calib_chunk_size=chunk,
            use_external_data_format=False,
        )
    )()
    quantized = onnx.load(target)
    assert any(node.op_type == "QuantizeLinear" for node in quantized.graph.node)
    output = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(
        None, {"input": np.array([[0.25, 0.5]], dtype=np.float32)}
    )[0]
    np.testing.assert_allclose(output, [[0.625, 0.875]], atol=0.05)
