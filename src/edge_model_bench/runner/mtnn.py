"""Moore Threads M1000 NPU ``.mtnn`` runtime runner."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


_DTYPES = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.uint8,
    4: np.int16,
    5: np.int32,
    6: np.int64,
    7: np.bool_,
}


class MtnnRunner:
    """Load and run one MTNN model using the optional ``mtnn_api`` package.

    Input tensors are supplied as float32, as in the rest of this repository.
    The runner adapts them to the model interface before copying to NPU memory:
    FP16 models receive a float32-to-float16 cast, while quantized inputs use
    MTNN's ``mtnn_get_tensor_from_array`` helper.  Outputs are dequantized (if
    needed) and promoted to float32 for numerically stable task postprocessing.
    With ``output_mapping=False`` (the default), outputs are returned as a
    list, matching :class:`runner.onnx.OnnxRunner`.
    """

    def __init__(
        self,
        model_path: str | Path,
        initialize: bool = False,
        npu_mode: int = 0,
        n_npu_device: int = 1,
        init_flag: int = 0,
        device_index: int | None = None,
        frequency_level: int | None = None,
        runtime_env: Mapping[str, str | int] | None = None,
        output_mapping: bool = False,
        reverse_output_dims: bool = True,
        dequantize_outputs: bool = True,
        cast_outputs_to_float32: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.npu_mode = npu_mode
        self.n_npu_device = n_npu_device
        self.init_flag = init_flag
        self.device_index = device_index
        self.frequency_level = frequency_level
        self.runtime_env = dict(runtime_env or {})
        self.output_mapping = output_mapping
        self.reverse_output_dims = reverse_output_dims
        self.dequantize_outputs = dequantize_outputs
        self.cast_outputs_to_float32 = cast_outputs_to_float32
        self.latencies: list[float] = []

        self._api: Any = None
        self._ffi: Any = None
        self._model: Any = None
        self._input_attrs: list[Any] = []
        self._output_attrs: list[Any] = []
        self._input_memory: Any = None
        self._output_memory: Any = None
        self.input_names: tuple[str, ...] = ()
        self.output_names: tuple[str, ...] = ()
        if initialize:
            self.initialize()

    def _tensor_name(self, attr: Any, index: int, prefix: str) -> str:
        raw_name = attr.name
        if isinstance(raw_name, bytes):
            name = raw_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        else:
            # CFFI exposes ``char name[256]`` as cdata; its ``str`` does not
            # contain the actual tensor name.  Fall back for lightweight mock
            # APIs that expose it as a normal string.
            try:
                name = self._ffi.string(raw_name).decode("utf-8", errors="replace")
            except (AttributeError, TypeError):
                name = str(raw_name).split("\0", 1)[0]
        return name or f"{prefix}_{index}"

    def _command(self, name: str, fallback: int) -> int:
        return int(getattr(self._api.mtnn_lib, name, fallback))

    def initialize(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MTNN 模型不存在: {self.model_path}")
        try:
            import mtnn_api
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MtnnRunner 只能在安装 mtnn_api 和 M1000 NPU Runtime 的环境中运行"
            ) from exc

        for name, value in self.runtime_env.items():
            os.environ[str(name)] = str(value)

        self._api = mtnn_api
        self._ffi = mtnn_api.ffi
        self._model = mtnn_api.MTNN()
        work_mode = self._ffi.new("mtnn_work_mode_t*")
        work_mode.init_flag = self.init_flag
        work_mode.n_npu_device = self.n_npu_device
        work_mode.e_npu_mode = self.npu_mode
        self._model.init(str(self.model_path), work_mode)

        if self.device_index is not None:
            value = self._ffi.new("int*", self.device_index)
            self._model.set_info(
                self._command("MTNN_SET_NPU_INDEX", 7), value, self._ffi.sizeof("int")
            )
        if self.frequency_level is not None:
            value = self._ffi.new("int*", self.frequency_level)
            self._model.set_info(
                self._command("MTNN_SET_NPU_FREQUENCY", 8),
                value,
                self._ffi.sizeof("int"),
            )

        io_count = self._ffi.new("mtnn_input_output_num*")
        self._model.get_info(
            self._command("MTNN_GET_IN_OUT_NUM", 0),
            io_count,
            self._ffi.sizeof("mtnn_input_output_num"),
        )
        self._input_memory = self._model.get_inputs(io_count.n_input)
        self._output_memory = self._model.get_outputs(io_count.n_output)
        self._input_attrs = self._get_attributes(
            io_count.n_input, self._command("MTNN_GET_INPUT_ATTR", 1)
        )
        self._output_attrs = self._get_attributes(
            io_count.n_output, self._command("MTNN_GET_OUTPUT_ATTR", 2)
        )
        self.input_names = tuple(
            self._tensor_name(attr, index, "input")
            for index, attr in enumerate(self._input_attrs)
        )
        self.output_names = tuple(
            self._tensor_name(attr, index, "output")
            for index, attr in enumerate(self._output_attrs)
        )

    def _get_attributes(self, count: int, command: int) -> list[Any]:
        attributes = []
        for index in range(count):
            attr = self._ffi.new("mtnn_tensor_attr*")
            attr.index = index
            self._model.get_info(command, attr, self._ffi.sizeof("mtnn_tensor_attr"))
            attributes.append(attr)
        return attributes

    def _ordered_inputs(self, image: np.ndarray | Mapping[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(image, Mapping):
            expected = set(self.input_names)
            actual = set(image)
            if actual != expected:
                raise ValueError(
                    f"MTNN 输入名不匹配，缺少={sorted(expected - actual)}，"
                    f"额外={sorted(actual - expected)}"
                )
            return [np.asarray(image[name]) for name in self.input_names]
        if len(self.input_names) != 1:
            raise ValueError(f"多输入 MTNN 需要 dict 输入: {list(self.input_names)}")
        return [np.asarray(image)]

    def _set_input(self, index: int, value: np.ndarray) -> None:
        attr = self._input_attrs[index]
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.size != attr.n_elems:
            raise ValueError(
                f"输入 {self.input_names[index]!r} 元素数不匹配: 得到 {array.size}，"
                f"模型要求 {attr.n_elems}"
            )
        # When MTC force_fp32input_list is false, the exported model commonly
        # exposes FP16 I/O.  Preserve benchmark preprocessing in FP32, then
        # explicitly cast at this boundary.  The MTNN helper is reserved for
        # genuinely quantized input tensors, where it applies model metadata.
        dtype = _DTYPES.get(int(attr.type))
        if dtype in {np.float32, np.float16}:
            converted = np.ascontiguousarray(array, dtype=dtype)
            if converted.nbytes != attr.size:
                raise ValueError(
                    f"输入 {self.input_names[index]!r} 字节数不匹配: "
                    f"得到 {converted.nbytes}，模型要求 {attr.size}"
                )
            self._ffi.memmove(
                self._input_memory[index].logical_addr, converted.tobytes(), attr.size
            )
            return

        # Keep CFFI-owned memory alive until mtnn_get_tensor_from_array returns.
        source = self._ffi.new("float[]", array.reshape(-1).tolist())
        tensor = self._api.mtnn_lib.mtnn_get_tensor_from_array(
            self._model.mgr[0], index, source
        )
        if tensor == self._ffi.NULL:
            raise RuntimeError(f"mtnn_get_tensor_from_array 失败: input={index}")
        self._ffi.memmove(self._input_memory[index].logical_addr, tensor, attr.size)

    def _read_output(self, index: int) -> np.ndarray:
        attr = self._output_attrs[index]
        raw = self._ffi.buffer(
            self._output_memory[index].logical_addr, self._output_memory[index].size
        )[:]
        dtype = _DTYPES.get(int(attr.type))
        if dtype is None:
            raise RuntimeError(f"不支持的 MTNN 输出类型: {attr.type}")
        output = np.frombuffer(raw, dtype=dtype).copy()
        if self.dequantize_outputs:
            if attr.qnt_type == 1:  # dynamic fixed point
                output = output.astype(np.float32) * (2.0 ** -attr.fl)
            elif attr.qnt_type == 2:  # affine asymmetric
                zero_point = int(attr.zp)
                if np.issubdtype(dtype, np.signedinteger) and zero_point >= 2**31:
                    zero_point -= 2**32
                output = (output.astype(np.float32) - zero_point) * attr.scale
        # MTC commonly emits FP16 inputs/outputs when the force-FP32 lists are
        # disabled.  Promote a non-quantized FP16 output before the classifier
        # computes softmax, so that the task postprocessor stays in FP32.
        if self.cast_outputs_to_float32 and output.dtype != np.float32:
            output = output.astype(np.float32)
        dims = tuple(int(attr.dims[i]) for i in range(attr.n_dims))
        if self.reverse_output_dims:
            dims = dims[::-1]
        return output.reshape(dims)

    def __call__(self, image: np.ndarray | Mapping[str, np.ndarray]) -> Any:
        if self._model is None:
            self.initialize()
        for index, value in enumerate(self._ordered_inputs(image)):
            self._set_input(index, value)

        start = perf_counter()
        self._model.set_inputs(len(self._input_attrs), self._ffi.NULL)
        self._model.inference(self._ffi.NULL)
        self.latencies.append(perf_counter() - start)

        outputs = [self._read_output(index) for index in range(len(self._output_attrs))]
        if self.output_mapping:
            return dict(zip(self.output_names, outputs))
        return outputs

    def close(self) -> None:
        if self._model is None:
            return
        try:
            self._model.destroy()
        finally:
            self._model = None
            self._input_memory = None
            self._output_memory = None
