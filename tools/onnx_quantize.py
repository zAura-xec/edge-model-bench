#!/usr/bin/env python3
"""Standalone ONNX static quantization tool with chunked Entropy calibration.

Example (resnet18)::

    python tools/onnx_quantize.py \\
        --model /path/to/resnet18.onnx \\
        --calib-list work/resnet18_calib/calib_list.txt \\
        --output work/resnet18_int8/resnet18_static_quantized.onnx \\
        --method Entropy \\
        --chunk-size 8 \\
        --max-samples 32

To build a calibration list from a directory of ``.npy`` tensors::

    python tools/onnx_quantize.py --make-calib-list work/npy --calib-list work/npy/calib_list.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as ``python tools/onnx_quantize.py`` without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _make_calib_list(npy_dir: Path, output: Path, pattern: str = "*.npy") -> Path:
    paths = sorted(npy_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"目录中没有匹配 {pattern!r} 的文件: {npy_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(str(path.resolve()) for path in paths) + "\n",
        encoding="utf-8",
    )
    logging.info("写入校准清单 %s (%d samples)", output, len(paths))
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX Runtime 静态量化工具（支持分块 Entropy collect_data）",
    )
    parser.add_argument("--model", type=Path, help="输入 FP32 ONNX 模型路径")
    parser.add_argument("--output", type=Path, help="输出量化 ONNX 路径")
    parser.add_argument(
        "--calib-list",
        type=Path,
        help="校准清单（每行一个 .npy 绝对/相对路径）",
    )
    parser.add_argument(
        "--make-calib-list",
        type=Path,
        metavar="NPY_DIR",
        help="从目录生成校准清单后退出（或配合量化一起使用）",
    )
    parser.add_argument("--method", default="Entropy", help="校准方法: Entropy/MinMax/Percentile")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="每次 collect_data() 处理的样本数（降低峰值内存，默认 8）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多使用的校准样本数（默认用满清单）",
    )
    parser.add_argument("--input-name", default=None, help="模型输入名（单输入可省略）")
    parser.add_argument("--activation-type", default="QInt8")
    parser.add_argument("--weight-type", default="QInt8")
    parser.add_argument("--quant-format", default="QDQ")
    parser.add_argument(
        "--op-types",
        default="Conv,MatMul,Add,Mul,Sigmoid,GlobalAveragePool",
        help="逗号分隔的待量化算子类型；空字符串表示默认全部",
    )
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument(
        "--no-auto-batch",
        action="store_true",
        help="禁止把 CHW .npy 自动扩展为 NCHW",
    )
    parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="不使用 external data 格式保存模型",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.make_calib_list is not None:
        if args.calib_list is None:
            args.calib_list = args.make_calib_list / "calib_list.txt"
        _make_calib_list(args.make_calib_list, args.calib_list)
        if args.model is None:
            return 0

    if args.model is None or args.output is None or args.calib_list is None:
        logging.error("量化需要 --model / --output / --calib-list")
        return 2

    from edge_model_bench.converter.static_onnx import OnnxStaticQuantConfig, OnnxStaticQuantizer

    op_types = (
        [item.strip() for item in args.op_types.split(",") if item.strip()]
        if args.op_types is not None
        else None
    )
    if op_types == []:
        op_types = None

    config = OnnxStaticQuantConfig(
        input_model=args.model,
        output_model=args.output,
        calibration_list=args.calib_list,
        max_calibration_samples=args.max_samples,
        input_name=args.input_name,
        calibrate_method=args.method,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        op_types_to_quantize=op_types,
        calib_chunk_size=args.chunk_size,
        auto_batch_dim=not args.no_auto_batch,
        use_external_data_format=not args.no_external_data,
    )

    started = time.perf_counter()
    output = OnnxStaticQuantizer(config)()
    elapsed = time.perf_counter() - started
    logging.info("量化完成: %s (%.1fs)", output, elapsed)

    # Quick smoke: load with ORT and run a random / first-calib tensor.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        # Prefer first calibration sample when shapes match.
        sample = None
        first = Path(
            args.calib_list.read_text(encoding="utf-8").splitlines()[0].strip()
        )
        if first.is_file():
            array = np.ascontiguousarray(np.load(first), dtype=np.float32)
            while array.ndim < len(shape):
                array = array[None]
            if list(array.shape) == shape:
                sample = array
        if sample is None:
            sample = np.random.randn(*shape).astype(np.float32)
        outs = sess.run(None, {inp.name: sample})
        logging.info(
            "ORT smoke OK: input=%s%s outputs=%s",
            inp.name,
            tuple(sample.shape),
            [tuple(o.shape) for o in outs],
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        logging.warning("ORT smoke 失败: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
