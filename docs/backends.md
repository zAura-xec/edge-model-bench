# Backend setup and compatibility

The base package is independent of vendor SDKs. Install SDKs only in the environment that performs the relevant conversion or device execution. A successful config composition is not a hardware compatibility test.

| Backend | Host-side conversion | Inference | Config examples |
| --- | --- | --- | --- |
| ONNX Runtime | No conversion for floating ONNX; static quantization helpers available | `onnxruntime`, installed with `[onnx]` | `pipeline/task/cls/onnx/resnet18` |
| QNN / QAIRT | Official QAIRT SDK providing `qairt` and `qti` Python APIs | Compatible QAIRT runtime and target device | `pipeline/task/det/qnn/fp16/yolov8n` |
| Horizon HBR | Official OpenExplorer toolchain with `hb_compile` on PATH; `[onnx]` for model inspection | `hbm_runtime` on a supported RDK S-series board | `pipeline/conv/cls/hbr/fp16/resnet18`, `pipeline/eval/cls/hbr/resnet18_int8` |
| MTNN | Official MTC toolkit with `mtc` on PATH | Vendor MTNN Python binding / shared library | `pipeline/conv/cls/mtnn/fp16/resnet18`, `pipeline/eval/cls/mtnn/resnet18_fp16` |

Consult each vendor's official SDK documentation and distribution terms. These SDKs are not redistributed here. The QNN adapter imports internal SDK API paths, so SDK upgrades may require adapter changes.

## ONNX Runtime

```bash
python -m pip install -e '.[onnx]'
edge-model-bench demo --stage evaluate-onnx --provider CPUExecutionProvider
```

Generate the demo files first. To use CUDA, install the GPU ONNX Runtime distribution appropriate to your CUDA/cuDNN versions in a separate environment, replacing the CPU `onnxruntime` distribution. Do not install both distributions together. Use the [official provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) to select compatible versions. Inspect `onnxruntime.get_available_providers()` before running. The locked `onnx` extra installs the CPU distribution.

Static ONNX quantization utility:

```bash
python tools/onnx_quantize.py --help
python tools/onnx_quantize.py \
  --model /path/to/model.onnx \
  --calib-list /path/to/calibration_npy.txt \
  --output ./work/quantized/model.onnx \
  --method Entropy --chunk-size 8
```

Calibration tensors must already match the model's preprocessing and layout. Quantization and MTNN import are separate modules; standalone quantization does not load a vendor runtime.

## QNN / QAIRT

Use the official SDK environment where both `import qairt` and `import qti` succeed. The base package deliberately does not guess an SDK package from a public package index. Configure the chipset, output tensor names and precision using the selected YAML's `conversion.qnn_config` fields.

The QNN examples contain a default chipset example, not automatic target discovery. Override it for your device. Baseline evaluation can reuse these configs without loading the conversion SDK:

```bash
edge-model-bench pipeline/task/det/qnn/fp16/yolov8n \
  --stage evaluate-onnx --output ./output/cpu/result.json
```

## Horizon HBR

Earlier adapter work used OpenExplorer packages hbdk4 4.7.5, hmct 2.6.5, horizon-tc-ui 3.5.3 and hbm-infer 3.13.6 in a Python 3.10 environment. This records the adapter's historical context; those binaries are not requirements of the core package and this refactor does not certify that SDK matrix.

Use a vendor-supported environment, then install this package with dependencies compatible with that SDK. Do not force a modern NumPy into an SDK environment with conflicting requirements. If the SDK owns its dependency environment, review conflicts before installing; `--no-deps` is appropriate only when you have supplied compatible dependencies yourself.

Separate calibration and conversion:

```bash
edge-model-bench pipeline/conv/cls/hbr/int8/resnet18 --stage calibrate
edge-model-bench pipeline/conv/cls/hbr/int8/resnet18 --stage convert
```

For evaluation, deploy the source/package and compiled artifact to the target board using your own authenticated deployment process. Set `BENCHMARK_ROOT` and `BENCHMARK_WORK_ROOT` there. No host addresses, passwords or automatic SSH connections are configured in this repository.

`tools/bpu_smoke_test.py` and `tools/bpu_stress_test.py` are optional board diagnostics. They use synthetic inputs and are not accuracy evaluations. Review device availability and input dtype compatibility before running stress tests.

## MTNN

The MTC executable and runtime must be installed from the vendor distribution. Common path overrides:

```bash
export BENCHMARK_MTNN_WORK_ROOT=./work/mtnn/cls
export MTNN_MTC_OUTPUT_ROOT=/path/to/toolkit/data/net_data/onnx
edge-model-bench pipeline/conv/cls/mtnn/fp16/resnet18
```

`MTNN_MTC_OUTPUT_ROOT` identifies the toolkit's generated artifact tree; it is not the final benchmark result directory. Review the `MtnnRunner` options in `config/_base_/runner/mtnn.yaml` for runtime initialization and environment settings.

## Validation status

Automated tests cover configuration composition and public imports without vendor SDKs, a generated ONNX model on CPU, resource cleanup, warmup/timing behavior, result serialization and reporting. Vendor toolchain invocations, device execution, accuracy against real datasets, and throughput under board load require separate testing. Record the SDK/runtime versions, model export recipe, dataset subset, device, batch size and preprocessing with any published results.
