# Edge Model Bench

**Configurable accuracy evaluation and inference benchmarking for edge AI models.**

[简体中文](README.zh-CN.md) · [Backend setup](docs/backends.md) · [Add a backend](docs/new_chip_benchmark_pipeline.md) · [Contributing](CONTRIBUTING.md)

Edge Model Bench composes datasets, preprocessing, model execution, output decoding, and metrics through YAML. Reuse the same evaluation protocol across an ONNX baseline and device runtimes, or use each component directly from Python.

```text
Dataset → preprocessing → Runner → postprocessor → Evaluator → result.json
                             ↑
                   optional model conversion
```

## Capabilities and boundaries

| Area | Included |
| --- | --- |
| Tasks | Classification, object detection, semantic/instance segmentation, point-cloud classification, BEV |
| Datasets | ImageNet-style CSV, COCO, VOC, ADE20K, ModelNet10, nuScenes adapters and manifest utilities |
| Execution | ONNX Runtime; optional QAIRT/QNN, Horizon HBR, and MTNN adapters |
| Conversion | Vendor adapter code; ONNX Runtime static quantization, including chunked calibration |
| Metrics | Top-k, precision/recall/F1, detection/instance AP, semantic IoU, point-cloud classification, BEV metrics |
| Performance | Mean, p50 and p90 inference latency; inverse mean latency as `fps` |
| Outputs | JSON results, sequential batch execution, scalar-metric CSV export |

The CPU demo and configuration composition are tested without vendor SDKs. Hardware adapters require the corresponding SDK, runtime and compatible device; hardware execution is **not** covered by ordinary CI. Model YAMLs are examples of export layouts and preprocessing, not a guarantee that an arbitrary model with the same name is compatible.

No model weights, datasets, vendor SDKs, or published benchmark scores are bundled.

## Quick start: a CPU example with no downloads of models or data

Use Python **3.10–3.13**. Some vendor environments impose narrower requirements; see [backend setup](docs/backends.md).

From a source checkout:

```bash
git clone https://github.com/zAura-xec/edge-model-bench.git
cd edge-model-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[onnx]'
python examples/create_demo.py
edge-model-bench demo
```

Or use the committed lockfile with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra onnx
uv run python examples/create_demo.py
uv run edge-model-bench demo
```

The example creates six solid-color images and a tiny ONNX color classifier in `work/demo/`. It runs two warmup calls and evaluates six samples on CPU. Open `work/demo/evaluation/result.json`; `top_1_accuracy` should be `1.0`, with six timed inferences. This synthetic example checks the plumbing and does not measure real-world model quality.

The base installation (`pip install -e .`) supports reusable pipelines, datasets, decoders, metrics and configuration inspection. Install the `onnx` extra only when using ONNX Runtime or ONNX conversion helpers. Vendor SDKs are installed separately.

## Use your own model and data

Resource paths default to `./benchmark` and output paths to `./work`, relative to your **current working directory**. Override them with environment variables or Hydra arguments:

```bash
export BENCHMARK_ROOT=/path/to/benchmark
export BENCHMARK_WORK_ROOT=/path/to/results
edge-model-bench pipeline/task/cls/onnx/resnet18 --print-config
edge-model-bench pipeline/task/cls/onnx/resnet18
```

A typical example config expects:

```text
benchmark/
├── onnx/cls/resnet18-onnx-float/resnet18.onnx
└── dataset/ImageNet-1K/validation_1000.csv
```

Classification CSV format:

```csv
image_path,label
/path/to/images/image_a.jpg,0
/path/to/images/image_b.jpg,1
```

Use integer labels matching the model's class indices. Relative image paths require the dataset's `image_root` option, or are interpreted relative to the process working directory. The `ImageNet1K` adapter also works for other classification datasets with this CSV schema.

Override model paths, class-related settings and preprocessing to match your actual export:

```bash
edge-model-bench pipeline/task/cls/onnx/resnet18 \
  model_path=/path/to/model.onnx \
  validation_csv=/path/to/validation.csv \
  input_size='[224,224]' \
  work_root=./work/my_cpu_run
```

Review color order, resize policy, normalization, input/output names, tensor layout and class mapping before comparing results. Detection and segmentation exports vary especially widely. Calibration and validation should be separate datasets. The legacy nuScenes mini manifest helper samples from the same mini sample pool for both uses; prepare separate scene splits before using it for calibration/accuracy comparisons.

`.env.example` lists optional variables; export them in your shell. The CLI does not automatically read `.env`.

## Configuration and stages

Configs are trusted executable input: Hydra `_target_` values instantiate Python classes. Inspect unfamiliar configs before running them.

Both a bundled config name and a source YAML path work:

```bash
edge-model-bench demo --print-config
python tools/pipeline.py config/demo.yaml --output ./output/demo/result.json
python -m edge_model_bench demo --no-save-result
```

Bundled configs work after wheel installation and from another directory. For a custom config tree, use `--config-root /path/to/config` so absolute Hydra defaults resolve against that tree. A standalone YAML outside the built-in tree uses its own parent as the config root.

| Stage | Behavior |
| --- | --- |
| `task` (default) | Run the configured pipeline; conversion tasks can prepare calibration first |
| `calibrate` | Iterate `pipeline.calibration_dataset` to write calibration tensors |
| `convert` | Run only the converter; prepare required calibration beforehand |
| `quantize-onnx` | Run the selected converter's independent ONNX quantizer, if provided |
| `convert-static-onnx` | Import an already quantized ONNX using a converter that supports it |
| `evaluate-onnx` | Evaluate `model_path` or `conversion.input_model` with ONNX Runtime |
| `evaluate` | Evaluate a prebuilt artifact with the configured runner |
| `evaluate-qnn` | Compatibility alias for `evaluate` |

Example for a device task with a separate CPU baseline:

```bash
edge-model-bench pipeline/task/det/qnn/fp16/yolov8n \
  --stage evaluate-onnx --provider CPUExecutionProvider \
  --output ./output/yolov8n_cpu/result.json
```

This stage instantiates evaluation components only, so the conversion SDK is not loaded. Its decoder still must match the ONNX output names and layout. `--provider` applies to `evaluate-onnx`; `task` uses the runner settings in YAML. A required provider that is unavailable raises an error instead of silently reporting CPU performance as GPU performance.

For prebuilt device evaluation, set `pipeline.model`, `input_bin`, or `+compiled_model=/path/to/artifact`. Legacy QNN configs can resolve a single compiled `.bin` from their output directory; ambiguous matches fail.

Evaluation stages save JSON by default to `<work_dir>/result.json` (`./work/result.json` when unset). Preparation/conversion stages save only with `--save-result`. Use `--output` to isolate different baselines and devices.

## Results and performance interpretation

Task-specific metrics remain at the top level. For example:

```json
{
  "num_samples": 6,
  "top_1_accuracy": 1.0,
  "average_latency_ms": 0.25,
  "p50_latency_ms": 0.24,
  "p90_latency_ms": 0.28,
  "fps": 4000.0,
  "num_inferences": 6
}
```

Latency numbers above are illustrative. Timings cover the runner's execution call; model loading, data reading, preprocessing, postprocessing and metric calculation are excluded. Some runtimes include data transfer in that call. `fps = 1000 / average_latency_ms` is inference-call throughput for the configured inputs, not end-to-end application throughput. Compare the same input size, batch size, warmup, runtime, precision and device conditions. Set `+pipeline.warmup=10` on a config without an existing warmup field.

Results omit timing metrics if a runner records no durations. Empty datasets and non-finite result values fail explicitly. Files are replaced atomically after successful JSON serialization.

The included AP and BEV metrics implement this project's evaluation protocols. Do not present them as COCO/nuScenes official leaderboard results without checking against the official evaluator. FastSAM decoding is a proposal/proxy workflow and is not a prompted segmentation benchmark.

## Batch runs and reports

The batch tool runs explicit model configs sequentially and writes a separate log and result for each job:

```bash
python tools/benchmark_all.py \
  config/pipeline/task/cls/onnx/resnet18.yaml \
  config/pipeline/task/cls/onnx/resnet50.yaml \
  --output-dir ./output/cpu --keep-going
python tools/export_benchmark_csv.py ./output/cpu --output ./output/cpu.csv
```

Use repeatable `--override key=value` options for shared overrides, and `--stage evaluate-onnx --provider CPUExecutionProvider` for ONNX baselines of device configs. Batches stop on the first failure unless `--keep-going` is set, and return a nonzero exit status if any job fails. Old per-job result files are removed before rerunning. There is no implicit remote execution or resume cache.

CSV export includes scalar fields from every result object, with `result_path` identifying each run. Nested confusion matrices, class lists and classification reports stay in JSON.

## Reuse from Python

```python
from functools import partial
from edge_model_bench.pipeline import ModelEvaluator
from edge_model_bench.runner import OnnxRunner

# Supply your own dataset, decoder and evaluator, or instantiate them with Hydra.
result = ModelEvaluator(
    model="model.onnx",
    dataset=dataset,
    runner=partial(OnnxRunner, providers=["CPUExecutionProvider"]),
    postprocessor=decoder,
    evaluator=metrics,
    warmup=10,
)()
```

A dataset provides `len(dataset)` and `(inputs, sample)` through indexing. A runner factory takes a model artifact and returns a callable with `latencies` (seconds) and `close()`. A postprocessor maps `(sample, outputs)` to a prediction sample. An evaluator maps samples to a metric dictionary. See the [extension guide](docs/new_chip_benchmark_pipeline.md) for a complete backend adapter example.

## Repository layout

```text
src/edge_model_bench/    Installable package: pipelines, runners, converters, datasets and metrics
config/                 Hydra configuration tree, also included in wheels
examples/               Synthetic CPU demo generator
tools/                  Batch/report tools, quantization, dataset utilities and release preparation
tests/                  Core behavior, all config composition, CPU integration and tool tests
docs/                   Backend setup, extension guide and release checklist
.github/                CI, issue forms and pull request template
```

The `Qnn` prefix on some decoders is historical: these NumPy postprocessors can also decode matching named outputs from other runtimes. They do not import QNN. Public Python imports now use `edge_model_bench.*`; old top-level imports such as `from runner import ...` must be updated.

## Development

```bash
uv sync --extra onnx
uv run pytest
uv run ruff check src tools examples tests
uv run python -m build
uv run python tools/audit_release.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests and backend contributions, and [SECURITY.md](SECURITY.md) for reporting sensitive issues.

## License

Released under the [MIT License](LICENSE), with copyright attribution retained from the destination repository. Vendor SDKs, models and datasets retain their own terms. See [third-party notes](THIRD_PARTY_NOTICES.md) and the [release checklist](docs/releasing.md).
