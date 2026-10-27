# Add a task or inference backend

Keep dataset definitions, preprocessing, decoding and metric computation independent of the hardware. A new execution backend usually needs a runner, optionally a converter, and a YAML config.

## Component contracts

| Component | Contract |
| --- | --- |
| Dataset | `len(dataset)` and `dataset[index] -> (inputs, sample)` |
| Converter | Callable returning an artifact path or a model accepted by the runner |
| Runner factory | Callable accepting the artifact and returning a runner |
| Runner | Callable on inputs, `latencies: list[float]` in seconds, `close()` |
| Postprocessor | `(sample, raw_outputs) -> prediction_sample` |
| Evaluator | `prediction_samples -> dict` with JSON-friendly metrics |

Inputs can be a NumPy array or a dictionary for multi-input models. Choose an output mapping or list that agrees with your postprocessor. Prefer explicit tensor names over guessing output positions. Postprocessors receive a deep copy of sample metadata, so predictions do not mutate the dataset's cached annotations.

The evaluator loop always closes a successfully constructed runner, including when inference or decoding fails. The backend constructor must clean up its own partially initialized resources if it raises.

## Implement a runner

Place the adapter in `src/edge_model_bench/runner/my_backend.py`. Import the optional runtime in the constructor so other users do not need it:

```python
from time import perf_counter


class MyRunner:
    def __init__(self, model_path, device="auto"):
        try:
            import my_runtime
        except ImportError as exc:
            raise RuntimeError("Install the official MyRuntime SDK first") from exc
        self.session = my_runtime.load(str(model_path), device=device)
        self.latencies = []

    def __call__(self, inputs):
        start = perf_counter()
        outputs = self.session.run(inputs)
        # Synchronize here if run() is asynchronous, before stopping the timer.
        self.latencies.append(perf_counter() - start)
        return outputs

    def close(self):
        self.session.close()
```

This is an interface example: `my_runtime` is a placeholder, not a bundled dependency. Do not add a proprietary SDK or local wheel path to the base dependency list.

## Compose a pipeline

The smallest reusable configuration can use your new runner with an existing dataset and decoder:

```yaml
# @package _global_
work_dir: ./work/my_backend
model_path: /path/to/compiled/model
pipeline:
  _target_: edge_model_bench.pipeline.ModelEvaluator
  model: ${model_path}
  warmup: 10
  dataset:
    _target_: edge_model_bench.dataset.ImageNet1K
    csv_path: /path/to/validation.csv
    pipeline:
      - _target_: edge_model_bench.dataset.LoadingImage
      - _target_: edge_model_bench.dataset.BGR2RGB
      - _target_: edge_model_bench.dataset.Resize
        size: [224, 224]
      - _target_: edge_model_bench.dataset.Normalize
        mean: 0
        std: 255
      - _target_: edge_model_bench.dataset.HWC2NCHW
  runner:
    _target_: edge_model_bench.runner.my_backend.MyRunner
    _partial_: true
  postprocessor:
    _target_: edge_model_bench.classifier.OnnxClassifier
    output_index: 0
  evaluator:
    _target_: edge_model_bench.evaluator.ClsTaskEvaluator
```

This decoder expects a list whose first output contains class logits. Use the named-output classifier or your own decoder if the runtime returns a dictionary. Change preprocessing to match the actual model.

For optional conversion, use `ModelTask` with `converter`, `calibration_dataset` and the same evaluation components. `ModelConverter` prepares calibration and then invokes the converter. The `--stage convert` CLI option invokes only the converter; run `--stage calibrate` first if required.

```bash
edge-model-bench /path/to/custom.yaml --print-config
edge-model-bench /path/to/custom.yaml
edge-model-bench /path/to/custom.yaml --stage evaluate
```

For larger configuration trees use `--config-root`. Hydra absolute defaults refer to that root. All relative file paths are interpreted from the shell's current directory.

## Add a new task

Implement a dataset adapter or reuse one of the existing CSV/manifest readers. Define the prediction sample schema, add a decoder, and implement a callable metric evaluator. Keep hardware code out of those modules.

Useful existing schemas include classification `{label, pred: {class_id: score}}`, detection boxes/scores/categories and segmentation masks. Read the corresponding existing evaluator before writing a decoder; field names, coordinate systems, category IDs and mask sizes must agree.

Return scalar metrics and optionally structured details. The JSON writer handles NumPy scalars/arrays and paths, but rejects NaN and infinity. The CSV tool exports only scalar fields, so full confusion matrices and per-class reports stay in JSON.

## Verification before proposing an adapter

1. Test with known inputs and verify names, shapes, dtype, layout and units.
2. Verify the runner closes on success and inference/decoding failure.
3. Verify warmup is excluded, recorded durations are positive seconds and asynchronous work is synchronized.
4. Compare identical samples and preprocessing against an ONNX baseline.
5. Test artifact selection explicitly; do not silently pick one of several model files.
6. Record hardware and SDK versions with results. Keep SDK binaries, credentials, internal hosts, datasets and weights outside Git.
7. Add config composition and relevant behavioral regression tests. Mark device-dependent checks separately from CPU CI.

Historical `Qnn*` decoder names describe output layouts; they do not require a QNN runtime. Reuse a compatible decoder before adding a backend-specific copy.
