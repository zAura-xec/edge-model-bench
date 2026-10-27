# Contributing

This project is licensed under the [MIT License](LICENSE). Contributions are submitted under the same license; preserve applicable third-party notices and identify adapted source in your pull request.

## Development setup

```bash
uv sync --extra onnx
uv run pytest
uv run ruff check src tools examples tests
uv run python -m build
uv run python tools/audit_release.py
```

Use Python 3.10–3.13 for the core. Work in a vendor-supported environment when testing a hardware backend. Keep vendor libraries optional and document tested SDK/device versions.

## What belongs in the core

Reusable dataset adapters, preprocessing, model-output decoders, metrics, runners, converters and configuration utilities are welcome. Keep fixed fleet addresses, deployment credentials, internal storage layouts and one-off experiments in private operational tooling.

A new backend should reuse task logic and decoders where possible. Follow [the extension guide](docs/new_chip_benchmark_pipeline.md). Use `edge_model_bench.*` imports and Hydra targets; add configs under the relevant task/backend directory.

## Pull requests

Describe the concrete behavior change, the model/data assumptions, and how it was verified. Include small synthetic behavioral tests for pipeline changes or bug fixes; do not commit production datasets or model weights. Run config composition tests when changing YAML defaults or targets. Record device checks separately from CPU tests, including any checks you could not run.

Do not invent benchmark scores or present a simplified metric as an official dataset leaderboard score. Preserve original attribution when adapting code and identify its license/source for review.

## Local settings

Use `BENCHMARK_ROOT`, `BENCHMARK_WORK_ROOT`, Hydra overrides, or ignored `config/local/` files for local paths. Relative paths resolve from your current working directory. Check staged files before committing; `.gitignore` does not remove material already tracked in history.
