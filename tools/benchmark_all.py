"""Run explicitly selected configs sequentially with isolated result/log paths."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", type=Path, help="Model YAML files (shell globs allowed)"
    )
    parser.add_argument("--stage", default="task")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--override", action="append", default=[], help="Hydra key=value (repeatable)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/batch"))
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    failed = []
    for index, config in enumerate(args.configs):
        directory = args.output_dir / f"{index:03d}_{config.stem}"
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        # Do not leave an old successful result behind if a rerun fails.
        result_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "edge_model_bench",
            str(config),
            *args.override,
            "--stage",
            args.stage,
            "--provider",
            args.provider,
            "--save-result",
            "--output",
            str(result_path),
        ]
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'FAIL' if status else 'OK'} {config}: {directory}", flush=True)
        if status:
            failed.append(str(config))
            if not args.keep_going:
                break
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
