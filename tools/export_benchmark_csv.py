"""Collect scalar metrics from result.json files without assuming a backend/task."""

import argparse
import csv
import json
from pathlib import Path


def export_results(root: Path, output: Path) -> int:
    rows = []
    for path in sorted(root.rglob("result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue  # Conversion-only outputs may be artifact path strings.
        row = {
            key: value
            for key, value in data.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        row["result_path"] = path.relative_to(root).as_posix()
        rows.append(row)
    if not rows:
        raise ValueError(f"No metric objects found below {root}")
    fields = ["result_path", *sorted(set().union(*(row.keys() for row in rows)) - {"result_path"})]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/results.csv"))
    args = parser.parse_args()
    print(f"Exported {export_results(args.root, args.output)} results to {args.output}")


if __name__ == "__main__":
    main()
