"""Source-checkout entry point; installed users can run edge-model-bench."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_model_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
