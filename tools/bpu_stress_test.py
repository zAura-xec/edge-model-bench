#!/usr/bin/env python3
"""Multi-thread pure BPU stress to observe /sys/.../bpu/ratio."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    from hbm_runtime import HB_HBMRuntime

    sess = HB_HBMRuntime(args.model)
    name = sess.model_names[0]
    sess.set_scheduling_params(priority={name: 0}, bpu_cores={name: [0]})
    in_names = list(sess.input_names[name])
    shapes = sess.input_shapes[name]
    feed = {}
    for n in in_names:
        shape = [d if d and d > 0 else 1 for d in shapes[n]]
        feed[n] = np.ascontiguousarray(
            np.random.randint(0, 256, size=shape, dtype=np.uint8)
        )
    nested = {name: feed}

    for _ in range(args.warmup):
        sess.run(nested)

    stop = False
    counts = [0] * args.threads

    def worker(wid: int) -> None:
        local = 0
        while not stop:
            sess.run(nested)
            local += 1
        counts[wid] = local

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(args.threads)
    ]
    for t in threads:
        t.start()

    samples: list[tuple[float, float]] = []
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        total = Path("/sys/devices/system/bpu/ratio").read_text().strip()
        core0 = Path("/sys/devices/system/bpu/bpu0/ratio").read_text().strip()
        try:
            samples.append((float(total), float(core0)))
        except ValueError:
            samples.append((0.0, 0.0))
        time.sleep(0.05)

    stop = True
    for t in threads:
        t.join(timeout=3)

    totals = [s[0] for s in samples]
    print(f"threads={args.threads} seconds={args.seconds}")
    print(f"infer_counts={counts} sum={sum(counts)}")
    print(f"approx_fps={sum(counts) / args.seconds:.1f}")
    print(f"bpu_samples={len(totals)}")
    print(f"bpu_ratio_max={max(totals) if totals else 0}")
    print(f"bpu_ratio_avg={sum(totals)/len(totals) if totals else 0:.2f}")
    print(
        f"bpu_ratio_nonzero_pct="
        f"{(100 * sum(v > 0 for v in totals) / len(totals)) if totals else 0:.1f}"
    )
    print("trace_head", samples[:5])
    print("trace_tail", samples[-5:])
    print("users:")
    print(Path("/sys/devices/system/bpu/users").read_text())


if __name__ == "__main__":
    main()
