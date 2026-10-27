#!/usr/bin/env python3
"""Pure BPU smoke test (model-zoo style): no dataset/eval, only HB_HBMRuntime.run.

Reference:
  https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s/samples/vision/mobilenetv1

- Loads a compiled ``.hbm``
- Feeds dummy tensors matching model input shapes (NV12 Y/UV or featuremap)
- Loops pure ``sess.run`` and samples ``/sys/devices/system/bpu/ratio``
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np


def read_bpu_ratio() -> tuple[str, str]:
    total = Path("/sys/devices/system/bpu/ratio")
    core0 = Path("/sys/devices/system/bpu/bpu0/ratio")
    t = total.read_text().strip() if total.exists() else "N/A"
    c = core0.read_text().strip() if core0.exists() else "N/A"
    return t, c


def monitor_bpu(stop_evt: threading.Event, samples: list, interval: float = 0.02) -> None:
    while not stop_evt.is_set():
        total, core0 = read_bpu_ratio()
        try:
            samples.append((time.time(), float(total), float(core0)))
        except ValueError:
            samples.append((time.time(), total, core0))
        time.sleep(interval)


def _dtype_from_hbm(dt) -> np.dtype:
    # hbm_runtime.hbDNNDataType values: F32=7, U8=3, ...
    name = str(dt)
    if "U8" in name or name.endswith("3"):
        return np.uint8
    if "F16" in name:
        return np.float16
    return np.float32


def build_feed(sess, model_name: str) -> dict[str, np.ndarray]:
    """Build dummy inputs (model-zoo NV12 Y/UV or single featuremap)."""
    in_names = list(sess.input_names[model_name])
    in_shapes = sess.input_shapes[model_name]
    in_dtypes = sess.input_dtypes[model_name]
    feed: dict[str, np.ndarray] = {}
    for name in in_names:
        shape = [d if d and d > 0 else 1 for d in list(in_shapes[name])]
        dtype = _dtype_from_hbm(in_dtypes[name])
        if dtype == np.uint8:
            arr = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        else:
            arr = np.random.randn(*shape).astype(dtype)
        feed[name] = np.ascontiguousarray(arr)
        print(f"[info] feed {name}: shape={arr.shape} dtype={arr.dtype}")
    return feed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure BPU HBM smoke test")
    parser.add_argument("--model", required=True, help="Path to .hbm")
    parser.add_argument("--loops", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    from hbm_runtime import HB_HBMRuntime

    print(f"[info] loading {model_path.resolve()}")
    t0, c0 = read_bpu_ratio()
    print(f"[info] bpu_ratio before load: total={t0} bpu0={c0}")

    sess = HB_HBMRuntime(str(model_path))
    model_name = sess.model_names[0]
    in_names = list(sess.input_names[model_name])
    out_names = list(sess.output_names[model_name])
    print(f"[info] model={model_name}")
    print(f"[info] inputs={in_names}")
    print(f"[info] outputs={out_names}")
    print(f"[info] input_shapes={sess.input_shapes[model_name]}")
    print(f"[info] input_dtypes={sess.input_dtypes[model_name]}")

    # model-zoo style scheduling
    sess.set_scheduling_params(
        priority={model_name: args.priority},
        bpu_cores={model_name: args.bpu_cores},
    )

    feed = build_feed(sess, model_name)
    # multi-model form used by model-zoo: {model_name: {input: tensor}}
    nested_feed = {model_name: feed}

    print(f"[info] warmup={args.warmup}")
    for _ in range(args.warmup):
        _ = sess.run(nested_feed)

    samples: list = []
    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_bpu, args=(stop_evt, samples, 0.02), daemon=True)
    mon.start()

    latencies: list[float] = []
    print(f"[info] pure BPU loops={args.loops}")
    t_start = time.perf_counter()
    for i in range(args.loops):
        s = time.perf_counter()
        outs = sess.run(nested_feed)
        latencies.append(time.perf_counter() - s)
        if i == 0:
            model_outs = outs.get(model_name, next(iter(outs.values())))
            for k, v in model_outs.items():
                a = np.asarray(v)
                print(f"[info] out {k}: shape={a.shape} dtype={a.dtype}")
    total = time.perf_counter() - t_start
    stop_evt.set()
    mon.join(timeout=1.0)

    lat = np.asarray(latencies, dtype=np.float64) * 1000.0
    print("========== RESULT ==========")
    print(f"loops           : {args.loops}")
    print(f"total_s         : {total:.4f}")
    print(f"avg_latency_ms  : {lat.mean():.4f}")
    print(f"p50_latency_ms  : {np.percentile(lat, 50):.4f}")
    print(f"p90_latency_ms  : {np.percentile(lat, 90):.4f}")
    print(f"min_latency_ms  : {lat.min():.4f}")
    print(f"max_latency_ms  : {lat.max():.4f}")
    print(f"fps             : {args.loops / total:.2f}")

    if samples:
        ratios = np.asarray([s[1] for s in samples if isinstance(s[1], (int, float))], dtype=np.float64)
        if ratios.size:
            print(f"bpu_samples     : {ratios.size}")
            print(f"bpu_ratio_avg   : {ratios.mean():.2f}")
            print(f"bpu_ratio_max   : {ratios.max():.2f}")
            print(f"bpu_ratio_min   : {ratios.min():.2f}")
            print(f"bpu_ratio_nonzero_pct : {(ratios > 0).mean() * 100:.1f}%")
        else:
            print("bpu_samples     : unparsed")
    else:
        print("bpu_samples     : 0")

    t, c = read_bpu_ratio()
    print(f"bpu_ratio_now   : total={t} bpu0={c}")
    users = Path("/sys/devices/system/bpu/users")
    if users.exists():
        print("bpu_users:")
        print(users.read_text())
    print("========== PASS ==========")


if __name__ == "__main__":
    main()
