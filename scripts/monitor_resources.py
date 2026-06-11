#!/usr/bin/env python3
"""Standalone CPU/GPU monitor. Run in a separate terminal while training/eval runs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.logging.resource_monitor import ResourceMonitor, sample_resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample CPU/GPU usage over time.")
    parser.add_argument("--duration", type=int, default=60, help="Seconds to monitor")
    parser.add_argument("--interval", type=float, default=2.0, help="Sample interval (seconds)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "resource_monitor.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring for {args.duration}s (interval={args.interval}s)")
    print(f"Output: {out}")
    print("timestamp           elapsed  cpu%   ram_gb  gpu_alloc_gb  gpu_util%")

    monitor = ResourceMonitor(out, interval_s=args.interval)
    monitor._start_time = time.time()
    # Prime cpu_percent for meaningful first reading.
    sample_resources()

    end = time.time() + args.duration
    while time.time() < end:
        snap = monitor.record()
        gpu_alloc = f"{snap.gpu_mem_allocated_gb:.2f}" if snap.gpu_mem_allocated_gb is not None else "n/a"
        gpu_util = f"{snap.gpu_util_percent:.0f}" if snap.gpu_util_percent is not None else "n/a"
        ts = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))
        print(
            f"{ts}  {snap.elapsed_s:6.1f}s  {snap.cpu_percent:5.1f}  "
            f"{snap.ram_used_gb:6.2f}  {gpu_alloc:>12}  {gpu_util:>8}"
        )
        time.sleep(args.interval)

    monitor.record(label="final")
    monitor.save()
    monitor.print_summary()


if __name__ == "__main__":
    main()
