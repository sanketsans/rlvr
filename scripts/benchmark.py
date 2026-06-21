#!/usr/bin/env python3
"""Run a benchmark suite: greedy decoding, pass@k / best-of-n, and more."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.env import load_project_env
from qwen3_rlvr.eval.benchmark import BenchmarkSpec, run_benchmark_suite, save_benchmark_results
from qwen3_rlvr.logging.resource_monitor import ResourceMonitor

load_project_env()


def _load_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_benchmarks(raw: List[dict]) -> List[BenchmarkSpec]:
    specs: List[BenchmarkSpec] = []
    for entry in raw:
        specs.append(
            BenchmarkSpec(
                name=entry["name"],
                recipes=list(entry["recipes"]),
                temperature=float(entry["temperature"]),
                n_generations=int(entry["n_generations"]),
                k=[int(x) for x in entry["k"]],
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RLVR benchmark suite.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Sample CPU/GPU usage; writes resource_monitor.json to output-dir",
    )
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    common = _get(cfg, "common", default={}) or {}
    benchmarks = _parse_benchmarks(cfg.get("benchmarks", []))
    if not benchmarks:
        raise SystemExit("Config must define at least one entry under `benchmarks`.")

    model_path = cfg["model"]
    output_dir = args.output_dir or cfg.get("output_dir") or str(ROOT / "outputs" / "benchmark_default")
    max_samples = args.max_samples if args.max_samples is not None else common.get("max_samples")

    def _run() -> None:
        result = run_benchmark_suite(
            model_path=model_path,
            benchmarks=benchmarks,
            max_samples=max_samples,
            max_new_tokens=common.get("max_new_tokens", 512),
            question_batch_size=common.get("question_batch_size", 8),
            dtype=common.get("dtype", "bfloat16"),
            seed=common.get("seed", 42),
            method=common.get("method", "unbiased"),
            config_path=args.config,
        )
        out_path = save_benchmark_results(result, output_dir)
        summary_path = Path(output_dir) / "benchmark_summary_table.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(result.summary_table, f, indent=2)

        print(f"\nModel: {model_path}")
        print(f"Saved: {out_path}")
        print(f"Saved: {summary_path}")
        print("\nSummary:")
        for row in result.summary_table:
            pass_at_1 = row.get("pass@1", row.get("accuracy", 0.0))
            print(
                f"  {row['benchmark']:12s} {row['recipe']:22s} "
                f"T={row['temperature']:.1f} n={row['n_generations']:2d} "
                f"pass@1={pass_at_1:.4f}"
            )

    monitor_path = Path(output_dir) / "resource_monitor.json"
    if args.monitor_resources:
        with ResourceMonitor(monitor_path, interval_s=args.monitor_interval, label="benchmark") as monitor:
            _run()
        monitor.print_summary()
    else:
        _run()


if __name__ == "__main__":
    main()
