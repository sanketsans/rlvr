#!/usr/bin/env python3
"""Phase 2: HTML gallery from training samples.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.viz.progress_report import build_progress_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML progress report from samples.jsonl.")
    parser.add_argument(
        "--samples",
        type=str,
        default=str(ROOT / "outputs" / "qwen3_rlvr_p1_grpo_gsm8k_debug" / "samples.jsonl"),
    )
    parser.add_argument("--output", type=str, default=None, help="HTML output path")
    parser.add_argument("--per-stage", type=int, default=6)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    if not samples_path.is_file():
        raise SystemExit(f"samples file not found: {samples_path}")

    output = Path(args.output) if args.output else samples_path.with_suffix(".html")
    out = build_progress_report(samples_path, output, per_stage=args.per_stage)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
