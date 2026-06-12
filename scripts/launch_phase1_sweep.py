#!/usr/bin/env python3
"""Launch a grid of Phase 1 GRPO experiments via SkyPilot (no per-run config files).

Edit the SWEEP_GRID below, then:

  # Preview commands
  python scripts/launch_phase1_sweep.py --dry-run

  # Launch all SkyPilot jobs
  WANDB_API_KEY=... python scripts/launch_phase1_sweep.py

  # Run one combo locally (no SkyPilot)
  python scripts/launch_phase1_sweep.py --local --only lr=1e-6,bs=4,ga=1
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SKY_YAML = ROOT / "skypilot" / "sky_phase1_rlvr_reinfoce.yaml"
BASE_CONFIG = ROOT / "configs" / "phase1_rlvr_gsm8k_reinforce.yaml"
BASE_WANDB_TAGS = ["phase1", "gsm8k", "reinforce", "qwen2.5-0.5b-instruct", "sweep"]

# ---------------------------------------------------------------------------
# Edit this grid — no extra YAML files needed.
# effective_batch_size = batch_size * grad_accum_steps
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    "lr": [1e-6, 5e-7],
    "batch_size": [2, 4],
    "grad_accum_steps": [1, 2],
}


@dataclass(frozen=True)
class Experiment:
    lr: float
    batch_size: int
    grad_accum_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def exp_name(self) -> str:
        lr_tag = _format_lr(self.lr)
        return f"qwen25_p1_lr{lr_tag}_bs{self.batch_size}_ga{self.grad_accum_steps}"

    @property
    def wandb_tags(self) -> List[str]:
        lr_tag = _format_lr(self.lr)
        return BASE_WANDB_TAGS + [
            f"lr:{lr_tag}",
            f"bs:{self.batch_size}",
            f"ga:{self.grad_accum_steps}",
            f"eff_bs:{self.effective_batch_size}",
        ]

    @property
    def selector(self) -> str:
        return f"lr={_format_lr(self.lr)},bs={self.batch_size},ga={self.grad_accum_steps}"


def _format_lr(lr: float) -> str:
    if lr == 0:
        return "0"
    exp = int(f"{lr:e}".split("e")[1])
    coeff = lr / (10**exp)
    coeff_str = f"{coeff:g}".replace(".", "p")
    return f"{coeff_str}e{exp}"


def build_experiments(grid: dict[str, Sequence]) -> List[Experiment]:
    keys = ["lr", "batch_size", "grad_accum_steps"]
    combos = itertools.product(*(grid[k] for k in keys))
    return [Experiment(lr=lr, batch_size=bs, grad_accum_steps=ga) for lr, bs, ga in combos]


def _sky_launch_cmd(exp: Experiment, wandb_api_key: str | None) -> List[str]:
    env = {
        "EXP_NAME": exp.exp_name,
        "LR": str(exp.lr),
        "BATCH_SIZE": str(exp.batch_size),
        "GRAD_ACCUM_STEPS": str(exp.grad_accum_steps),
        "WANDB_TAGS": ",".join(exp.wandb_tags),
    }
    cmd = ["sky", "jobs", "launch", str(SKY_YAML)]
    for key, value in env.items():
        cmd.extend(["--env", f"{key}={value}"])
    if wandb_api_key:
        cmd.extend(["--env", f"WANDB_API_KEY={wandb_api_key}"])
    return cmd


def _local_train_cmd(exp: Experiment, output_root: Path) -> List[str]:
    out_dir = output_root / exp.exp_name
    return [
        sys.executable,
        str(ROOT / "scripts" / "train_grpo.py"),
        "--config",
        str(BASE_CONFIG),
        "--output-dir",
        str(out_dir),
        "--lr",
        str(exp.lr),
        "--batch-size",
        str(exp.batch_size),
        "--grad-accum-steps",
        str(exp.grad_accum_steps),
        "--wandb-name",
        exp.exp_name,
        "--wandb-tags",
        ",".join(exp.wandb_tags),
    ]


def _filter_experiments(experiments: Iterable[Experiment], only: str | None) -> List[Experiment]:
    if not only:
        return list(experiments)
    wanted = {item.strip() for item in only.split(";") if item.strip()}
    selected = [exp for exp in experiments if exp.selector in wanted]
    if not selected:
        raise SystemExit(f"No experiments matched --only. Available selectors:\n" + "\n".join(
            f"  {exp.selector}" for exp in experiments
        ))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Phase 1 GRPO hyperparameter sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--local", action="store_true", help="Run train_grpo.py locally instead of SkyPilot")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run subset only, e.g. 'lr=1e-6,bs=4,ga=1' or semicolon-separated list",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.environ.get("OUT_ROOT", str(ROOT / "outputs")),
        help="Local output root when using --local",
    )
    args = parser.parse_args()

    if not BASE_CONFIG.exists():
        raise SystemExit(f"Missing base config: {BASE_CONFIG}")
    if not args.local and not SKY_YAML.exists():
        raise SystemExit(f"Missing SkyPilot yaml: {SKY_YAML}")

    experiments = _filter_experiments(build_experiments(SWEEP_GRID), args.only)
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    print(f"Planned experiments ({len(experiments)}):")
    for exp in experiments:
        print(
            f"  {exp.exp_name} | lr={exp.lr} bs={exp.batch_size} "
            f"ga={exp.grad_accum_steps} eff_bs={exp.effective_batch_size}"
        )

    if args.local:
        output_root = Path(args.output_root)
        commands = [_local_train_cmd(exp, output_root) for exp in experiments]
    else:
        if not wandb_api_key and not args.dry_run:
            print("Warning: WANDB_API_KEY is not set; Sky jobs may fail W&B init.", file=sys.stderr)
        commands = [_sky_launch_cmd(exp, wandb_api_key) for exp in experiments]

    for cmd in commands:
        print("\n$ " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    if not args.dry_run:
        mode = "local runs" if args.local else "SkyPilot launches"
        print(f"\nSubmitted {len(commands)} {mode}.")


if __name__ == "__main__":
    main()
