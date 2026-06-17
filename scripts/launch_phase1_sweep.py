#!/usr/bin/env python3
"""Launch a grid of Phase 1 GRPO experiments via SkyPilot (no per-run config files).

Edit the SWEEP_GRID below, then:

  # Preview commands
  python scripts/launch_phase1_sweep.py --dry-run

  # Launch all SkyPilot jobs in parallel (Kueue queues them until GPUs are free)
  python scripts/launch_phase1_sweep.py

  # Safe to close terminal after all jobs are submitted (use nohup if you want to detach immediately):
  nohup python scripts/launch_phase1_sweep.py > sweep.log 2>&1 &

  # Run one combo locally (no SkyPilot)
  python scripts/launch_phase1_sweep.py --local --only lr=1e-6,bs=4,ga=1
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_rlvr.env import load_project_env

SKY_YAML = ROOT / "skypilot" / "sky_phase1_rlvr_grpo.yaml"
BASE_CONFIG = ROOT / "configs" / "phase1_rlvr_gsm8k_grpo.yaml"
BASE_WANDB_TAGS = ["phase1", "gsm8k", "grpo", "qwen2.5-0.5b-instruct", "sweep"]

# ---------------------------------------------------------------------------
# Edit this grid — no extra YAML files needed.
# effective_batch_size = batch_size * grad_accum_steps
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    "lr": [1e-6, 5e-7, 1e-5],
    "batch_size": [4],
    "grad_accum_steps": [1],
    'max_steps': [500],
}


@dataclass(frozen=True)
class Experiment:
    lr: float
    batch_size: int
    grad_accum_steps: int
    max_steps: int
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def exp_name(self) -> str:
        lr_tag = _format_lr(self.lr)
        return f"qwen25_p1_lr{lr_tag}_bs{self.batch_size}_ga{self.grad_accum_steps}_max{self.max_steps}_grpo"

    @property
    def wandb_tags(self) -> List[str]:
        lr_tag = _format_lr(self.lr)
        return BASE_WANDB_TAGS + [
            f"lr:{lr_tag}",
            f"bs:{self.batch_size}",
            f"ga:{self.grad_accum_steps}",
            f"eff_bs:{self.effective_batch_size}",
            f"max_steps:{self.max_steps}",
        ]

    @property
    def selector(self) -> str:
        return f"lr={_format_lr(self.lr)},bs={self.batch_size},ga={self.grad_accum_steps},max={self.max_steps}"


def _format_lr(lr: float) -> str:
    if lr == 0:
        return "0"
    exp = int(f"{lr:e}".split("e")[1])
    coeff = lr / (10**exp)
    coeff_str = f"{coeff:g}".replace(".", "p")
    return f"{coeff_str}e{exp}"


def build_experiments(grid: dict[str, Sequence]) -> List[Experiment]:
    keys = ["lr", "batch_size", "grad_accum_steps", "max_steps"]
    combos = itertools.product(*(grid[k] for k in keys))
    return [Experiment(lr=lr, batch_size=bs, grad_accum_steps=ga, max_steps=ms) for lr, bs, ga, ms in combos]


def _sky_launch_cmd(exp: Experiment, wandb_api_key: str | None, *, yes: bool) -> List[str]:
    env = {
        "WANDB_EXP_NAME": exp.exp_name,
        "LR": str(exp.lr),
        "BATCH_SIZE": str(exp.batch_size),
        "GRAD_ACCUM_STEPS": str(exp.grad_accum_steps),
        "WANDB_TAGS": ",".join(exp.wandb_tags),
        "MAX_STEPS": str(exp.max_steps),
    }
    cmd = ["sky", "jobs", "launch", "-n", exp.exp_name, str(SKY_YAML)]
    if yes:
        cmd.append("-y")
    for key, value in env.items():
        cmd.extend(["--env", f"{key}={value}"])
    # if wandb_api_key:
    #     cmd.extend(["--env", f"WANDB_API_KEY={wandb_api_key}"])
    return cmd


def _submit_sky_job(cmd: List[str], exp_name: str) -> Tuple[str, int]:
    proc = subprocess.run(cmd, check=False)
    return exp_name, proc.returncode


def _launch_sky_jobs_parallel(jobs: List[Tuple[List[str], str]], *, max_workers: int) -> None:
    workers = min(max_workers, len(jobs))
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_submit_sky_job, cmd, name) for cmd, name in jobs]
        for future in as_completed(futures):
            name, code = future.result()
            if code != 0:
                failures.append(name)
    if failures:
        raise SystemExit(f"Failed to submit {len(failures)} job(s): {', '.join(sorted(failures))}")


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
        "--max-steps",
        str(exp.max_steps),
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
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Prompt before each sky jobs launch (default: pass -y to skip prompts)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Submit SkyPilot jobs one at a time (default: submit all in parallel)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=32,
        help="Max concurrent sky jobs launch subprocesses (default: 32)",
    )
    args = parser.parse_args()

    if not BASE_CONFIG.exists():
        raise SystemExit(f"Missing base config: {BASE_CONFIG}")
    if not args.local and not SKY_YAML.exists():
        raise SystemExit(f"Missing SkyPilot yaml: {SKY_YAML}")

    load_project_env()
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
        jobs = [(_local_train_cmd(exp, output_root), exp.exp_name) for exp in experiments]
    else:
        if not wandb_api_key and not args.dry_run:
            raise SystemExit(
                "WANDB_API_KEY is not set. Add it to rlvr/.env or export it, then relaunch.\n"
                "Manual launch also requires: --env WANDB_API_KEY=$WANDB_API_KEY"
            )
        jobs = [
            (_sky_launch_cmd(exp, wandb_api_key, yes=not args.ask), exp.exp_name)
            for exp in experiments
        ]

    for cmd, name in jobs:
        print(f"\n[{name}]\n$ " + " ".join(cmd))

    if args.dry_run:
        return

    if args.local:
        for cmd, _ in jobs:
            subprocess.run(cmd, check=True)
    elif args.sequential:
        for cmd, name in jobs:
            _, code = _submit_sky_job(cmd, name)
            if code != 0:
                raise SystemExit(f"Failed to submit job: {name}")
    else:
        print(f"\nSubmitting {len(jobs)} SkyPilot jobs in parallel (Kueue will schedule as GPUs free)...")
        _launch_sky_jobs_parallel(jobs, max_workers=args.max_parallel)

    mode = "local runs" if args.local else "SkyPilot launches"
    print(f"\nSubmitted {len(jobs)} {mode}.")
    if not args.local:
        print("Jobs run on the cluster independently — monitor with: sky jobs queue")


if __name__ == "__main__":
    main()
