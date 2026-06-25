"""Lightweight CPU / GPU resource sampling over time."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import psutil
import torch


@dataclass
class ResourceSnapshot:
    timestamp: float
    elapsed_s: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_available: bool
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None
    gpu_mem_total_gb: Optional[float] = None
    gpu_mem_allocated_gb: Optional[float] = None
    gpu_mem_reserved_gb: Optional[float] = None
    label: Optional[str] = None


@dataclass
class ResourceMonitorSummary:
    num_samples: int
    duration_s: float
    cpu_percent_avg: float
    cpu_percent_max: float
    ram_used_gb_max: float
    gpu_mem_allocated_gb_max: Optional[float]
    gpu_mem_reserved_gb_max: Optional[float]
    gpu_mem_used_gb_max: Optional[float]
    gpu_util_percent_max: Optional[float]
    output_path: str
    samples: List[ResourceSnapshot] = field(repr=False)


def _query_nvidia_smi() -> Optional[tuple[float, float, float]]:
    """Return (gpu_util%, mem_used_gb, mem_total_gb) for GPU 0."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        util, used_mib, total_mib = [float(x.strip()) for x in out.splitlines()[0].split(",")]
        return util, used_mib / 1024, total_mib / 1024
    except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
        return None


def sample_resources(elapsed_s: float = 0.0, label: Optional[str] = None) -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    snap = ResourceSnapshot(
        timestamp=time.time(),
        elapsed_s=elapsed_s,
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_gb=vm.used / (1024**3),
        ram_total_gb=vm.total / (1024**3),
        ram_percent=vm.percent,
        gpu_available=torch.cuda.is_available(),
        label=label,
    )

    if torch.cuda.is_available():
        snap.gpu_mem_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        snap.gpu_mem_reserved_gb = torch.cuda.memory_reserved() / (1024**3)

    smi = _query_nvidia_smi()
    if smi is not None:
        snap.gpu_util_percent, snap.gpu_mem_used_gb, snap.gpu_mem_total_gb = smi

    return snap


class ResourceMonitor:
    """Background sampler; use as a context manager around training/eval."""

    def __init__(
        self,
        output_path: str | Path,
        interval_s: float = 2.0,
        label: Optional[str] = None,
    ):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self.label = label
        self._samples: List[ResourceSnapshot] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        # Prime cpu_percent so the first background sample is meaningful.
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)
        self.record(label="final")
        self.save()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.record()

    def record(self, label: Optional[str] = None) -> ResourceSnapshot:
        elapsed = time.time() - self._start_time
        snap = sample_resources(
            elapsed_s=elapsed,
            label=label or self.label,
        )
        self._samples.append(snap)
        return snap

    def save(self) -> ResourceMonitorSummary:
        summary = self.summarize()
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {k: v for k, v in asdict(summary).items() if k != "samples"},
                    "samples": [asdict(s) for s in self._samples],
                },
                f,
                indent=2,
            )
        return summary

    def summarize(self) -> ResourceMonitorSummary:
        if not self._samples:
            return ResourceMonitorSummary(
                num_samples=0,
                duration_s=0.0,
                cpu_percent_avg=0.0,
                cpu_percent_max=0.0,
                ram_used_gb_max=0.0,
                gpu_mem_allocated_gb_max=None,
                gpu_mem_reserved_gb_max=None,
                gpu_mem_used_gb_max=None,
                gpu_util_percent_max=None,
                output_path=str(self.output_path),
                samples=[],
            )

        cpu_vals = [s.cpu_percent for s in self._samples]
        gpu_alloc = [
            s.gpu_mem_allocated_gb for s in self._samples if s.gpu_mem_allocated_gb is not None
        ]
        gpu_reserved = [
            s.gpu_mem_reserved_gb for s in self._samples if s.gpu_mem_reserved_gb is not None
        ]
        gpu_used = [s.gpu_mem_used_gb for s in self._samples if s.gpu_mem_used_gb is not None]
        gpu_util = [s.gpu_util_percent for s in self._samples if s.gpu_util_percent is not None]

        return ResourceMonitorSummary(
            num_samples=len(self._samples),
            duration_s=self._samples[-1].elapsed_s,
            cpu_percent_avg=sum(cpu_vals) / len(cpu_vals),
            cpu_percent_max=max(cpu_vals),
            ram_used_gb_max=max(s.ram_used_gb for s in self._samples),
            gpu_mem_allocated_gb_max=max(gpu_alloc) if gpu_alloc else None,
            gpu_mem_reserved_gb_max=max(gpu_reserved) if gpu_reserved else None,
            gpu_mem_used_gb_max=max(gpu_used) if gpu_used else None,
            gpu_util_percent_max=max(gpu_util) if gpu_util else None,
            output_path=str(self.output_path),
            samples=self._samples,
        )

    def print_summary(self) -> None:
        s = self.summarize()
        print(f"Resource monitor ({s.num_samples} samples, {s.duration_s:.1f}s)")
        print(f"  CPU: avg {s.cpu_percent_avg:.1f}%, peak {s.cpu_percent_max:.1f}%")
        print(f"  RAM peak: {s.ram_used_gb_max:.2f} GB")
        if s.gpu_mem_allocated_gb_max is not None:
            print(f"  GPU torch allocated peak: {s.gpu_mem_allocated_gb_max:.2f} GB")
            print(f"  GPU torch reserved peak:  {s.gpu_mem_reserved_gb_max:.2f} GB")
        if s.gpu_mem_used_gb_max is not None:
            print(f"  GPU nvidia-smi used peak: {s.gpu_mem_used_gb_max:.2f} GB")
        if s.gpu_util_percent_max is not None:
            print(f"  GPU util peak: {s.gpu_util_percent_max:.0f}%")
        print(f"  Saved: {s.output_path}")
