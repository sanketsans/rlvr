"""Unified benchmark suite for model capability assessment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qwen3_rlvr.eval.recipe_eval import evaluate_recipe_quick
from qwen3_rlvr.model.load import LoadedModel, load_model_and_tokenizer


@dataclass
class BenchmarkSpec:
    name: str
    recipes: List[str]
    temperature: float
    n_generations: int
    k: List[int]


@dataclass
class BenchmarkRun:
    benchmark_name: str
    recipe: str
    temperature: float
    n_generations: int
    k_values: List[int]
    num_examples: int
    metrics: Dict[str, float]
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    model_path: str
    config_path: Optional[str]
    created_at: str
    runs: List[BenchmarkRun]
    summary_table: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "created_at": self.created_at,
            "runs": [
                {
                    "benchmark_name": run.benchmark_name,
                    "recipe": run.recipe,
                    "temperature": run.temperature,
                    "n_generations": run.n_generations,
                    "k_values": run.k_values,
                    "num_examples": run.num_examples,
                    "metrics": run.metrics,
                    "by_source": run.by_source,
                }
                for run in self.runs
            ],
            "summary_table": self.summary_table,
        }


def _metric_subset(metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = ("accuracy", "avg_num_correct")
    out = {k: float(metrics[k]) for k in keys if k in metrics}
    for key, value in metrics.items():
        if key.startswith("pass@"):
            out[key] = float(value)
    return out


def build_summary_table(runs: Sequence[BenchmarkRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        row: Dict[str, Any] = {
            "benchmark": run.benchmark_name,
            "recipe": run.recipe,
            "temperature": run.temperature,
            "n_generations": run.n_generations,
            "num_examples": run.num_examples,
        }
        row.update(run.metrics)
        rows.append(row)
    return rows


def run_benchmark_suite(
    model_path: str,
    benchmarks: Sequence[BenchmarkSpec],
    *,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    question_batch_size: int = 8,
    dtype: str = "bfloat16",
    seed: int = 42,
    method: str = "unbiased",
    config_path: Optional[str] = None,
    loaded: Optional[LoadedModel] = None,
    show_progress: bool = True,
) -> BenchmarkSuiteResult:
    owns_model = loaded is None
    if loaded is None:
        if show_progress:
            print(f"Loading model: {model_path}")
        loaded = load_model_and_tokenizer(model_path=model_path, dtype=dtype)

    runs: List[BenchmarkRun] = []
    try:
        for spec in benchmarks:
            if show_progress:
                print(
                    f"\nBenchmark '{spec.name}': "
                    f"T={spec.temperature}, n={spec.n_generations}, k={spec.k}"
                )
            for recipe in spec.recipes:
                if show_progress:
                    print(f"  recipe={recipe}")
                result = evaluate_recipe_quick(
                    loaded=loaded,
                    recipe=recipe,
                    max_samples=max_samples,
                    n_generations=spec.n_generations,
                    k_values=spec.k,
                    max_new_tokens=max_new_tokens,
                    temperature=spec.temperature,
                    seed=seed,
                    question_batch_size=question_batch_size,
                    method=method,
                )
                metrics = _metric_subset(result)
                runs.append(
                    BenchmarkRun(
                        benchmark_name=spec.name,
                        recipe=recipe,
                        temperature=spec.temperature,
                        n_generations=spec.n_generations,
                        k_values=list(spec.k),
                        num_examples=int(result["num_examples"]),
                        metrics=metrics,
                        by_source=result.get("by_source", {}),
                    )
                )
                if show_progress:
                    print(f"    pass@1={metrics.get('pass@1', metrics.get('accuracy', 0.0)):.4f}")
    finally:
        if owns_model:
            del loaded

    suite = BenchmarkSuiteResult(
        model_path=model_path,
        config_path=config_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        runs=runs,
        summary_table=build_summary_table(runs),
    )
    return suite


def save_benchmark_results(result: BenchmarkSuiteResult, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "benchmark_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path
