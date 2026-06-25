"""Rejection-sampling curation helpers for SFT datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from qwen3_rlvr.logging.logger import setup_logger
from qwen3_rlvr.rewards.extract import answers_match

logger = setup_logger(__name__)


@dataclass(frozen=True)
class CuratedRow:
    prompt_id: int
    question: str
    ground_truth: str
    completion: str
    completion_length: int
    rollout_index: int
    variant: str
    rank: int | None = None
    success_ratio: float | None = None

    def to_record(self) -> dict:
        return asdict(self)


def _completion_length(text: str) -> int:
    return len(text.strip())


def rank_correct_completions(
    completions: Sequence[str],
    ground_truth: str,
    *,
    top_k: int = 2,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], float]:
    """Return (top_k_correct, all_correct) as (completion, rollout_index) pairs."""
    correct: List[Tuple[str, int]] = []
    for rollout_index, completion in enumerate(completions):
        if answers_match(completion, ground_truth):
            correct.append((completion, rollout_index))

    success_ratio = len(correct) / len(completions)
    ranked = sorted(
        correct,
        key=lambda item: (_completion_length(item[0]), item[1]),
    )
    return ranked[:top_k], correct, success_ratio


def curate_rollouts(
    prompt_id: int,
    question: str,
    ground_truth: str,
    completions: Sequence[str],
    *,
    top_k: int = 2,
) -> Tuple[List[CuratedRow], List[CuratedRow], float]:
    top_rows: List[CuratedRow] = []
    all_rows: List[CuratedRow] = []

    top_correct, all_correct, success_ratio = rank_correct_completions(
        completions,
        ground_truth,
        top_k=top_k,
    )

    for rank, (completion, rollout_index) in enumerate(top_correct):
        top_rows.append(
            CuratedRow(
                prompt_id=prompt_id,
                question=question,
                ground_truth=ground_truth,
                completion=completion,
                completion_length=_completion_length(completion),
                rollout_index=rollout_index,
                variant="top2",
                rank=rank,
                success_ratio=success_ratio,
            )
        )

    for completion, rollout_index in all_correct:
        all_rows.append(
            CuratedRow(
                prompt_id=prompt_id,
                question=question,
                ground_truth=ground_truth,
                completion=completion,
                completion_length=_completion_length(completion),
                rollout_index=rollout_index,
                variant="all_correct",
                rank=None,
                success_ratio=success_ratio,
            )
        )

    return top_rows, all_rows, success_ratio


def rows_to_parquet(rows: Sequence[CuratedRow], path: Path) -> None:
    raise NotImplementedError("Parquet export removed; write JSONL via append_rows_jsonl instead.")


def append_rows_jsonl(rows: Sequence[CuratedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_record()) + "\n")


def load_rows_jsonl(path: Path) -> List[CuratedRow]:
    rows: List[CuratedRow] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            rows.append(CuratedRow(**payload))
    return rows


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    raise NotImplementedError("Parquet export removed; use JSONL paths from the manifest.")


def classify_difficulty(success_ratio: float) -> str:
    """Map rollout success ratio to curriculum bucket."""
    if success_ratio >= 0.7:
        return "easy"
    if success_ratio >= 0.3:
        return "mid"
    if success_ratio > 0.0:
        return "hard"
    return "very_hard"


def load_prompt_success_ratios(path: Path) -> Dict[int, float]:
    ratios: Dict[int, float] = {}
    if not path.is_file():
        return ratios
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                prompt_id_str, ratio_str = line.split(",", 1)
                ratios[int(prompt_id_str)] = float(ratio_str)
            else:
                ratios[int(line)] = 0.0
    return ratios


def load_processed_prompt_ids(path: Path) -> set[int]:
    return set(load_prompt_success_ratios(path).keys())


def mark_prompt_processed(path: Path, prompt_id: int, success_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{prompt_id},{success_ratio}\n")


def summarize_rows(rows: Iterable[CuratedRow]) -> Dict[str, Any]:
    rows = list(rows)
    prompt_ids = {row.prompt_id for row in rows}
    avg_success_ratio = sum(row.success_ratio for row in rows) / len(rows)
    lengths = [row.completion_length for row in rows]
    return {
        "num_rows": len(rows),
        "num_unique_prompts": len(prompt_ids),
        "avg_success_ratio": avg_success_ratio,
        "avg_completion_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "min_completion_length": min(lengths) if lengths else 0,
        "max_completion_length": max(lengths) if lengths else 0,
    }


def write_manifest(
    path: Path,
    *,
    model_path: str,
    output_dir: Path,
    num_source_prompts: int,
    n_rollouts_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
    top2_jsonl_path: Path,
    all_correct_jsonl_path: Path,
    top2_summary: Dict[str, Any],
    all_correct_summary: Dict[str, Any],
    config_path: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> dict:
    logger.info(f"Writing manifest to {path}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "gsm8k_rejection_sampling",
        "model_path": model_path,
        "config_path": config_path,
        "source_dataset": "openai/gsm8k",
        "source_split": "train",
        "num_source_prompts": num_source_prompts,
        "n_rollouts_per_prompt": n_rollouts_per_prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "output_dir": str(output_dir),
        "variants": {
            "top2": {
                "path": str(top2_jsonl_path),
                "description": "Top-2 correct completions per prompt; shorter completions preferred.",
                "scoring": "correct + shorter completion length",
                **top2_summary,
            },
            "all_correct": {
                "path": str(all_correct_jsonl_path),
                "description": "All correct completions across rollouts per prompt.",
                "scoring": "correct only",
                **all_correct_summary,
            },
        },
    }
    if extra:
        manifest.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {path}")
    return manifest


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
