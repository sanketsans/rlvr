"""JSONL artifacts for training samples and offline visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def training_stage(step: int, max_steps: int) -> str:
    if max_steps <= 0:
        return "early"
    pct = step / max_steps
    if pct < 0.1:
        return "early"
    if pct < 0.5:
        return "mid"
    return "late"


class SampleLogger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_many(self, records: Iterable[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
