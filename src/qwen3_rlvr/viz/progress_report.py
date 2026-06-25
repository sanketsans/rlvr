"""Build HTML progress report from training samples.jsonl."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_samples(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_progress_report(samples_path: Path, output_path: Path, per_stage: int = 6) -> Path:
    samples = load_samples(samples_path)
    by_stage: Dict[str, List[dict]] = defaultdict(list)
    for row in samples:
        by_stage[row.get("stage", "unknown")].append(row)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>GRPO Training Progress</title>",
        "<style>",
        "body{font-family:sans-serif;margin:2rem;background:#111;color:#eee}",
        "h1,h2{color:#7dd3fc}",
        ".card{border:1px solid #333;border-radius:8px;padding:1rem;margin:1rem 0;background:#1a1a1a}",
        ".q{color:#a5f3fc}.gt{color:#86efac}.bad{color:#fca5a5}.good{color:#86efac}",
        "pre{white-space:pre-wrap;background:#0a0a0a;padding:0.75rem;border-radius:6px}",
        "</style></head><body>",
        f"<h1>GRPO progress report</h1><p>Source: {html.escape(str(samples_path))}</p>",
    ]

    for stage in ("early", "mid", "late"):
        rows = by_stage.get(stage, [])[:per_stage]
        parts.append(f"<h2>{stage.upper()} ({len(by_stage.get(stage, []))} logged)</h2>")
        if not rows:
            parts.append("<p>No samples for this stage.</p>")
            continue
        for row in rows:
            reward = row.get("reward", 0)
            reward_cls = "good" if reward >= 1 else "bad"
            parts.append("<div class='card'>")
            parts.append(
                f"<div>step {row.get('step')} · reward <span class='{reward_cls}'>{reward}</span></div>"
            )
            parts.append(
                f"<div class='q'><b>Q:</b> {html.escape(str(row.get('question', '')))}</div>"
            )
            parts.append(
                f"<div class='gt'><b>GT:</b> {html.escape(str(row.get('ground_truth', '')))}</div>"
            )
            parts.append(f"<pre>{html.escape(str(row.get('completion', '')))}</pre>")
            parts.append("</div>")

    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
