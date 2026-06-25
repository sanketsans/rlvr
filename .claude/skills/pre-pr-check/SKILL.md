---
name: pre-pr-check
description: Run the qwen3-rlvr pre-PR gate (ruff lint + ruff format check + pytest) before opening or updating a pull request into main. Use this whenever you are about to create a PR, push a branch for review, or the user asks to "check before a PR", "is this ready to merge", or "run the checks".
---

# Pre-PR check (qwen3-rlvr)

Always run this gate **before creating or updating a PR into `main`**. It mirrors
the GitHub Actions CI (`.github/workflows/ci.yml`), which gates every PR via
branch protection — so if this passes locally, CI should be green. Do not open a
PR until all three steps pass. See `CONTRIBUTING.md` for the canonical policy.

## Setup

This is a `src`-layout package; tests import `qwen3_rlvr`, which CI makes
importable via an editable install. Install dev deps the same way CI does:

```bash
pip install -e ".[dev]"
```

If you cannot do an editable install locally, prefix the pytest command with
`PYTHONPATH=src` instead.

## The gate (run all three — same as CI)

```bash
ruff check .            # lint: pyflakes (F), import order (I), pycodestyle (E/W)
ruff format --check .   # formatting must already be applied
pytest -q               # unit tests
```

One-liner:

```bash
ruff check . && ruff format --check . && pytest -q
```

(If not installed editable: `... && PYTHONPATH=src pytest -q`.)

## Fixing failures

- **`ruff check` fails:** `ruff check --fix .` auto-fixes unused imports and
  import ordering. Re-run `ruff check .` to confirm.
- **`ruff format --check` fails:** run `ruff format .` to rewrite files in place,
  then review the diff before committing. CI fails if anything is unformatted.
- **`pytest` fails:** fix the code or, if a test references a refactored-away
  symbol, update the test to the new contract — do not delete a test just to go
  green.

## Config notes (`pyproject.toml` → `[tool.ruff]`)

- `select = ["E", "W", "F", "I"]` — pyupgrade/bugbear are intentionally **off**,
  so leave existing type annotations and idioms as written; keep PR diffs focused.
- `ignore = ["E501", "E402"]` — line length is the formatter's job; `E402` covers
  the intentional `sys.path.insert(...)` pattern in `scripts/`.
- `__init__.py` ignores `F401` (re-exported names).

## After it passes

Only then create the PR. For GitHub operations use `npx -y gh-axi` (per global
tooling preference), e.g. `npx -y gh-axi pr create --base main`.
