# Contributing

Thanks for contributing! All pull requests into `main` run the CI suite below and
**must be green before they can be merged** (enforced via branch protection).

## Local checks

Run the same checks CI runs before you push:

```bash
pip install -e ".[dev]"

ruff check .            # lint (pyflakes, import order, pycodestyle)
ruff format .           # auto-format (use --check in CI)
pytest -q               # unit tests
```

`ruff format .` rewrites files in place; CI runs `ruff format --check .` and fails if
anything is unformatted, so format locally before pushing.

## Tests

Unit tests live in `tests/` and cover the pure, deterministic logic (answer
extraction, advantage computation, Pass@K, data recipes) — no GPU or network needed.
When you add or change behavior in those areas, add a test alongside it. Tests that
require a model or downloads don't belong in this suite.

## Style

`ruff` is configured in `pyproject.toml` (`[tool.ruff]`). It intentionally does **not**
enable pyupgrade/bugbear rules, so existing type annotations and idioms are left as-is —
keep changes focused and avoid unrelated refactors in a PR.
