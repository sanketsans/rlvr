"""Load project-level environment variables from qwen3_rlvr/.env."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_LOADED = False


def load_project_env() -> Path:
    """Load qwen3_rlvr/.env regardless of the process working directory."""
    global _ENV_LOADED
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    _ENV_LOADED = True
    return env_path
