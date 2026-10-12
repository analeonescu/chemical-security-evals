"""
Loads API keys from a .env file at the repo root, so they don't have to be
exported in every shell.

Inspect's CLI reads .env by itself, but the plain Python entry points in this
repo (score_eval.py, the dataset builders) do not, so they call load_env()
before touching os.environ.

Real values belong in .env, which is gitignored. .env.example is the committed
template and must never contain a real key.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"


def _parse_simple(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser, used only if python-dotenv is unavailable."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


def load_env(path: Path = ENV_PATH, override: bool = False) -> bool:
    """Load .env into os.environ. Returns True if a file was read.

    Variables already set in the environment win, so an explicit `export` still
    overrides the file unless override=True.
    """
    path = Path(path)
    if not path.is_file():
        return False

    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(path, override=override)
        declared = list(dotenv_values(path))
    except ImportError:
        declared = []
        for key, value in _parse_simple(path).items():
            declared.append(key)
            if override or not os.environ.get(key):
                os.environ[key] = value

    # A template copied without editing leaves `KEY=`, which would otherwise
    # register as set-but-blank and produce a confusing auth error downstream.
    for key in declared:
        if os.environ.get(key) == "":
            del os.environ[key]

    return True


def require_key(*names: str) -> str:
    """First non-empty value among `names`, loading .env first."""
    load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    raise SystemExit(
        f"No API key found. Set one of {', '.join(names)} in {ENV_PATH} "
        f"(copy .env.example to .env and fill it in), or export it in your shell."
    )
