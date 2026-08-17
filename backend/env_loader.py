from __future__ import annotations

import logging
import os
from pathlib import Path

from backend.runtime_paths import get_project_root


logger = logging.getLogger("viet-toxic-env")

BASE_DIR = get_project_root()

ENV_FILES = [
    BASE_DIR / ".env",
    BASE_DIR / ".env.local",
    BASE_DIR / "backend" / ".env",
    BASE_DIR / "backend" / ".env.local",
]


def _load_env_fallback(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value


def load_env_files() -> bool:
    loaded_any = False

    try:
        from dotenv import load_dotenv

        for env_path in ENV_FILES:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                loaded_any = True

    except ImportError:
        logger.warning("python-dotenv not installed; using basic .env parser fallback")

        for env_path in ENV_FILES:
            if env_path.exists():
                _load_env_fallback(env_path)
                loaded_any = True

    return loaded_any
