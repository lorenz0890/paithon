import os
from pathlib import Path

from paithon import OpenAIProvider, RuntimeConfig, RuntimeEngine


def _cache_dir() -> Path:
    return Path(os.environ.get("PAITHON_CACHE_DIR", ".paithon_cache/myproj"))


def _operating_mode() -> str:
    return os.environ.get("PAITHON_OPERATING_MODE", "review_first")


def _max_heal_attempts() -> int:
    return int(os.environ.get("PAITHON_MAX_HEAL_ATTEMPTS", "1"))


engine = RuntimeEngine(
    provider=OpenAIProvider(),
    config=RuntimeConfig(
        cache_dir=_cache_dir(),
        operating_mode=_operating_mode(),
        max_heal_attempts=_max_heal_attempts(),
    ),
)
