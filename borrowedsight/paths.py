"""Repository and external data locations. Caches never live inside the repo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _resolve_data_root() -> Path:
    env_root = os.environ.get("BORROWEDSIGHT_DATA_ROOT")
    if env_root:
        return Path(env_root)
    local_win = Path(r"D:\bs_data\BorrowedSight")
    if local_win.exists():
        return local_win
    return REPO.parent / "borrowedsight_data"


DATA_ROOT = _resolve_data_root()
MINI_ROOT = DATA_ROOT / "mini200"
CACHE_CLEAR = DATA_ROOT / "cache" / "mini200_clear"
CACHE_LISA = DATA_ROOT / "cache" / "mini200_lisa"
CHECKPOINTS = DATA_ROOT / "checkpoints"
CONFIGS = REPO / "configs"
OUTPUTS = REPO / "outputs"
REGISTRY = REPO / "experiments" / "registry.csv"
COHFF_ROOT = REPO / "refs" / "CoHFF"
PYTHON = Path(sys.executable)
