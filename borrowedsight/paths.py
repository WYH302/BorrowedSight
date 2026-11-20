"""Repository and data locations. Caches never live in the repo."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\bs_data\BorrowedSight")
MINI_ROOT = DATA_ROOT / "mini200"
CACHE_CLEAR = DATA_ROOT / "cache" / "mini200_clear"
CACHE_LISA = DATA_ROOT / "cache" / "mini200_lisa"
CHECKPOINTS = DATA_ROOT / "checkpoints"
CONFIGS = REPO / "configs"
OUTPUTS = REPO / "outputs"
REGISTRY = REPO / "experiments" / "registry.csv"
COHFF_ROOT = REPO / "refs" / "CoHFF"
PYTHON = Path(r"D:\App1\bs_cuda\Scripts\python.exe")
