"""Append-only experiment registry. Unmeasured cells stay blank."""

from __future__ import annotations

import csv
import subprocess
from datetime import date
from pathlib import Path

from borrowedsight.metrics import mbps_from_bytes
from borrowedsight.paths import REGISTRY, REPO

COLUMNS = [
    "exp_id",
    "date",
    "git_commit",
    "model_name",
    "seed",
    "bytes_per_frame",
    "mbps_10hz",
    "mIoU",
    "vehicle_IoU",
    "road_IoU",
    "unknown_F1",
    "unobserved_as_free_err",
    "status",
]


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit"


def _ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return
    if rows[0] == COLUMNS:
        return
    if len(rows) == 1:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return
    raise RuntimeError(f"registry header is {rows[0]}, expected {COLUMNS}")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def log_run(
    *,
    exp_id: str,
    model_name: str,
    seed: int,
    bytes_per_frame: float | None,
    metrics: dict | None,
    status: str,
    when: str | None = None,
    commit: str | None = None,
) -> dict:
    metrics = metrics or {}
    row = {
        "exp_id": exp_id,
        "date": when or date.today().isoformat(),
        "git_commit": commit or git_commit(),
        "model_name": model_name,
        "seed": int(seed),
        "bytes_per_frame": None if bytes_per_frame is None else float(bytes_per_frame),
        "mbps_10hz": None if bytes_per_frame is None else mbps_from_bytes(bytes_per_frame),
        "mIoU": metrics.get("mIoU"),
        "vehicle_IoU": metrics.get("vehicle_IoU"),
        "road_IoU": metrics.get("road_IoU"),
        "unknown_F1": metrics.get("unknown_F1"),
        "unobserved_as_free_err": metrics.get("unobserved_as_free_err"),
        "status": status,
    }
    _ensure_header(REGISTRY)
    with REGISTRY.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writerow({key: _fmt(row[key]) for key in COLUMNS})
    return row
