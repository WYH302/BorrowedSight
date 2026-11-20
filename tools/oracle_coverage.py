"""Phase 2 oracle: no training. Coverage by sender count, range, and LISA weather."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from borrowedsight.data import Mini200
from borrowedsight.grid import INTERVALS, MIN_BOUND, transform_points, voxel_centers, voxel_index
from borrowedsight.paths import CONFIGS, OUTPUTS
from borrowedsight.weather import lisa_fog, lisa_rain, load_weather
from tools.ray_visibility import ray_visibility, union_tristate

BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0))


def _ranges() -> np.ndarray:
    centers = voxel_centers(
        np.stack(np.meshgrid(np.arange(100), np.arange(100), np.arange(8), indexing="ij"), axis=-1).reshape(-1, 3)
    ).reshape(100, 100, 8, 3)
    # centers are already in the ego frame when the grid is ego-centered
    return np.linalg.norm(centers, axis=-1)


def _subset_tri(sample, n_senders: int, mode: str, rng, weather) -> np.ndarray:
    device = torch.device("cuda")
    grids = []
    n = min(int(n_senders), len(sample["points"]))
    for pose, pts, labels in list(zip(sample["sender_pose"], sample["points"], sample["point_sem"]))[:n]:
        pts = np.asarray(pts, dtype=np.float64)
        labels = np.asarray(labels)
        if mode == "fog" and len(pts):
            pts = lisa_fog(pts, rng, weather)
        elif mode == "rain" and len(pts):
            pts, _flags = lisa_rain(pts, rng, weather)
        if len(pts) == 0:
            continue
        ego_pts = transform_points(pts, pose, sample["ego_pose"])
        origin = transform_points(np.zeros((1, 3)), pose, sample["ego_pose"])[0]
        grids.append(
            ray_visibility(
                torch.as_tensor(ego_pts, dtype=torch.float32, device=device),
                torch.as_tensor(origin, dtype=torch.float32, device=device),
            )
        )
    if not grids:
        return np.zeros((100, 100, 8), dtype=np.uint8)
    return union_tristate(grids).detach().cpu().numpy()


def _scores(sample, tri, dist) -> dict:
    gt = sample["gt_semantic"]
    vehicle = gt == 8
    occupied = gt > 0
    rows = {}
    for lo, hi in BINS:
        mask = (dist >= lo) & (dist < hi)
        def ratio(numer, denom):
            den = float(np.count_nonzero(denom))
            if den <= 0:
                return None
            return float(np.count_nonzero(numer)) / den
        rows[f"{int(lo)}_{int(hi)}"] = {
            "occupied_recall": ratio(occupied & mask & (tri == 2), occupied & mask),
            "vehicle_recall": ratio(vehicle & mask & (tri == 2), vehicle & mask),
            "unknown_fraction": ratio(mask & (tri == 0), mask),
        }
    return rows


def main() -> None:
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    weather = load_weather()
    data = Mini200(cfg["root"], "val", int(cfg["val_every"]))
    dist = _ranges()
    rng = np.random.default_rng(0)
    table = []
    # A handful of val frames is the oracle sample. Every frame would repeat the same ray code.
    limit = min(8, len(data))
    for n_senders in (1, 2, 3):
        for mode in ("clear", "fog", "rain"):
            bucket = []
            for index in range(limit):
                sample = data[index]
                tri = _subset_tri(sample, n_senders, mode, rng, weather)
                bucket.append(_scores(sample, tri, dist))
            merged = {}
            for key in bucket[0]:
                merged[key] = {}
                for metric in bucket[0][key]:
                    vals = [row[key][metric] for row in bucket if row[key][metric] is not None]
                    merged[key][metric] = None if not vals else float(np.mean(vals))
            table.append({"n_senders": n_senders, "weather": mode, "bins": merged, "n_frames": limit})
            print(f"oracle senders={n_senders} weather={mode}", flush=True)
    out = OUTPUTS / "phase2_oracle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    # CSV of the numbers that were actually computed.
    lines = ["n_senders,weather,bin,metric,value"]
    for row in table:
        for bin_name, metrics in row["bins"].items():
            for metric, value in metrics.items():
                cell = "" if value is None else f"{value:.8g}"
                lines.append(f"{row['n_senders']},{row['weather']},{bin_name},{metric},{cell}")
    (out / "coverage.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(table, out / "vehicle_recall.png")
    print(f"wrote {out}", flush=True)


def _plot(table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in ("clear", "fog", "rain"):
        xs, ys = [], []
        for row in table:
            if row["weather"] != mode:
                continue
            vals = [row["bins"][k]["vehicle_recall"] for k in row["bins"] if row["bins"][k]["vehicle_recall"] is not None]
            xs.append(row["n_senders"])
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(xs, ys, marker="o", label=mode)
    ax.set_xlabel("senders")
    ax.set_ylabel("vehicle recall")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
