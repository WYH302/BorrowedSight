"""Phase 1 BEV checks, plus a strict real-data redraw when a full scene exists."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from borrowedsight.data import simulate_frame
from borrowedsight.grid import transform_points, voxel_index
from borrowedsight.paths import CONFIGS, OUTPUTS


def _save_bev(grid, path: Path, title: str, cmap: str) -> None:
    image = np.max(np.asarray(grid), axis=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.T, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_points(sample, path: Path) -> None:
    clouds = []
    for pose, pts in zip(sample["sender_pose"], sample["points"]):
        if len(pts) == 0:
            continue
        clouds.append(transform_points(pts, pose, sample["ego_pose"]))
    fig, ax = plt.subplots(figsize=(5, 5))
    if clouds:
        pts = np.concatenate(clouds, axis=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c=pts[:, 2], cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title("neighbor points in ego frame")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_alignment(sample, path: Path) -> float:
    sem = sample["gt_semantic"]
    vehicle = np.max(sem == 8, axis=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(vehicle.T, origin="lower", cmap="gray", interpolation="nearest")
    centers = []
    # Vehicle boxes are class 8 in the raster. Mark their voxel centroids.
    xs, ys = np.nonzero(vehicle)
    if len(xs):
        # imshow uses the transposed grid, so the horizontal axis is y.
        ax.scatter(ys[::20], xs[::20], s=4, c="crimson")
    ax.set_title("vehicle voxels")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    # A hit-labeled vehicle point should land on a vehicle voxel.
    hits = 0
    total = 0
    for pose, pts, labels in zip(sample["sender_pose"], sample["points"], sample["point_sem"]):
        if len(pts) == 0:
            continue
        ego_pts = transform_points(pts, pose, sample["ego_pose"])
        labels = np.asarray(labels)
        keep = labels == 8
        if not np.any(keep):
            continue
        idx, valid = voxel_index(ego_pts[keep])
        valid_idx = idx[valid]
        if len(valid_idx) == 0:
            continue
        on = sem[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] == 8
        hits += int(on.sum())
        total += int(len(on))
    return float(hits / total) if total else float("nan")


def render_synthetic(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    a = simulate_frame(0, 10, n_beams=int(cfg["n_beams"]), n_azimuth=int(cfg["n_azimuth"]))
    b = simulate_frame(1, 20, n_beams=int(cfg["n_beams"]), n_azimuth=int(cfg["n_azimuth"]))
    paths = {
        "semantic": root / "01_semantic_scene0.png",
        "tristate": root / "02_tristate_scene0.png",
        "points": root / "03_neighbor_points_scene0.png",
        "vehicles": root / "04_vehicle_voxels_scene0.png",
        "semantic_b": root / "05_semantic_scene1.png",
    }
    _save_bev(a["gt_semantic"], paths["semantic"], "semantic BEV scene 0", "tab20")
    _save_bev(a["gt_tristate"], paths["tristate"], "tri-state BEV scene 0", "viridis")
    _save_points(a, paths["points"])
    align_a = _save_alignment(a, paths["vehicles"])
    _save_bev(b["gt_semantic"], paths["semantic_b"], "semantic BEV scene 1", "tab20")
    return {"paths": {k: str(v) for k, v in paths.items()}, "vehicle_hit_on_vehicle_voxel": align_a}


def find_complete_validate(root: Path) -> Path | None:
    """A scene is complete only when lidar, pose, and semantic exist for >= 10 frames."""
    if not root.exists():
        return None
    for scene in root.rglob("*"):
        if not scene.is_dir():
            continue
        if any(scene.rglob("*.part")) or any(scene.rglob("*.aria2")):
            continue
        yamls = [p for p in scene.rglob("*.yaml") if p.is_file() and p.stat().st_size > 100]
        if len(yamls) < 10:
            continue
        complete = 0
        for yml in yamls:
            stem = yml.with_suffix("")
            lidar = stem.with_suffix(".pcd")
            semantic = None
            for candidate in (
                yml.with_name(stem.name + "_semantic.pcd"),
                yml.with_name(stem.name + "_semantic_label.pcd"),
                yml.with_name("semantic_" + stem.name + ".pcd"),
            ):
                if candidate.exists() and candidate.stat().st_size > 100:
                    semantic = candidate
                    break
            if lidar.exists() and lidar.stat().st_size > 1000 and semantic is not None:
                try:
                    text = yml.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "lidar_pose" in text:
                    complete += 1
        if complete >= 10:
            return scene
    return None


def main() -> None:
    out = OUTPUTS / "phase1_check"
    summary = render_synthetic(out)
    real = find_complete_validate(Path(r"D:\bs_data\BorrowedSight"))
    summary["real_scene"] = None if real is None else str(real)
    summary["real_redraw"] = "skipped_incomplete" if real is None else "not_run"
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
