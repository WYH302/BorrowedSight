"""Real OPV2V lidar oracle: coverage, unknown fraction, and 3D box recall.

Uses borrowedsight.grid (official CoHFF/OpenCOOD degree-based CARLA poses)
and borrowedsight.weather (disjoint LISA fog/rain), stratified by sender count,
LISA weather, range bin, and traffic density (sparse/medium/dense).
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from borrowedsight.grid import GRID_SIZE, INTERVALS, MIN_BOUND, transform_points, voxel_centers, voxel_index, x_to_world
from borrowedsight.weather import lisa_fog, lisa_rain, load_weather
from tools.ray_visibility import ray_visibility, union_tristate

from borrowedsight.paths import DATA_ROOT, OUTPUTS

ROOT = DATA_ROOT / "OPV2V" / "validate"
OUT = OUTPUTS / "phase2_oracle_real"
VIS = OUTPUTS / "phase1_check_real"
OUT.mkdir(parents=True, exist_ok=True)
VIS.mkdir(parents=True, exist_ok=True)

NX, NY, NZ = GRID_SIZE
MAX_FRAMES = 90


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            free_b, _ = torch.cuda.mem_get_info()
            if free_b > 2 * 1024**3:
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def read_pcd_xyz(path: Path, max_points: int = 120000) -> np.ndarray:
    with path.open("rb") as f:
        header = []
        data_type = "binary"
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", errors="ignore").strip()
            header.append(s)
            if s.startswith("DATA"):
                data_type = s.split()[-1].lower()
                break
        fields, sizes, types, counts = [], [], [], []
        width, npoints = 0, 0
        for s in header:
            if s.startswith("FIELDS"):
                fields = s.split()[1:]
            elif s.startswith("SIZE"):
                sizes = [int(v) for v in s.split()[1:]]
            elif s.startswith("TYPE"):
                types = s.split()[1:]
            elif s.startswith("COUNT"):
                counts = [int(v) for v in s.split()[1:]]
            elif s.startswith("WIDTH"):
                width = int(s.split()[1])
            elif s.startswith("POINTS"):
                npoints = int(s.split()[1])
        if not counts:
            counts = [1] * len(fields)
        if data_type == "ascii":
            raw = np.loadtxt(f, dtype=np.float64, max_rows=max_points)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            return raw[:, :3] if raw.size else np.zeros((0, 3), dtype=np.float64)
        dtype_map = {
            "F": {4: "<f4", 8: "<f8"},
            "U": {1: "u1", 2: "<u2", 4: "<u4"},
            "I": {1: "i1", 2: "<i2", 4: "<i4"},
        }
        dt = []
        for name, sz, tp, ct in zip(fields, sizes, types, counts):
            base = dtype_map.get(tp, {}).get(sz, "<f4")
            dt.append((name, base) if ct == 1 else (name, base, (ct,)))
        arr = np.fromfile(f, dtype=np.dtype(dt), count=min(npoints or width, max_points))
        if arr.size == 0 or "x" not in arr.dtype.names:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def vehicle_centers_ego(metas: list[dict], ego_pose: np.ndarray) -> np.ndarray:
    """Deduplicated 3D vehicle centers in EGO lidar frame using exact CoHFF x_to_world."""
    inv_ego = np.linalg.inv(x_to_world(ego_pose))
    seen: dict[str, np.ndarray] = {}
    for meta in metas:
        vehicles = meta.get("vehicles") or {}
        if isinstance(vehicles, list):
            items = [(str(i), v) for i, v in enumerate(vehicles)]
        else:
            items = [(str(k), v) for k, v in vehicles.items()]
        for vid, v in items:
            if vid in seen or not isinstance(v, dict):
                continue
            loc = v.get("location")
            ext = v.get("extent") or [0.0, 0.0, 0.0]
            if not isinstance(loc, (list, tuple)) or len(loc) < 3:
                continue
            # Vehicle center in world: location + z-extent (matches CoHFF opv2v_dataset.py)
            pw = np.array([float(loc[0]), float(loc[1]), float(loc[2]) + float(ext[2]) * 0.5, 1.0], dtype=np.float64)
            pe = (inv_ego @ pw)[:3]
            if abs(pe[0]) < 20.0 and abs(pe[1]) < 20.0:
                seen[vid] = pe
    if not seen:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(list(seen.values()), axis=0)


def voxel_range_masks() -> dict[str, np.ndarray]:
    coords = np.stack(
        np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centers = voxel_centers(coords).reshape(NX, NY, NZ, 3)
    r = np.hypot(centers[..., 0], centers[..., 1])
    return {
        "0-10": (r >= 0.0) & (r < 10.0),
        "10-20": (r >= 10.0) & (r < 20.0),
        "20-40": (r >= 20.0) & (r < 40.0),
    }


def density_bucket(n_vehicles: int) -> str:
    if n_vehicles <= 4:
        return "sparse"
    if n_vehicles <= 8:
        return "medium"
    return "dense"


def main() -> None:
    device = pick_device()
    weather_cfg = load_weather()
    bin_masks = voxel_range_masks()
    rng = np.random.default_rng(0)
    print(f"Real OPV2V oracle running on device={device}", flush=True)

    scenes = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    rows = []
    vis_saved = 0
    n_frames = 0
    per_scene_cap = max(1, MAX_FRAMES // max(1, len(scenes)))

    for scene in scenes:
        cavs = sorted(
            [d for d in scene.iterdir() if d.is_dir()],
            key=lambda x: (int(x.name) < 0, int(x.name)) if x.name.lstrip("-").isdigit() else (False, x.name),
        )
        if len(cavs) < 2:
            continue
        stem_sets = [
            {p.stem for p in cav.glob("*.yaml") if p.stem.isdigit() and (cav / f"{p.stem}.pcd").exists()}
            for cav in cavs
        ]
        common = sorted(set.intersection(*stem_sets)) if stem_sets else []
        if not common:
            continue
        step = max(1, len(common) // per_scene_cap)
        sample_stems = common[::step][:per_scene_cap]

        for stem in sample_stems:
            metas, poses, pcs = [], [], []
            ok = True
            for cav in cavs:
                try:
                    meta = load_yaml(cav / f"{stem}.yaml")
                    pose = np.asarray(meta["lidar_pose"], dtype=np.float64)
                    pts = read_pcd_xyz(cav / f"{stem}.pcd")
                except Exception:
                    ok = False
                    break
                metas.append(meta)
                poses.append(pose)
                pcs.append(pts)
            if not ok:
                continue

            # Official CoHFF EGO = cavs[0]; senders = nearest neighbors (excluding EGO!).
            ego_idx = 0
            ego_pose = poses[ego_idx]
            dists = [
                (float(np.linalg.norm(poses[i][:2] - ego_pose[:2])), i)
                for i in range(len(cavs))
                if i != ego_idx
            ]
            dists.sort()
            senders = [i for _, i in dists]

            gt_centers = vehicle_centers_ego(metas, ego_pose)
            # Exclude ego itself (~within 1.0m of origin) from neighbor target recall
            if len(gt_centers):
                gt_centers = gt_centers[np.linalg.norm(gt_centers[:, :2], axis=1) > 1.2]
            dens = density_bucket(len(gt_centers))

            for mode in ("clear", "fog", "rain"):
                # Precompute degraded sender point clouds in EGO frame once per mode
                sender_grids = []
                bev_pts_all = []
                for si in senders[:3]:
                    pts_s = pcs[si]
                    if mode == "fog" and len(pts_s):
                        pts_s = lisa_fog(pts_s, rng, weather_cfg)
                    elif mode == "rain" and len(pts_s):
                        pts_s, _ = lisa_rain(pts_s, rng, weather_cfg)
                    if len(pts_s) == 0:
                        sender_grids.append(torch.zeros(GRID_SIZE, dtype=torch.uint8, device=device))
                        continue
                    pts_e = transform_points(pts_s, poses[si], ego_pose)
                    origin_e = transform_points(np.zeros((1, 3)), poses[si], ego_pose)[0]
                    # Subsample very dense sweeps mildly only if on CPU? Keep full accuracy!
                    grid = ray_visibility(
                        torch.as_tensor(pts_e, dtype=torch.float32, device=device),
                        torch.as_tensor(origin_e, dtype=torch.float32, device=device),
                    )
                    sender_grids.append(grid)
                    if mode == "clear":
                        in_box = (
                            (pts_e[:, 0] > -20)
                            & (pts_e[:, 0] < 20)
                            & (pts_e[:, 1] > -20)
                            & (pts_e[:, 1] < 20)
                            & (pts_e[:, 2] > -2.5)
                            & (pts_e[:, 2] < 0.9)
                        )
                        if np.any(in_box):
                            bev_pts_all.append(pts_e[in_box][:: max(1, int(in_box.sum()) // 3000), :2])

                for n_send in (1, 2, 3):
                    if n_send > len(sender_grids):
                        continue
                    tri = union_tristate(sender_grids[:n_send]).detach().cpu().numpy()
                    obs = tri > 0
                    occ = tri == 2
                    cov = float(obs.mean())
                    unk = float((tri == 0).mean())
                    hit_gt = 0
                    if len(gt_centers):
                        g_idx, g_val = voxel_index(gt_centers)
                        for idx3, ok_v in zip(g_idx, g_val):
                            if not ok_v:
                                continue
                            x0, x1 = max(0, idx3[0] - 2), min(NX, idx3[0] + 3)
                            y0, y1 = max(0, idx3[1] - 2), min(NY, idx3[1] + 3)
                            if np.any(occ[x0:x1, y0:y1, :]):
                                hit_gt += 1
                        recall = hit_gt / float(len(gt_centers))
                    else:
                        recall = float("nan")
                    row = {
                        "scene": scene.name,
                        "stem": stem,
                        "weather": mode,
                        "density": dens,
                        "n_senders": n_send,
                        "coverage": cov,
                        "unknown": unk,
                        "box_recall": recall,
                        "n_gt_boxes": len(gt_centers),
                        "cov_0_10": float(obs[bin_masks["0-10"]].mean()),
                        "cov_10_20": float(obs[bin_masks["10-20"]].mean()),
                        "cov_20_40": float(obs[bin_masks["20-40"]].mean()),
                    }
                    rows.append(row)

                    if mode == "clear" and vis_saved < 5 and n_send == min(3, len(sender_grids)) and bev_pts_all:
                        fig, ax = plt.subplots(figsize=(5.5, 5.5))
                        pts2 = np.concatenate(bev_pts_all, axis=0)
                        ax.scatter(pts2[:, 0], pts2[:, 1], s=0.4, c="#1f77b4", alpha=0.45, label="Neighbor LiDAR")
                        ax.scatter([0.0], [0.0], s=60, c="black", marker="^", label="EGO (no sensor)")
                        if len(gt_centers):
                            ax.scatter(gt_centers[:, 0], gt_centers[:, 1], s=35, c="crimson", marker="x", label="GT vehicles")
                        ax.set_xlim(-20, 20)
                        ax.set_ylim(-20, 20)
                        ax.set_aspect("equal")
                        ax.legend(loc="upper right", fontsize=7)
                        ax.set_title(f"{scene.name} #{stem} (senders={n_send}, CoHFF deg)")
                        fig.tight_layout()
                        fig.savefig(VIS / f"{vis_saved:02d}_{scene.name}_{stem}.png", dpi=140)
                        plt.close(fig)
                        vis_saved += 1
            n_frames += 1
            if n_frames % 10 == 0 or n_frames == 1:
                print(f"frame {n_frames}/{MAX_FRAMES} {scene.name} {stem} gt_veh={len(gt_centers)} ({dens})", flush=True)

    csv_path = OUT / "coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate summary by (weather, n_senders) and by (density, n_senders)
    lines = [
        "# Real OPV2V Oracle (Exact CoHFF Degree Poses + LISA Weather + Congestion)",
        "",
        f"Frames processed: {n_frames} across {len(scenes)} validate scenes. Total rows: {len(rows)}.",
        "Vehicle metric is 3D box occupied-hit recall from OPV2V YAML (exact degree-based x_to_world).",
        "",
        "## By Weather and Senders",
        "",
        "| weather | senders | n | coverage | unknown | box_recall | cov_0_10 | cov_10_20 | cov_20_40 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("clear", "fog", "rain"):
        for n_send in (1, 2, 3):
            sub = [r for r in rows if r["weather"] == mode and r["n_senders"] == n_send]
            if not sub:
                continue
            rec = [r["box_recall"] for r in sub if not math.isnan(r["box_recall"])]
            lines.append(
                f"| {mode} | {n_send} | {len(sub)} | "
                f"{np.mean([r['coverage'] for r in sub]):.4f} | "
                f"{np.mean([r['unknown'] for r in sub]):.4f} | "
                f"{(np.mean(rec) if rec else float('nan')):.4f} | "
                f"{np.mean([r['cov_0_10'] for r in sub]):.4f} | "
                f"{np.mean([r['cov_10_20'] for r in sub]):.4f} | "
                f"{np.mean([r['cov_20_40'] for r in sub]):.4f} |"
            )
    lines.extend([
        "",
        "## By Traffic Density (Clear)",
        "",
        "| density | senders | n | coverage | unknown | box_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dens in ("sparse", "medium", "dense"):
        for n_send in (1, 2, 3):
            sub = [r for r in rows if r["weather"] == "clear" and r["density"] == dens and r["n_senders"] == n_send]
            if not sub:
                continue
            rec = [r["box_recall"] for r in sub if not math.isnan(r["box_recall"])]
            lines.append(
                f"| {dens} | {n_send} | {len(sub)} | "
                f"{np.mean([r['coverage'] for r in sub]):.4f} | "
                f"{np.mean([r['unknown'] for r in sub]):.4f} | "
                f"{(np.mean(rec) if rec else float('nan')):.4f} |"
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    for mode, color in (("clear", "#1f77b4"), ("fog", "#ff7f0e"), ("rain", "#2ca02c")):
        xs, yc, yr = [], [], []
        for s in (1, 2, 3):
            sub = [r for r in rows if r["weather"] == mode and r["n_senders"] == s]
            if not sub:
                continue
            xs.append(s)
            yc.append(float(np.mean([r["coverage"] for r in sub])))
            rec = [r["box_recall"] for r in sub if not math.isnan(r["box_recall"])]
            yr.append(float(np.mean(rec)) if rec else float("nan"))
        axes[0].plot(xs, yc, "-o", color=color, label=mode)
        axes[1].plot(xs, yr, "-o", color=color, label=mode)
    axes[0].set_xlabel("senders")
    axes[0].set_ylabel("observed coverage")
    axes[1].set_xlabel("senders")
    axes[1].set_ylabel("3D box occupied recall")
    for ax in axes:
        ax.set_xticks([1, 2, 3])
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Real OPV2V Oracle (Exact CoHFF Coordinates + LISA Weather)")
    fig.savefig(OUT / "coverage.png", dpi=160)
    plt.close(fig)
    print(f"WROTE {csv_path} ({len(rows)} rows, {vis_saved} BEV checks)", flush=True)


if __name__ == "__main__":
    main()
