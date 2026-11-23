"""Verify Phase 1 real-data alignment on Semantic-OPV2V 4LidarSurround + raw OPV2V.

Builds exact CoHFF semantic voxels for validate/2021_08_23_13_17_21, checks that
neighbor raw OPV2V lidar points and YAML GT vehicle boxes sit directly on
Semantic-OPV2V class-8 (Vehicles) voxels, and renders 5 real validate BEV frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from borrowedsight.grid import GRID_SIZE, transform_points, voxel_index, x_to_world
from borrowedsight.labels import VEHICLE, load_cohff_name_table
from borrowedsight.paths import DATA_ROOT, OUTPUTS
from borrowedsight.semantic_vox import build_scene_voxels, parse_raw_pcd_xyz
from tools.ray_visibility import ray_visibility, union_tristate

OPV2V = DATA_ROOT / "OPV2V"
SEM_RAW = DATA_ROOT / "SemanticOPV2V_4LidarSurround"
VOX_ROOT = DATA_ROOT / "semantic_voxels"
OUT = OUTPUTS / "phase1_check_real_semantic"


def main() -> None:
    scene = "2021_08_23_13_17_21"
    raw_scene = OPV2V / "validate" / scene
    sem_scene = SEM_RAW / "validate" / scene
    out_npz = VOX_ROOT / "validate" / f"{scene}.npz"

    print(f"Building exact CoHFF semantic voxels for validate/{scene}...", flush=True)
    info = build_scene_voxels(raw_scene, out_npz, semantic_scene_dir=sem_scene)
    print(f"Built {out_npz}: {info}", flush=True)

    blob = np.load(out_npz, allow_pickle=False)
    cav_ids = [str(c) for c in blob["cav_ids"]]
    stems = [str(s) for s in blob["stems"]]
    poses = blob["poses"]
    collab_sem = blob["collab_sem"]
    names = load_cohff_name_table()

    # Class voxel histogram across the real scene
    u, c = np.unique(collab_sem, return_counts=True)
    class_counts = {f"{int(k)}_{names.get(int(k), 'Unlabeled')}": int(cnt) for k, cnt in zip(u, c)}
    print("Real scene mapped class voxel counts:", json.dumps(class_counts, indent=2), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    pick_indices = np.linspace(0, len(stems) - 1, 5, dtype=int).tolist()

    box_on_veh_hits = 0
    box_total = 0
    neighbor_pts_on_occ_hits = 0
    neighbor_pts_total = 0
    frame_reports = []

    for rank_i, fi in enumerate(pick_indices):
        stem = stems[fi]
        ego_pose = poses[fi, 0]
        inv_ego = np.linalg.inv(x_to_world(ego_pose))
        sem = collab_sem[fi]

        # Gather neighbor-only raw OPV2V lidar points (ZERO EGO sensor!)
        neighbor_pts_ego = []
        tri_grids = []
        for ki in range(1, len(cav_ids)):
            pts_s = parse_raw_pcd_xyz(raw_scene / cav_ids[ki] / f"{stem}.pcd")
            pts_e = transform_points(pts_s, poses[fi, ki], ego_pose)
            origin_e = transform_points(np.zeros((1, 3)), poses[fi, ki], ego_pose)[0]
            neighbor_pts_ego.append(pts_e)
            tri_grids.append(
                ray_visibility(
                    torch.as_tensor(pts_e, dtype=torch.float32, device=device),
                    torch.as_tensor(origin_e, dtype=torch.float32, device=device),
                )
            )
        tri = union_tristate(tri_grids).numpy()
        pts_all = np.concatenate(neighbor_pts_ego, axis=0) if neighbor_pts_ego else np.zeros((0, 3))
        idx_p, val_p = voxel_index(pts_all)
        idx_v = idx_p[val_p]
        if len(idx_v):
            on_sem = sem[idx_v[:, 0], idx_v[:, 1], idx_v[:, 2]] > 0
            neighbor_pts_on_occ_hits += int(on_sem.sum())
            neighbor_pts_total += int(len(idx_v))
            pt_align = float(on_sem.mean())
        else:
            pt_align = float("nan")

        # Load YAML GT vehicle centers and check whether they sit on Semantic class-8 (Vehicles) voxels
        with (raw_scene / cav_ids[0] / f"{stem}.yaml").open("r", encoding="utf-8") as h:
            meta0 = yaml.safe_load(h)
        veh_xy = []
        frame_hits = 0
        frame_boxes = 0
        for vid, v in (meta0.get("vehicles") or {}).items():
            loc = v.get("location")
            ext = v.get("extent") or [0, 0, 0]
            if not isinstance(loc, (list, tuple)) or len(loc) < 3:
                continue
            pw = np.array([float(loc[0]), float(loc[1]), float(loc[2]) + float(ext[2]) * 0.5, 1.0])
            pe = (inv_ego @ pw)[:3]
            if abs(pe[0]) >= 19.5 or abs(pe[1]) >= 19.5:
                continue
            veh_xy.append(pe)
            g_idx, g_ok = voxel_index(pe.reshape(1, 3))
            if bool(g_ok[0]):
                ix, iy = int(g_idx[0, 0]), int(g_idx[0, 1])
                x0, x1 = max(0, ix - 3), min(GRID_SIZE[0], ix + 4)
                y0, y1 = max(0, iy - 3), min(GRID_SIZE[1], iy + 4)
                hit = bool(np.any(sem[x0:x1, y0:y1, :] == VEHICLE))
                frame_hits += int(hit)
                frame_boxes += 1
        box_on_veh_hits += frame_hits
        box_total += frame_boxes
        frame_reports.append(
            {
                "frame": stem,
                "yaml_vehicles_in_grid": frame_boxes,
                "yaml_vehicles_on_class8_voxel": frame_hits,
                "neighbor_lidar_on_semantic_occ": pt_align,
                "tristate_unknown_frac": float((tri == 0).mean()),
                "tristate_free_frac": float((tri == 1).mean()),
                "tristate_occ_frac": float((tri == 2).mean()),
            }
        )

        # Render 3-panel audit figure
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
        # Panel 1: Semantic BEV (highest non-zero class priority: Vehicle on top)
        sem_bev = np.max(sem, axis=2)
        veh_bev = np.any(sem == VEHICLE, axis=2)
        sem_bev = np.where(veh_bev, VEHICLE, sem_bev)
        axes[0].imshow(sem_bev.T, origin="lower", cmap="tab20", vmin=0, vmax=12, interpolation="nearest")
        if veh_xy:
            va = np.stack(veh_xy, axis=0)
            g_idx, g_ok = voxel_index(va)
            axes[0].scatter(g_idx[g_ok, 0], g_idx[g_ok, 1], s=45, c="red", marker="x", linewidths=1.5, label="YAML boxes")
        axes[0].scatter([50], [50], s=45, c="black", marker="^", label="EGO")
        axes[0].set_title(f"1. CoHFF Semantic GT (Vehicles=8) {frame_hits}/{frame_boxes}")
        axes[0].legend(loc="upper right", fontsize=7)

        # Panel 2: Semantic Vehicle voxels + Neighbor-only raw LiDAR points
        axes[1].imshow(veh_bev.T, origin="lower", cmap="gray_r", interpolation="nearest")
        if len(idx_v):
            sub = idx_v[:: max(1, len(idx_v) // 4000)]
            axes[1].scatter(sub[:, 0], sub[:, 1], s=1.0, c="#1f77b4", alpha=0.55, label="Neighbor LiDAR")
        if veh_xy:
            va = np.stack(veh_xy, axis=0)
            g_idx, g_ok = voxel_index(va)
            axes[1].scatter(g_idx[g_ok, 0], g_idx[g_ok, 1], s=45, c="crimson", marker="x", linewidths=1.5, label="YAML boxes")
        axes[1].set_title(f"2. Neighbor LiDAR on Sem-Veh (align={pt_align:.3f})")
        axes[1].legend(loc="upper right", fontsize=7)

        # Panel 3: Neighbor-only Tri-State Visibility BEV (0=Unknown, 1=Free, 2=Occ)
        tri_bev = np.max(tri, axis=2)
        axes[2].imshow(tri_bev.T, origin="lower", cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        axes[2].set_title(f"3. Neighbor Tri-State (Unk={float((tri==0).mean()):.2f})")

        for ax in axes:
            ax.set_xlabel("x voxel")
            ax.set_ylabel("y voxel")
        fig.suptitle(f"Real OPV2V + Semantic-OPV2V 4LidarSurround Audit: validate/{scene} #{stem}")
        fig.savefig(OUT / f"{rank_i:02d}_{scene}_{stem}.png", dpi=150)
        plt.close(fig)

    summary = {
        "scene": scene,
        "cav_ids": cav_ids,
        "total_frames": len(stems),
        "npz_bytes": info["npz_bytes"],
        "yaml_vehicles_on_semantic_vehicle_voxels": f"{box_on_veh_hits}/{box_total}",
        "yaml_vehicle_alignment_rate": float(box_on_veh_hits / max(1, box_total)),
        "neighbor_lidar_on_semantic_occ_rate": float(neighbor_pts_on_occ_hits / max(1, neighbor_pts_total)),
        "class_voxel_counts": class_counts,
        "frames": frame_reports,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
