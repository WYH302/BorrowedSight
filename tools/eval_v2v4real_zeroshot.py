"""Second Real-World Dataset Validation: V2V4Real (CVPR 2023, Real 32-Beam Velodyne VLP-32C LiDAR).

Runs two zero-shot real-road validation audits on real Los Angeles two-CAV sweeps (outside repo at
`D:/bs_data/BorrowedSight/v2v4real/{val,test_01}`):

Audit A (Envelope & Wire Compression Audit on ALL real 32-beam sweeps):
  - Computes 3D ray-cast tri-state grids (`100x100x8`, `40x40x3.2m`) on every genuine 32-beam Velodyne sweep.
  - Compresses via BorrowedSight's exact adaptive 512 B Zstd visibility slot (`50x50+z_max` -> `50x50` -> `25x25`).
  - Measures exact Zstd byte distribution (min/mean/p95/max), tier hit rates, and proves 0% slot overflow.

Audit B (Zero-Shot Sim-to-Real Cooperative Occupancy & Tri-State Transfer, NO V2V4Real training):
  - Evaluates OPV2V-trained BorrowedSight v3.0 models (`B0`, `B3-Compact v3`, `CodeFilling-Occ`, `B4-TDVF v3`,
    `B4-TDVF-MultiBW v3`) directly on overlapping real-road V2V4Real 2-CAV frames (`0` ego LiDAR points!).
  - Evaluates (1) Neighbor-Observable Tri-State Occupancy (`Occ IoU`, `Free IoU`, `Unknown F1`, `UFE`, `Brier`)
    across `{32, 16, 8} KB` exact wire budgets, AND (2) Real-Road 3D Ground-Truth Vehicle Box Occupancy Recall
    & Vehicle Semantic Recall inside V2V4Real's annotated 3D `vehicles` boxes!
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zstandard

from borrowedsight.geometry import geo_stats
from borrowedsight.grid import GRID_SIZE, MAX_BOUND, MIN_BOUND, voxel_centers_torch, voxel_index_torch
from borrowedsight.labels import VEHICLE
from borrowedsight.packer import (
    ENVELOPE_SLOT,
    FAIR_BUDGET_BYTES,
    TARGET_BYTES,
    coarse_from_tristate,
    envelope50_bitplanes,
    pack_hybrid_env50,
    packet_layout,
)
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.ray_visibility import ray_visibility

V2V4REAL_ROOT = DATA_ROOT / "v2v4real"
OUT = OUTPUTS / "real_opv2v" / "v2v4real_zeroshot"


def load_pcd_xyz(path: Path) -> np.ndarray:
    """Fast loader for V2V4Real ASCII or binary .pcd v0.7 files -> [N, 3] float32."""
    with path.open("rb") as f:
        data_mode = "ascii"
        n_pts = 0
        fields = []
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("FIELDS"):
                fields = s.split()[1:]
            elif s.startswith("POINTS"):
                n_pts = int(s.split()[1])
            elif s.startswith("DATA"):
                data_mode = s.split()[1].lower()
                break
        if n_pts == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if data_mode == "ascii":
            raw = f.read().decode("ascii", errors="ignore")
            arr = np.fromstring(raw.replace("\n", " "), sep=" ", dtype=np.float32)
            ncols = max(3, len(fields))
            n_rows = arr.size // ncols
            if n_rows == 0:
                return np.zeros((0, 3), dtype=np.float32)
            pts = arr[: n_rows * ncols].reshape(n_rows, ncols)[:, :3]
            finite = np.isfinite(pts).all(axis=1)
            return np.ascontiguousarray(pts[finite], dtype=np.float32)
        else:
            ncols = max(3, len(fields))
            arr = np.frombuffer(f.read(n_pts * ncols * 4), dtype=np.float32)
            n_rows = arr.size // ncols
            pts = arr[: n_rows * ncols].reshape(n_rows, ncols)[:, :3]
            finite = np.isfinite(pts).all(axis=1)
            return np.ascontiguousarray(pts[finite], dtype=np.float32)


def mat4_to_pose6(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 SE(3) matrix (world_from_lidar) to OPV2V/CARLA 6-DoF [x, y, z, roll, yaw, pitch] in degrees.

    Exact inverse of `borrowedsight.grid.x_to_world`:
      T[2,0] = sin(pitch), T[2,1] = -cos(pitch)*sin(roll), T[2,2] = cos(pitch)*cos(roll),
      T[0,0] = cos(pitch)*cos(yaw), T[1,0] = cos(pitch)*sin(yaw).
    """
    T = np.asarray(T, dtype=np.float64)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    sin_p = np.clip(T[2, 0], -1.0, 1.0)
    pitch = math.asin(float(sin_p))
    cos_p = math.cos(pitch)
    if abs(cos_p) > 1e-6:
        roll = math.atan2(float(-T[2, 1]), float(T[2, 2]))
        yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-T[0, 1]), float(T[1, 1]))
    return np.array([x, y, z, math.degrees(roll), math.degrees(yaw), math.degrees(pitch)], dtype=np.float32)


def discover_v2v4real_pairs(roots: list[Path]) -> list[dict]:
    """Discover all synchronized (CAV 0, CAV 1) frame pairs across extracted V2V4Real splits."""
    pairs = []
    seen_scenes: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        # Support arbitrary nesting (e.g. val/val/<scenario> or test_01/<scenario>)
        cand_dirs = [p.parent for p in sorted(root.rglob("0")) if p.is_dir() and (p.parent / "1").is_dir()]
        for sc in cand_dirs:
            if sc in seen_scenes:
                continue
            seen_scenes.add(sc)
            c0, c1 = sc / "0", sc / "1"
            for y0 in sorted(c0.glob("*.yaml")):
                stem = y0.stem
                p0 = c0 / f"{stem}.pcd"
                y1 = c1 / f"{stem}.yaml"
                p1 = c1 / f"{stem}.pcd"
                if not (p0.exists() and y1.exists() and p1.exists()):
                    continue
                d0 = yaml.load(y0.read_text(encoding="utf-8", errors="ignore"), Loader=yaml.UnsafeLoader)
                d1 = yaml.load(y1.read_text(encoding="utf-8", errors="ignore"), Loader=yaml.UnsafeLoader)
                T0 = np.asarray(d0["lidar_pose"], dtype=np.float64)
                T1 = np.asarray(d1["lidar_pose"], dtype=np.float64)
                rel = np.linalg.inv(T0) @ T1
                dist_xy = float(np.linalg.norm(rel[:2, 3]))
                pairs.append(
                    {
                        "split": root.name,
                        "scene": sc.name,
                        "stem": stem,
                        "pcd0": p0,
                        "pcd1": p1,
                        "T0": T0,
                        "T1": T1,
                        "pose0": mat4_to_pose6(T0),
                        "pose1": mat4_to_pose6(T1),
                        "speed0": float(d0.get("ego_speed", 0.0) or 0.0),
                        "speed1": float(d1.get("ego_speed", 0.0) or 0.0),
                        "vehicles0": d0.get("vehicles") or {},
                        "vehicles1": d1.get("vehicles") or {},
                        "dist_xy": dist_xy,
                    }
                )
    return pairs


def vehicle_box_mask_ego(vehicles: dict, T_ego: np.ndarray) -> np.ndarray:
    """Rasterize V2V4Real ground-truth 3D vehicle bounding boxes onto the 100x100x8 ego voxel grid.

    In V2V4Real CAV-0 YAML, `location` is already in CAV-0 lidar frame (check `abs(loc) < 20`).
    """
    mask = np.zeros(GRID_SIZE, dtype=bool)
    if not vehicles:
        return mask
    # Precompute voxel centers in ego frame
    gx, gy, gz = GRID_SIZE
    ix = torch.arange(gx, dtype=torch.long)
    iy = torch.arange(gy, dtype=torch.long)
    iz = torch.arange(gz, dtype=torch.long)
    X, Y, Z = torch.meshgrid(ix, iy, iz, indexing="ij")
    coords = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=-1)
    centers = voxel_centers_torch(coords).numpy()  # [80000, 3]

    for v in vehicles.values():
        loc = np.asarray(v.get("location", [999.0, 999.0, 0.0]), dtype=np.float64)
        ext = np.asarray(v.get("extent", [0.0, 0.0, 0.0]), dtype=np.float64)
        ang = np.asarray(v.get("angle", [0.0, 0.0, 0.0]), dtype=np.float64)
        ctr = np.asarray(v.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        c_xyz = loc + ctr
        if abs(c_xyz[0]) > 24.0 or abs(c_xyz[1]) > 24.0 or np.any(ext <= 0.1):
            continue
        yaw = math.radians(float(ang[1]))
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        rel = centers - c_xyz[None, :]
        rx = cy * rel[:, 0] - sy * rel[:, 1]
        ry = sy * rel[:, 0] + cy * rel[:, 1]
        rz = rel[:, 2]
        inside = (np.abs(rx) <= ext[0]) & (np.abs(ry) <= ext[1]) & (np.abs(rz) <= ext[2])
        if np.any(inside):
            mask |= inside.reshape(GRID_SIZE)
    return mask


def run_envelope_audit(pairs: list[dict], device: torch.device, max_sweeps: int = 0) -> dict:
    """Audit A: Real 32-beam Velodyne VLP-32C 512 B adaptive Zstd envelope compression proof."""
    c3 = zstandard.ZstdCompressor(level=3)
    c9 = zstandard.ZstdCompressor(level=9)
    z_hyb_sizes = []
    z_50_sizes = []
    z_25_sizes = []
    tier_counts = {"50x50_zmax": 0, "50x50_only": 0, "25x25_zmax": 0, "overflow": 0}
    occ_voxel_counts = []
    pt_counts = []

    sweeps = []
    for pr in pairs:
        sweeps.append(pr["pcd0"])
        sweeps.append(pr["pcd1"])
    if max_sweeps > 0:
        sweeps = sweeps[:max_sweeps]

    t0 = time.perf_counter()
    for si, pcd_path in enumerate(sweeps):
        pts = load_pcd_xyz(pcd_path)
        pt_counts.append(len(pts))
        if len(pts) == 0:
            continue
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
        tri = ray_visibility(pts_t, torch.zeros(3, device=device)).detach().cpu().numpy()
        occ_voxel_counts.append(int((tri == 2).sum()))

        env25 = coarse_from_tristate(tri, cells=25)
        env50 = envelope50_bitplanes(tri)
        hyb = pack_hybrid_env50(env50, env25)

        b_hyb = len(c3.compress(hyb.tobytes()))
        if b_hyb > ENVELOPE_SLOT - 2:
            b_hyb = len(c9.compress(hyb.tobytes()))
        b_50 = len(c3.compress(env50.tobytes()))
        if b_50 > ENVELOPE_SLOT - 2:
            b_50 = len(c9.compress(env50.tobytes()))
        b_25 = len(c3.compress(env25.tobytes()))
        if b_25 > ENVELOPE_SLOT - 2:
            b_25 = len(c9.compress(env25.tobytes()))

        z_hyb_sizes.append(b_hyb)
        z_50_sizes.append(b_50)
        z_25_sizes.append(b_25)

        if b_hyb <= ENVELOPE_SLOT - 2:
            tier_counts["50x50_zmax"] += 1
        elif b_50 <= ENVELOPE_SLOT - 2:
            tier_counts["50x50_only"] += 1
        elif b_25 <= ENVELOPE_SLOT - 2:
            tier_counts["25x25_zmax"] += 1
        else:
            tier_counts["overflow"] += 1

    n = max(1, len(z_hyb_sizes))
    hyb_arr = np.asarray(z_hyb_sizes, dtype=np.float64)
    e50_arr = np.asarray(z_50_sizes, dtype=np.float64)
    e25_arr = np.asarray(z_25_sizes, dtype=np.float64)
    return {
        "n_sweeps": int(len(z_hyb_sizes)),
        "seconds": round(time.perf_counter() - t0, 1),
        "mean_points_per_sweep": round(float(np.mean(pt_counts)), 1) if pt_counts else 0.0,
        "mean_occupied_voxels_40m": round(float(np.mean(occ_voxel_counts)), 1) if occ_voxel_counts else 0.0,
        "zstd_50x50_zmax_bytes": {
            "min": int(hyb_arr.min()), "mean": round(float(hyb_arr.mean()), 1),
            "p95": round(float(np.percentile(hyb_arr, 95)), 1), "max": int(hyb_arr.max()),
        },
        "zstd_50x50_only_bytes": {
            "min": int(e50_arr.min()), "mean": round(float(e50_arr.mean()), 1),
            "p95": round(float(np.percentile(e50_arr, 95)), 1), "max": int(e50_arr.max()),
        },
        "zstd_25x25_zmax_bytes": {
            "min": int(e25_arr.min()), "mean": round(float(e25_arr.mean()), 1),
            "p95": round(float(np.percentile(e25_arr, 95)), 1), "max": int(e25_arr.max()),
        },
        "tier_counts": tier_counts,
        "tier_rates_pct": {k: round(100.0 * v / n, 2) for k, v in tier_counts.items()},
    }


def run_zeroshot_transfer(
    pairs: list[dict],
    ckpt_dir: Path,
    cfg: dict,
    device: torch.device,
    max_dist_m: float = 35.0,
) -> dict:
    """Audit B: Zero-shot Sim-to-Real transfer of OPV2V-trained v3 models on overlapping V2V4Real 2-CAV frames."""
    from tools.eval_phase5_v3 import load_ckpt_v3

    # Build both directions (CAV 1 -> CAV 0 and CAV 0 -> CAV 1) on overlapping pairs (dist <= max_dist_m)
    overlap_pairs = [pr for pr in pairs if pr["dist_xy"] <= max_dist_m]
    print(f"V2V4Real overlapping pairs (dist <= {max_dist_m}m): {len(overlap_pairs)} / {len(pairs)}", flush=True)
    if not overlap_pairs:
        return {"n_overlap_pairs": 0, "note": "No overlapping pairs within max_dist_m"}

    tags = [t for t in ("B0_v3_32k", "Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k", "B4_v3_multibw") if (ckpt_dir / f"{t}_s0.pt").exists()]
    models = {t: load_ckpt_v3(t, 0, cfg, device)[0] for t in tags}
    if not models:
        return {"n_overlap_pairs": len(overlap_pairs), "note": "No v3 checkpoints found yet"}

    # Use frozen v3 SenderEncoder from any loaded model (all share identical sender_opv2v_v3.pt weights)
    ref_model = next(iter(models.values()))

    # Pre-encode sender sweeps once for all overlapping pairs (both directions: 1->0 and 0->1 -> 2x frames!)
    cached_eval_frames = []
    for pr in overlap_pairs:
        pts0 = load_pcd_xyz(pr["pcd0"])
        pts1 = load_pcd_xyz(pr["pcd1"])
        for recv_idx, (ego_pts, ego_T, ego_pose, ego_vehs, nbr_pts, nbr_T, nbr_pose, nbr_spd) in enumerate([
            (pts0, pr["T0"], pr["pose0"], pr["vehicles0"], pts1, pr["T1"], pr["pose1"], pr["speed1"]),
            (pts1, pr["T1"], pr["pose1"], pr["vehicles1"], pts0, pr["T0"], pr["pose0"], pr["speed0"]),
        ]):
            if len(nbr_pts) == 0:
                continue
            # Transform neighbor raw 32-beam points into ego frame using exact 4x4 SE(3) for physical neighbor-observable GT
            T_ego_from_nbr = np.linalg.inv(ego_T) @ nbr_T
            nbr_in_ego = (nbr_pts @ T_ego_from_nbr[:3, :3].T) + T_ego_from_nbr[:3, 3]
            nbr_orig_ego = T_ego_from_nbr[:3, 3].astype(np.float32)

            nbr_in_ego_t = torch.as_tensor(nbr_in_ego, dtype=torch.float32, device=device)
            gt_tri_nbr = ray_visibility(nbr_in_ego_t, torch.as_tensor(nbr_orig_ego, device=device)).detach().cpu().numpy()
            if int((gt_tri_nbr == 2).sum()) < 30:
                continue

            # Encode neighbor sweep in neighbor's local frame (strict Zero-Ego: ego_pts NEVER touched by encoder!)
            nbr_t = torch.as_tensor(nbr_pts, dtype=torch.float32, device=device).reshape(1, -1, 3)
            msk = torch.ones(1, nbr_t.shape[1], dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_s = ref_model.encoder(nbr_t, msk)
            tri_s = ray_visibility(nbr_t[0], torch.zeros(3, device=device))
            coords, stats, _ = geo_stats(nbr_t[0])
            if coords.numel() == 0:
                continue
            feat_map = out_s["feat"][0].float()
            sem_map = out_s["sem"][0].argmax(dim=0)
            feats = feat_map[:, coords[:, 0], coords[:, 1], coords[:, 2]].transpose(0, 1).contiguous()
            preds = sem_map[coords[:, 0], coords[:, 1], coords[:, 2]]
            occ_m = tri_s[coords[:, 0], coords[:, 1], coords[:, 2]] == 2
            coords, stats, feats, preds = coords[occ_m], stats[occ_m], feats[occ_m], preds[occ_m]
            tri_s_np = tri_s.detach().cpu().numpy()

            veh_mask = vehicle_box_mask_ego(ego_vehs, ego_T)
            cached_eval_frames.append(
                {
                    "scene": pr["scene"],
                    "stem": pr["stem"],
                    "recv_idx": recv_idx,
                    "dist_xy": pr["dist_xy"],
                    "ego_pose": ego_pose,
                    "gt_tri_nbr": gt_tri_nbr.astype(np.uint8),
                    "veh_box_mask": veh_mask,
                    "sender": {
                        "sender_id": 1 - recv_idx,
                        "pose": nbr_pose,
                        "timestamp": 0.0,
                        "speed": float(nbr_spd),
                        "yaw_rate": 0.0,
                        "coords": coords.detach().cpu(),
                        "feats": feats.detach().cpu(),
                        "geo": stats.detach().cpu(),
                        "pred_class": preds.detach().cpu(),
                        "envelope": coarse_from_tristate(tri_s_np, cells=20),
                        "envelope25": coarse_from_tristate(tri_s_np, cells=25),
                        "envelope50": envelope50_bitplanes(tri_s_np),
                    },
                }
            )
    print(f"Prepared {len(cached_eval_frames)} overlapping zero-ego V2V4Real directional frames", flush=True)

    results = {"n_frames": len(cached_eval_frames), "max_dist_m": max_dist_m, "budgets": {}}
    for tb in (TARGET_BYTES, FAIR_BUDGET_BYTES, 8192):
        b_res = {}
        for tag, model in models.items():
            bgt = packet_layout(model.method, target=tb, stage=model.default_stage, include_envelope=model.use_envelope, compact=model.compact_wire)
            n_tok = int(bgt["n_tokens"])
            # Confusion counts for tri-state [0=unk, 1=free, 2=occ] + vehicle box recall
            tri_cm = np.zeros((3, 3), dtype=np.int64)
            brier_sum, brier_cnt = 0.0, 0
            veh_obs_occ_total, veh_pred_occ_hit, veh_pred_cls_hit = 0, 0, 0
            sizes = []
            with torch.no_grad():
                for fr in cached_eval_frames:
                    s_row = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fr["sender"].items()}
                    sent = model.transmit(s_row, fr["ego_pose"], fr["recv_idx"], n_tok, budget=bgt, serialize=True)
                    blobs = [sent["blob"]] if sent["blob"] else []
                    if blobs:
                        sizes.append(len(blobs[0]))
                    pose_t = torch.as_tensor(fr["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
                    stamp_t = torch.zeros(1, dtype=torch.float32, device=device)
                    motion_t = torch.zeros(1, 2, dtype=torch.float32, device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(pose_t, stamp_t, motion_t, [blobs])
                    p_sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
                    p_tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
                    p_prob = F.softmax(pred["tri"].float(), dim=1)[0].detach().cpu().numpy()  # [3, X, Y, Z]
                    gt_t = fr["gt_tri_nbr"]
                    # Tri-state confusion matrix
                    idx_cm = gt_t.reshape(-1) * 3 + p_tri.reshape(-1)
                    tri_cm += np.bincount(idx_cm, minlength=9).reshape(3, 3)
                    # Brier
                    onehot = np.eye(3, dtype=np.float32)[gt_t].transpose(3, 0, 1, 2)
                    brier_sum += float(np.sum((p_prob - onehot) ** 2))
                    brier_cnt += int(gt_t.size)
                    # Vehicle 3D box recall inside neighbor-observed occupied vehicle voxels
                    vm = fr["veh_box_mask"] & (gt_t == 2)
                    if np.any(vm):
                        veh_obs_occ_total += int(vm.sum())
                        veh_pred_occ_hit += int((p_tri[vm] == 2).sum())
                        veh_pred_cls_hit += int((p_sem[vm] == VEHICLE).sum())

            def _iou(c):
                tp = tri_cm[c, c]
                denom = tri_cm[c, :].sum() + tri_cm[:, c].sum() - tp
                return float(tp / max(1, denom))

            unk_tp = tri_cm[0, 0]
            unk_f1 = float(2 * unk_tp / max(1, tri_cm[0, :].sum() + tri_cm[:, 0].sum()))
            ufe = float(tri_cm[0, 1] / max(1, tri_cm[0, :].sum()))
            b_res[tag] = {
                "bytes_mean": round(float(np.mean(sizes)), 1) if sizes else 0.0,
                "occ_IoU": round(_iou(2), 4),
                "free_IoU": round(_iou(1), 4),
                "tri_mIoU": round((_iou(0) + _iou(1) + _iou(2)) / 3.0, 4),
                "unknown_F1": round(unk_f1, 4),
                "UFE": round(ufe, 4),
                "tri_brier": round(brier_sum / max(1, brier_cnt), 4),
                "veh_box_occ_recall": round(veh_pred_occ_hit / max(1, veh_obs_occ_total), 4),
                "veh_box_sem_recall": round(veh_pred_cls_hit / max(1, veh_obs_occ_total), 4),
                "veh_obs_voxels": int(veh_obs_occ_total),
            }
            print(
                f"  [V2V4Real {tb:5d}B] {tag:18s}: occIoU={b_res[tag]['occ_IoU']:.4f} freeIoU={b_res[tag]['free_IoU']:.4f} "
                f"unkF1={b_res[tag]['unknown_F1']:.4f} UFE={b_res[tag]['UFE']:.4f} "
                f"vehOccRec={b_res[tag]['veh_box_occ_recall']:.4f} vehSemRec={b_res[tag]['veh_box_sem_recall']:.4f}",
                flush=True,
            )
        results["budgets"][str(tb)] = b_res
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope-only", action="store_true")
    ap.add_argument("--max-dist-m", type=float, default=35.0)
    ap.add_argument("--ckpt-dir", type=str, default="real_opv2v_v3")
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    roots = [V2V4REAL_ROOT / "val", V2V4REAL_ROOT / "test_01"]
    pairs = discover_v2v4real_pairs(roots)
    print(f"Discovered {len(pairs)} synchronized real 2-CAV V2V4Real frames across {[r.name for r in roots if r.exists()]}", flush=True)
    dists = np.array([p["dist_xy"] for p in pairs]) if pairs else np.zeros(0)
    if len(dists) > 0:
        print(
            f"  Inter-CAV dist: min={dists.min():.1f}m med={np.median(dists):.1f}m max={dists.max():.1f}m | "
            f"<=25m: {(dists<=25).sum()} | <=35m: {(dists<=35).sum()} | <=40m: {(dists<=40).sum()}",
            flush=True,
        )

    env_audit = run_envelope_audit(pairs, device)
    print("Audit A (Real 32-Beam Velodyne 512B Zstd Envelope):", json.dumps(env_audit, indent=2), flush=True)

    transfer = None
    if not args.envelope_only:
        transfer = run_zeroshot_transfer(pairs, CHECKPOINTS / args.ckpt_dir, cfg, device, max_dist_m=args.max_dist_m)

    summary = {
        "n_pairs_total": len(pairs),
        "splits": sorted({p["split"] for p in pairs}),
        "scenes": sorted({p["scene"] for p in pairs}),
        "envelope_audit": env_audit,
        "zeroshot_transfer": transfer,
    }
    (OUT / "v2v4real_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved V2V4Real report -> {OUT / 'v2v4real_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
