"""Zero-Byte Semantic-Selective Kinematic Forward Warping (SS-KFW) for V2X Latency Compensation.

Exclusively unlocked by BorrowedSight's wire-level 4-bit semantic class anchor (`c_u == VEHICLE`):
- Static scene anchors (`c_u != VEHICLE`) remain locked to exact world SE(3) coordinates.
- Dynamic vehicle anchors (`c_u == VEHICLE`) estimate per-vehicle world-frame velocity vectors
  from consecutive receiver-buffered packets (0 extra wire bytes, 0 extra parameters, strict zero-ego),
  then warp forward by `vel * dt_delay` before transforming into current ego coordinates.
"""

from __future__ import annotations

import numpy as np
import torch

from borrowedsight.grid import transform_points, voxel_centers_torch
from borrowedsight.labels import VEHICLE


def bev_connected_clusters(pts_w: np.ndarray, cell_m: float = 1.0, min_pts: int = 2):
    """Pure-NumPy 8-connected BEV clustering (zero scipy dependency, sub-millisecond).

    Returns (point_labels, box_centers_xy, cluster_ids).
    Using 2D axis-aligned box centers `(min + max) / 2` reduces LiDAR partial-occlusion
    centroid drift compared to raw point-mean centroids.
    """
    n = len(pts_w)
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 2), dtype=np.float32), []
    xy = pts_w[:, :2]
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / float(cell_m)).astype(np.int32)
    cell_map: dict[tuple[int, int], list[int]] = {}
    for idx in range(n):
        cell_map.setdefault((int(ij[idx, 0]), int(ij[idx, 1])), []).append(idx)

    visited: set[tuple[int, int]] = set()
    pt_lbl = np.zeros(n, dtype=np.int32)
    centers: list[np.ndarray] = []
    ids: list[int] = []
    cid = 0
    for cell in cell_map:
        if cell in visited:
            continue
        cid += 1
        stack = [cell]
        visited.add(cell)
        members: list[int] = []
        while stack:
            cx, cy = stack.pop()
            members.extend(cell_map[(cx, cy)])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in cell_map and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
        if len(members) >= int(min_pts):
            m_arr = np.asarray(members, dtype=np.int32)
            pt_lbl[m_arr] = cid
            sub = xy[m_arr]
            # Blend box center + mean centroid for stability against single outlier points
            box_c = 0.5 * (sub.min(axis=0) + sub.max(axis=0))
            mean_c = sub.mean(axis=0)
            centers.append(0.5 * box_c + 0.5 * mean_c)
            ids.append(cid)
    if not centers:
        return pt_lbl, np.zeros((0, 2), dtype=np.float32), []
    return pt_lbl, np.asarray(centers, dtype=np.float32), ids


def sender_coords_to_world(coords_np: np.ndarray, sender_pose: np.ndarray) -> np.ndarray:
    if len(coords_np) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers_s = voxel_centers_torch(torch.as_tensor(coords_np, dtype=torch.long)).numpy()
    zero_pose = np.zeros(6, dtype=np.float64)
    return transform_points(centers_s, sender_pose, zero_pose).astype(np.float32)


def estimate_vehicle_velocities(
    curr_veh_w: np.ndarray,
    prev_packets_veh_w: list[tuple[np.ndarray, float]],
    cell_m: float = 1.0,
    max_speed_mps: float = 32.0,
    static_thresh_mps: float = 0.75,
) -> tuple[np.ndarray, list[float]]:
    """Estimate per-point world velocity `(vx, vy, 0)` from 1-2 buffered prior packets.

    `prev_packets_veh_w`: list of `(prev_pts_w, dt_prev_seconds)` ordered most-recent first.
    Using a 2-step baseline (`dt_prev = 0.2 s` when available, falling back to `0.1 s`) halves
    voxel-quantization velocity noise before extrapolating to 200-500 ms delays.
    """
    n = len(curr_veh_w)
    vel = np.zeros((n, 3), dtype=np.float32)
    speeds_est: list[float] = []
    if n == 0 or not prev_packets_veh_w:
        return vel, speeds_est

    c_lbl, c_cent, c_ids = bev_connected_clusters(curr_veh_w, cell_m=cell_m)
    if len(c_cent) == 0:
        return vel, speeds_est

    # Accumulate velocity estimates across available prior lags (preferring 0.2s over 0.1s)
    v_sum = np.zeros((len(c_ids), 2), dtype=np.float32)
    w_sum = np.zeros((len(c_ids),), dtype=np.float32)

    for prev_pts_w, dt_prev in prev_packets_veh_w:
        if len(prev_pts_w) == 0 or dt_prev <= 1e-3:
            continue
        _, p_cent, _ = bev_connected_clusters(prev_pts_w, cell_m=cell_m)
        if len(p_cent) == 0:
            continue
        max_disp = float(max_speed_mps) * float(dt_prev)
        dist = np.linalg.norm(c_cent[:, None, :] - p_cent[None, :, :], axis=-1)
        pairs = []
        for ri in range(dist.shape[0]):
            ci = int(np.argmin(dist[ri]))
            d = float(dist[ri, ci])
            if d <= max_disp:
                pairs.append((d, ri, ci))
        pairs.sort(key=lambda x: x[0])
        used_r: set[int] = set()
        used_c: set[int] = set()
        # Longer baseline (e.g. 0.2s vs 0.1s) carries 2x lower quantization variance -> weight proportional to dt_prev
        w_lag = float(dt_prev)
        for d, ri, ci in pairs:
            if ri in used_r or ci in used_c:
                continue
            used_r.add(ri)
            used_c.add(ci)
            v_xy = (c_cent[ri] - p_cent[ci]) / float(dt_prev)
            v_sum[ri] += v_xy * w_lag
            w_sum[ri] += w_lag

    for ri, cid in enumerate(c_ids):
        if w_sum[ri] <= 0.0:
            continue
        v_xy = v_sum[ri] / w_sum[ri]
        spd = float(np.linalg.norm(v_xy))
        speeds_est.append(spd)
        if spd < float(static_thresh_mps):
            continue
        m = c_lbl == cid
        vel[m, 0] = v_xy[0]
        vel[m, 1] = v_xy[1]
    return vel, speeds_est


def warp_tokens_ss_kfw(
    coords_s: np.ndarray,
    cls_arr: np.ndarray,
    sender_pose: np.ndarray,
    ego_pose: np.ndarray,
    dt_delay: float,
    prev_packets_veh_w: list[tuple[np.ndarray, float]] | None = None,
    max_warp_m: float = 16.0,
) -> tuple[np.ndarray, list[float]]:
    """Warp sender token centers into current ego frame with SS-KFW on VEHICLE tokens only."""
    if len(coords_s) == 0:
        return np.zeros((0, 3), dtype=np.float32), []
    pts_w = sender_coords_to_world(coords_s, sender_pose)
    speeds: list[float] = []
    if dt_delay > 1e-3 and prev_packets_veh_w:
        veh_mask = np.asarray(cls_arr, dtype=np.int64) == VEHICLE
        if bool(veh_mask.any()):
            vel, speeds = estimate_vehicle_velocities(pts_w[veh_mask], prev_packets_veh_w)
            shift = vel * float(dt_delay)
            norms = np.linalg.norm(shift[:, :2], axis=-1, keepdims=True)
            scale = np.where(norms > max_warp_m, max_warp_m / np.maximum(norms, 1e-6), 1.0)
            pts_w[veh_mask] = pts_w[veh_mask] + shift * scale
    zero_pose = np.zeros(6, dtype=np.float64)
    pts_e = transform_points(pts_w, zero_pose, ego_pose).astype(np.float32)
    return pts_e, speeds
