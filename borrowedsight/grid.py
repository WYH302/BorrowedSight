"""CoHFF grid and pose convention.

Read from refs/CoHFF and copied here so training does not invent a second range.

task_config.yml:
    vox_range: [20, 20, 0.9, -20, -20, -2.5]
    h, w, z = 100, 100, 8

opv2v_dataset.py:
    intervals = (max_bound - min_bound) / (grid_size - 1)
    index = floor((point - min_bound) / intervals)
    a point is inside only when min < point < max on every axis

transformation_utils.py:
    pose is [x, y, z, roll, yaw, pitch] in degrees
    p_ego = inv(T_ego) @ T_sender @ p_sender
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Official CoHFF numbers. max is listed first in their yaml.
VOX_RANGE = (20.0, 20.0, 0.9, -20.0, -20.0, -2.5)
GRID_SIZE = (100, 100, 8)  # x, y, z

MAX_BOUND = np.array(VOX_RANGE[:3], dtype=np.float64)
MIN_BOUND = np.array(VOX_RANGE[3:], dtype=np.float64)
GRID_NP = np.array(GRID_SIZE, dtype=np.int64)
INTERVALS = (MAX_BOUND - MIN_BOUND) / (GRID_NP - 1)


def _as_pose(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (6,):
        raise ValueError(f"pose must be length 6, got {pose.shape}")
    return pose


def x_to_world(pose) -> np.ndarray:
    """4x4 matrix from a vehicle frame to the CARLA world frame."""
    x, y, z, roll, yaw, pitch = _as_pose(pose)
    c_y = math.cos(math.radians(float(yaw)))
    s_y = math.sin(math.radians(float(yaw)))
    c_r = math.cos(math.radians(float(roll)))
    s_r = math.sin(math.radians(float(roll)))
    c_p = math.cos(math.radians(float(pitch)))
    s_p = math.sin(math.radians(float(pitch)))

    matrix = np.identity(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def x1_to_x2(pose_src, pose_dst) -> np.ndarray:
    """Map a point in the source vehicle frame into the destination frame."""
    src_to_world = x_to_world(pose_src)
    world_to_dst = np.linalg.inv(x_to_world(pose_dst))
    return world_to_dst @ src_to_world


def transform_points(points, pose_src, pose_dst) -> np.ndarray:
    """p_dst = inv(T_dst) @ T_src @ p_src."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    matrix = x1_to_x2(pose_src, pose_dst)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (matrix @ homo.T).T
    return out[:, :3]


def voxel_index(points) -> tuple[np.ndarray, np.ndarray]:
    """CoHFF voxel index. Returns int64 indices [N,3] and a valid mask [N]."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=bool)
    valid = np.all((pts > MIN_BOUND) & (pts < MAX_BOUND), axis=-1)
    idx = np.floor((pts - MIN_BOUND) / INTERVALS).astype(np.int64)
    in_grid = np.all((idx >= 0) & (idx < GRID_NP), axis=-1)
    valid = valid & in_grid
    idx = np.where(valid[:, None], idx, 0)
    return idx, valid


def voxel_centers(indices) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.float64)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return MIN_BOUND + (idx + 0.5) * INTERVALS


def voxel_index_torch(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same CoHFF index. The division is done in float64."""
    if points.numel() == 0:
        empty = torch.zeros((0, 3), dtype=torch.long, device=points.device)
        mask = torch.zeros((0,), dtype=torch.bool, device=points.device)
        return empty, mask
    min_b = torch.as_tensor(MIN_BOUND, dtype=torch.float64, device=points.device)
    max_b = torch.as_tensor(MAX_BOUND, dtype=torch.float64, device=points.device)
    intervals = torch.as_tensor(INTERVALS, dtype=torch.float64, device=points.device)
    grid = torch.as_tensor(GRID_NP, dtype=torch.long, device=points.device)
    pts = points.to(dtype=torch.float64)
    valid = ((pts > min_b) & (pts < max_b)).all(dim=-1)
    idx = torch.floor((pts - min_b) / intervals).to(torch.long)
    in_grid = ((idx >= 0) & (idx < grid)).all(dim=-1)
    valid = valid & in_grid
    idx = torch.where(valid.unsqueeze(-1), idx, torch.zeros_like(idx))
    return idx, valid


def voxel_centers_torch(indices: torch.Tensor) -> torch.Tensor:
    min_b = torch.as_tensor(MIN_BOUND, dtype=torch.float64, device=indices.device)
    intervals = torch.as_tensor(INTERVALS, dtype=torch.float64, device=indices.device)
    return min_b + (indices.to(torch.float64) + 0.5) * intervals


def transform_points_torch(points: torch.Tensor, pose_src, pose_dst) -> torch.Tensor:
    matrix = x1_to_x2(pose_src, pose_dst)
    mat = torch.as_tensor(matrix, dtype=torch.float64, device=points.device)
    pts = points.to(dtype=torch.float64)
    if pts.numel() == 0:
        return pts
    ones = torch.ones(pts.shape[0], 1, dtype=torch.float64, device=pts.device)
    homo = torch.cat([pts, ones], dim=1)
    return (homo @ mat.T)[:, :3]


def linear_index(idx: torch.Tensor) -> torch.Tensor:
    """Flat index in [X,Y,Z] with Z fastest."""
    _gx, gy, gz = GRID_SIZE
    return idx[:, 0] * (gy * gz) + idx[:, 1] * gz + idx[:, 2]
