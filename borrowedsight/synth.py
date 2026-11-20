"""Analytic Mini-200 scenes. Boxes are known, lidar hits are ray-AABB tests."""

from __future__ import annotations

import numpy as np

from borrowedsight.grid import GRID_SIZE, voxel_centers, voxel_index, x_to_world

SENSOR_Z = 1.6
CLASS_PRIORITY = (3, 5, 6, 7, 1, 9, 12, 2, 10, 4, 11, 8)


def _box(cx, cy, sx, sy, z0, z1, cls: int) -> dict:
    return {
        "min": np.array([cx - sx * 0.5, cy - sy * 0.5, z0], dtype=np.float64),
        "max": np.array([cx + sx * 0.5, cy + sy * 0.5, z1], dtype=np.float64),
        "cls": int(cls),
    }


def actor_poses(scene: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Ego pose and three sender poses. Receiver id is not among the senders."""
    if int(scene) == 0:
        ex = -8.0 + float(frame) * 0.2
        ey = 0.0
        yaw = 0.0
        ego = np.array([ex, ey, SENSOR_Z, 0.0, yaw, 0.0], dtype=np.float64)
        senders = np.array(
            [
                [ex - 6.5, 2.4, SENSOR_Z, 0.0, 0.0, 0.0],
                [ex + 8.0, -2.2, SENSOR_Z, 0.0, 0.0, 0.0],
                [ex + 1.2, -2.6, SENSOR_Z, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        ey = -8.0 + float(frame) * 0.2
        ex = 0.0
        ego = np.array([ex, ey, SENSOR_Z, 0.0, 90.0, 0.0], dtype=np.float64)
        senders = np.array(
            [
                [2.4, ey - 6.5, SENSOR_Z, 0.0, 90.0, 0.0],
                [-2.2, ey + 8.0, SENSOR_Z, 0.0, 90.0, 0.0],
                [-2.5, ey + 1.2, SENSOR_Z, 0.0, 90.0, 0.0],
            ],
            dtype=np.float64,
        )
    return ego, senders


def world_boxes(scene: int, frame: int) -> list[dict]:
    ego, _senders = actor_poses(scene, frame)
    boxes: list[dict] = []
    if int(scene) == 0:
        boxes.append(_box(10.0, 0.0, 160.0, 8.0, 0.0, 0.25, 5))
        boxes.append(_box(10.0, 5.8, 160.0, 3.2, 0.0, 0.2, 6))
        boxes.append(_box(10.0, -5.8, 160.0, 3.2, 0.0, 0.2, 6))
        boxes.append(_box(10.0, 8.6, 160.0, 1.6, 0.0, 0.15, 3))
        boxes.append(_box(10.0, -8.6, 160.0, 1.6, 0.0, 0.15, 3))
        for x in np.arange(-40.0, 70.0, 12.0):
            boxes.append(_box(float(x), 11.2, 8.0, 2.4, 0.0, 4.0, 1))
            boxes.append(_box(float(x), -11.2, 8.0, 2.4, 0.0, 4.0, 1))
            boxes.append(_box(float(x) + 4.0, 7.6, 0.35, 0.35, 0.0, 3.0, 4))
            boxes.append(_box(float(x), 4.35, 8.0, 0.25, 0.0, 0.9, 2))
            boxes.append(_box(float(x), -4.45, 8.0, 0.2, 0.0, 0.7, 10))
            boxes.append(_box(float(x) + 2.0, 6.6, 0.15, 1.2, 0.4, 2.0, 11))
            boxes.append(_box(float(x) + 6.0, 8.4, 1.4, 1.4, 0.0, 1.3, 7))
            boxes.append(_box(float(x) + 1.0, 9.4, 3.0, 0.4, 0.0, 2.2, 9))
        boxes.append(_box(float(ego[0]) + 4.0, 0.0, 8.0, 6.0, 2.15, 2.55, 12))
        offsets = (
            (7.0, 1.3),
            (13.0, -1.6),
            (-6.0, -1.2),
            (4.0, -2.4),
            (17.0, 1.1),
            (-12.0, 1.4),
            (10.0, 2.2),
            (-3.0, -2.0),
        )
        for dx, dy in offsets:
            boxes.append(_box(float(ego[0]) + dx, float(ego[1]) + dy, 4.4, 1.9, 0.0, 1.5, 8))
    else:
        boxes.append(_box(0.0, 10.0, 8.0, 160.0, 0.0, 0.25, 5))
        boxes.append(_box(5.8, 10.0, 3.2, 160.0, 0.0, 0.2, 6))
        boxes.append(_box(-5.8, 10.0, 3.2, 160.0, 0.0, 0.2, 6))
        boxes.append(_box(8.6, 10.0, 1.6, 160.0, 0.0, 0.15, 3))
        boxes.append(_box(-8.6, 10.0, 1.6, 160.0, 0.0, 0.15, 3))
        for y in np.arange(-40.0, 70.0, 12.0):
            boxes.append(_box(11.2, float(y), 2.4, 8.0, 0.0, 4.0, 1))
            boxes.append(_box(-11.2, float(y), 2.4, 8.0, 0.0, 4.0, 1))
            boxes.append(_box(7.6, float(y) + 4.0, 0.35, 0.35, 0.0, 3.0, 4))
            boxes.append(_box(4.35, float(y), 0.25, 8.0, 0.0, 0.9, 2))
            boxes.append(_box(-4.45, float(y), 0.2, 8.0, 0.0, 0.7, 10))
            boxes.append(_box(6.6, float(y) + 2.0, 1.2, 0.15, 0.4, 2.0, 11))
            boxes.append(_box(8.4, float(y) + 6.0, 1.4, 1.4, 0.0, 1.3, 7))
            boxes.append(_box(9.4, float(y) + 1.0, 0.4, 3.0, 0.0, 2.2, 9))
        boxes.append(_box(0.0, float(ego[1]) + 4.0, 6.0, 8.0, 2.15, 2.55, 12))
        offsets = (
            (1.3, 7.0),
            (-1.6, 13.0),
            (-1.2, -6.0),
            (-2.4, 4.0),
            (1.1, 17.0),
            (1.4, -12.0),
            (2.2, 10.0),
            (-2.0, -3.0),
        )
        for dx, dy in offsets:
            boxes.append(_box(float(ego[0]) + dx, float(ego[1]) + dy, 4.4, 1.9, 0.0, 1.5, 8))
    return boxes


def _dirs(n_beams: int, n_azimuth: int) -> np.ndarray:
    elev = np.linspace(np.deg2rad(-18.0), np.deg2rad(6.0), int(n_beams))
    azim = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    ee, aa = np.meshgrid(elev, azim, indexing="ij")
    dirs = np.stack((np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)), axis=-1)
    return dirs.reshape(-1, 3)


def _contains(origin: np.ndarray, box: dict) -> bool:
    return bool(np.all(origin >= box["min"]) and np.all(origin <= box["max"]))


def cast_sweep(pose, boxes: list[dict], n_beams: int, n_azimuth: int, min_range: float, max_range: float):
    local = _dirs(n_beams, n_azimuth)
    rotation = x_to_world(pose)[:3, :3]
    dirs = local @ rotation.T
    origin = np.asarray(pose[:3], dtype=np.float64)
    use = [box for box in boxes if not _contains(origin, box)]
    if not use:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int16)
    bmin = np.stack([box["min"] for box in use], axis=0)
    bmax = np.stack([box["max"] for box in use], axis=0)
    classes = np.array([box["cls"] for box in use], dtype=np.int16)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs[:, None, :]
        t0 = (bmin[None, :, :] - origin.reshape(1, 1, 3)) * inv
        t1 = (bmax[None, :, :] - origin.reshape(1, 1, 3)) * inv
    t_near = np.minimum(t0, t1).max(axis=-1)
    t_far = np.maximum(t0, t1).min(axis=-1)
    hit_t = np.where(t_near > min_range, t_near, t_far)
    valid = np.logical_and(t_far >= np.maximum(t_near, min_range), hit_t < max_range)
    valid = np.logical_and(valid, np.isfinite(hit_t))
    hit_t = np.where(valid, hit_t, np.inf)
    best = hit_t.argmin(axis=1)
    best_t = hit_t[np.arange(hit_t.shape[0]), best]
    ok = np.isfinite(best_t)
    if not np.any(ok):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int16)
    points = (local[ok] * best_t[ok, None]).astype(np.float32)
    labels = classes[best[ok]]
    return points, labels


def raster_semantic(boxes: list[dict], ego_pose, hit_points_ego: np.ndarray, hit_labels: np.ndarray) -> np.ndarray:
    """Known occupancy on the CoHFF grid, then overwrite voxels that rays actually hit."""
    sem = np.zeros(GRID_SIZE, dtype=np.uint8)
    centers = voxel_centers(np.stack(np.meshgrid(
        np.arange(GRID_SIZE[0]), np.arange(GRID_SIZE[1]), np.arange(GRID_SIZE[2]), indexing="ij"
    ), axis=-1).reshape(-1, 3))
    world = _ego_to_world(centers, ego_pose)
    order = {cls: rank for rank, cls in enumerate(CLASS_PRIORITY)}
    for box in sorted(boxes, key=lambda item: order.get(item["cls"], 0)):
        inside = np.logical_and(
            np.all(world >= box["min"], axis=1),
            np.all(world <= box["max"], axis=1),
        )
        if np.any(inside):
            sem.reshape(-1)[inside] = np.uint8(box["cls"])
    if hit_points_ego.size:
        idx, valid = voxel_index(hit_points_ego)
        if np.any(valid):
            sem[idx[valid, 0], idx[valid, 1], idx[valid, 2]] = hit_labels[valid].astype(np.uint8)
    return sem


def _ego_to_world(points_ego: np.ndarray, ego_pose) -> np.ndarray:
    matrix = x_to_world(ego_pose)
    homo = np.concatenate([points_ego, np.ones((points_ego.shape[0], 1))], axis=1)
    return (matrix @ homo.T).T[:, :3]
