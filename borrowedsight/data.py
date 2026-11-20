"""Mini-200 samples and the receiver input boundary.

The receiver view never includes an ego lidar, ego image, or semantic label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from borrowedsight.grid import transform_points
from borrowedsight.metrics import semantic_target
from borrowedsight.synth import actor_poses, cast_sweep, raster_semantic, world_boxes
from tools.ray_visibility import ray_visibility, union_tristate

RECEIVER_ID = 0
SENDER_IDS = (1, 2, 3)


def simulate_frame(
    scene: int,
    frame: int,
    n_beams: int = 8,
    n_azimuth: int = 16,
    min_range: float = 1.0,
    max_range: float = 45.0,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ego, senders = actor_poses(scene, frame)
    boxes = world_boxes(scene, frame)
    points = []
    labels = []
    for pose in senders:
        pts, sem = cast_sweep(pose, boxes, n_beams, n_azimuth, min_range, max_range)
        points.append(pts)
        labels.append(sem)
    hit_ego = []
    hit_sem = []
    grids = []
    for pose, pts, sem in zip(senders, points, labels):
        if pts.shape[0] == 0:
            grids.append(torch.zeros(100, 100, 8, dtype=torch.uint8, device=device))
            continue
        ego_pts = transform_points(pts, pose, ego)
        origin = transform_points(np.zeros((1, 3)), pose, ego)[0]
        hit_ego.append(ego_pts)
        hit_sem.append(sem)
        grids.append(
            ray_visibility(
                torch.as_tensor(ego_pts, dtype=torch.float32, device=device),
                torch.as_tensor(origin, dtype=torch.float32, device=device),
            )
        )
    if hit_ego:
        hit_points = np.concatenate(hit_ego, axis=0)
        hit_labels = np.concatenate(hit_sem, axis=0)
    else:
        hit_points = np.zeros((0, 3), dtype=np.float64)
        hit_labels = np.zeros((0,), dtype=np.int16)
    semantic = raster_semantic(boxes, ego, hit_points, hit_labels)
    tristate = union_tristate(grids).detach().cpu().numpy()
    speed = 2.0
    return {
        "scene": int(scene),
        "frame": int(frame),
        "frame_id": f"s{scene}_f{frame:04d}",
        "timestamp": float(frame) * 0.1,
        "ego_pose": ego.astype(np.float32),
        "motion": np.array([speed, 0.0], dtype=np.float32),
        "receiver_id": RECEIVER_ID,
        "sender_ids": np.array(SENDER_IDS, dtype=np.int64),
        "sender_pose": senders.astype(np.float32),
        "points": points,
        "point_sem": labels,
        "gt_semantic": semantic,
        "gt_tristate": tristate.astype(np.uint8),
    }


def neighbor_points_in_ego(sample: dict, ego_lidar=None) -> np.ndarray:
    """Points the receiver is allowed to see: neighbor returns in the ego frame."""
    del ego_lidar
    clouds = []
    for pose, pts in zip(sample["sender_pose"], sample["points"]):
        if len(pts) == 0:
            continue
        clouds.append(transform_points(np.asarray(pts), pose, sample["ego_pose"]))
    if not clouds:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(clouds, axis=0)


def receiver_inputs(sample: dict, ego_lidar=None, messages=None):
    """Drop any ego lidar before the model call."""
    del ego_lidar
    pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32).reshape(1, 6)
    stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32)
    motion = torch.as_tensor(sample["motion"], dtype=torch.float32).reshape(1, 2)
    if messages is None:
        messages = [[]]
    return pose, stamp, motion, messages


def loss_targets(sample: dict) -> dict:
    sem = sample["gt_semantic"]
    tri = sample["gt_tristate"]
    return {
        "semantic": torch.as_tensor(semantic_target(sem, tri), dtype=torch.long),
        "tristate": torch.as_tensor(tri, dtype=torch.long),
        "semantic_grid": torch.as_tensor(sem, dtype=torch.long),
    }


class Mini200:
    def __init__(self, root: str | Path, split: str = "train", val_every: int = 5):
        self.root = Path(root)
        self.split = split
        self.val_every = int(val_every)
        self.files = sorted(self.root.glob("scene_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no Mini-200 scenes in {self.root}")
        self.index = []
        for path in self.files:
            with np.load(path) as blob:
                n_frames = int(blob["ego_pose"].shape[0])
            scene = int(path.stem.split("_")[1])
            for frame in range(n_frames):
                is_val = frame % self.val_every == 0
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue
                if split not in {"train", "val", "all"}:
                    raise KeyError(split)
                self.index.append((path, scene, frame))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        path, scene, frame = self.index[item]
        with np.load(path) as blob:
            return _frame_from_blob(blob, scene, frame)


def _frame_from_blob(blob, scene: int, frame: int) -> dict:
    offsets = blob["point_offsets"]
    points = blob["points"]
    point_sem = blob["point_sem"]
    sender_ids = blob["sender_ids"]
    n_senders = int(sender_ids.shape[0])
    clouds = []
    labels = []
    for sender in range(n_senders):
        start = int(offsets[frame, sender])
        end = int(offsets[frame, sender + 1])
        clouds.append(points[start:end])
        labels.append(point_sem[start:end])
    return {
        "scene": int(scene),
        "frame": int(frame),
        "frame_id": f"s{scene}_f{frame:04d}",
        "timestamp": float(blob["timestamp"][frame]),
        "ego_pose": np.asarray(blob["ego_pose"][frame], dtype=np.float32),
        "motion": np.asarray(blob["motion"][frame], dtype=np.float32),
        "receiver_id": int(blob["receiver_id"]),
        "sender_ids": np.asarray(sender_ids, dtype=np.int64),
        "sender_pose": np.asarray(blob["sender_pose"][frame], dtype=np.float32),
        "points": clouds,
        "point_sem": labels,
        "gt_semantic": np.asarray(blob["gt_semantic"][frame]),
        "gt_tristate": np.asarray(blob["gt_tristate"][frame]),
    }
