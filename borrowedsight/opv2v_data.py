"""Real OPV2V + Semantic-OPV2V 4LidarSurround dataset with strict Zero-Ego boundary.

- Sender inputs: raw OPV2V neighbor `.pcd` sweeps only (never 4LidarSurround, never EGO `.pcd`).
- Receiver inputs: ego_pose, timestamp, motion, and neighbor serialized packets only.
- Labels:
  - Sender local supervision: `sender_sem[frame, k]` from 4LidarSurround voxel archive.
  - Receiver collaborative supervision: `collab_sem[frame]` from 4LidarSurround +
    `gt_tristate` computed strictly from neighbor raw OPV2V lidar rays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from borrowedsight.grid import GRID_SIZE, transform_points
from borrowedsight.paths import DATA_ROOT
from borrowedsight.semantic_vox import parse_raw_pcd_xyz
from tools.ray_visibility import ray_visibility, union_tristate

OPV2V_ROOT = DATA_ROOT / "OPV2V"
SEM_VOX_ROOT = DATA_ROOT / "semantic_voxels"


class RealOPV2VDataset:
    """Index all voxelized Semantic-OPV2V scenes in a split (train / validate / test)."""

    def __init__(
        self,
        split: str,
        opv2v_root: Path = OPV2V_ROOT,
        vox_root: Path = SEM_VOX_ROOT,
        stride: int = 1,
        scenes: list[str] | None = None,
    ):
        self.split = split
        self.opv2v_root = Path(opv2v_root) / split
        self.vox_root = Path(vox_root) / split
        self.stride = max(1, int(stride))
        if not self.vox_root.is_dir():
            raise FileNotFoundError(f"No semantic voxel directory: {self.vox_root}")

        npz_files = sorted(self.vox_root.glob("*.npz"))
        if scenes is not None:
            allowed = set(scenes)
            npz_files = [p for p in npz_files if p.stem in allowed]
        if not npz_files:
            raise FileNotFoundError(f"No voxelized .npz scenes found in {self.vox_root}")

        self.scenes: list[dict] = []
        self.index: list[tuple[int, int]] = []
        for s_idx, path in enumerate(npz_files):
            raw_dir = self.opv2v_root / path.stem
            if not raw_dir.is_dir():
                continue
            with np.load(path, allow_pickle=False) as blob:
                cav_ids = [str(c) for c in blob["cav_ids"]]
                stems = [str(s) for s in blob["stems"]]
                poses = np.asarray(blob["poses"], dtype=np.float32)
                speeds = np.asarray(blob["speeds"], dtype=np.float32)
                n_vehicles = np.asarray(blob["n_vehicles"], dtype=np.int16)
            self.scenes.append(
                {
                    "name": path.stem,
                    "npz_path": path,
                    "raw_dir": raw_dir,
                    "cav_ids": cav_ids,
                    "stems": stems,
                    "poses": poses,
                    "speeds": speeds,
                    "n_vehicles": n_vehicles,
                }
            )
            for fi in range(0, len(stems), self.stride):
                self.index.append((s_idx, fi))

        self._npz_cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _load_scene_arrays(self, s_idx: int) -> dict[str, np.ndarray]:
        cached = self._npz_cache.get(s_idx)
        if cached is not None:
            return cached
        # Keep memory bounded: cache at most 8 scene archives in RAM
        if len(self._npz_cache) >= 8:
            oldest = next(iter(self._npz_cache))
            del self._npz_cache[oldest]
        with np.load(self.scenes[s_idx]["npz_path"], allow_pickle=False) as blob:
            cached = {
                "sender_sem": np.asarray(blob["sender_sem"], dtype=np.uint8),
                "collab_sem": np.asarray(blob["collab_sem"], dtype=np.uint8),
            }
        self._npz_cache[s_idx] = cached
        return cached

    def get_sender_sweep(self, s_idx: int, fi: int, cav_slot: int) -> tuple[np.ndarray, np.ndarray]:
        """Load one CAV's raw OPV2V lidar .pcd and its local-frame semantic voxel GT."""
        sc = self.scenes[s_idx]
        cav = sc["cav_ids"][cav_slot]
        stem = sc["stems"][fi]
        pts = parse_raw_pcd_xyz(sc["raw_dir"] / cav / f"{stem}.pcd")
        sem_vox = self._load_scene_arrays(s_idx)["sender_sem"][fi, cav_slot]
        return pts, sem_vox

    def __getitem__(self, item: int, device: torch.device | None = None, load_points: bool = True) -> dict:
        s_idx, fi = self.index[item]
        sc = self.scenes[s_idx]
        stem = sc["stems"][fi]
        cav_ids = sc["cav_ids"]
        poses = sc["poses"][fi]
        speeds = sc["speeds"][fi]
        ego_pose = poses[0]
        receiver_id = int(cav_ids[0])

        # Strictly exclude EGO (slot 0) from senders!
        sender_slots = list(range(1, len(cav_ids)))
        sender_ids = np.array([int(cav_ids[k]) for k in sender_slots], dtype=np.int64)
        sender_pose = poses[sender_slots]
        sender_speeds = speeds[sender_slots]
        if receiver_id in set(sender_ids.tolist()):
            raise AssertionError("Zero-Ego violation: receiver_id appeared in sender_ids")

        arrays = self._load_scene_arrays(s_idx)
        gt_semantic = arrays["collab_sem"][fi]

        points = []
        if load_points:
            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grids = []
            for k in sender_slots:
                pts_s = parse_raw_pcd_xyz(sc["raw_dir"] / cav_ids[k] / f"{stem}.pcd")
                points.append(pts_s)
                if len(pts_s) == 0:
                    grids.append(torch.zeros(GRID_SIZE, dtype=torch.uint8, device=dev))
                else:
                    pts_e = transform_points(pts_s, poses[k], ego_pose)
                    orig_e = transform_points(np.zeros((1, 3)), poses[k], ego_pose)[0]
                    grids.append(
                        ray_visibility(
                            torch.as_tensor(pts_e, dtype=torch.float32, device=dev),
                            torch.as_tensor(orig_e, dtype=torch.float32, device=dev),
                        )
                    )
            gt_tristate = union_tristate(grids).detach().cpu().numpy().astype(np.uint8)
        else:
            gt_tristate = None

        return {
            "scene": int(s_idx),
            "scene_name": sc["name"],
            "frame": int(fi),
            "stem": stem,
            "frame_id": f"{sc['name']}_{stem}",
            "timestamp": float(fi) * 0.1,
            "ego_pose": ego_pose.astype(np.float32),
            "motion": np.array([float(speeds[0]), 0.0], dtype=np.float32),
            "receiver_id": receiver_id,
            "sender_ids": sender_ids,
            "sender_pose": sender_pose.astype(np.float32),
            "sender_speeds": sender_speeds.astype(np.float32),
            "points": points,
            "gt_semantic": gt_semantic,
            "gt_tristate": gt_tristate,
            "n_vehicles": int(sc["n_vehicles"][fi]),
        }
