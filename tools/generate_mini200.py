"""Write the synthetic Mini-200 set. Does not touch the OPV2V download."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from borrowedsight.data import simulate_frame
from borrowedsight.paths import CONFIGS


def save_scene(scene: int, frames: list[dict], path: Path) -> None:
    n_frames = len(frames)
    n_senders = len(frames[0]["sender_ids"])
    offsets = np.zeros((n_frames, n_senders + 1), dtype=np.int64)
    clouds = []
    labels = []
    cursor = 0
    for frame_id, frame in enumerate(frames):
        offsets[frame_id, 0] = cursor
        for sender, (pts, sem) in enumerate(zip(frame["points"], frame["point_sem"])):
            clouds.append(np.asarray(pts, dtype=np.float32))
            labels.append(np.asarray(sem, dtype=np.int16))
            cursor += int(len(pts))
            offsets[frame_id, sender + 1] = cursor
    points = np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3), np.float32)
    point_sem = np.concatenate(labels, axis=0) if labels else np.zeros((0,), np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ego_pose=np.stack([frame["ego_pose"] for frame in frames]).astype(np.float32),
        timestamp=np.array([frame["timestamp"] for frame in frames], dtype=np.float32),
        motion=np.stack([frame["motion"] for frame in frames]).astype(np.float32),
        sender_pose=np.stack([frame["sender_pose"] for frame in frames]).astype(np.float32),
        sender_ids=frames[0]["sender_ids"].astype(np.int64),
        receiver_id=np.int64(frames[0]["receiver_id"]),
        point_offsets=offsets,
        points=points,
        point_sem=point_sem,
        gt_semantic=np.stack([frame["gt_semantic"] for frame in frames]).astype(np.uint8),
        gt_tristate=np.stack([frame["gt_tristate"] for frame in frames]).astype(np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIGS / "mini200.yaml")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    for scene in range(int(cfg["scenes"])):
        frames = []
        occupied = []
        for frame in range(int(cfg["frames_per_scene"])):
            sample = simulate_frame(
                scene,
                frame,
                n_beams=int(cfg["n_beams"]),
                n_azimuth=int(cfg["n_azimuth"]),
                min_range=float(cfg["min_range_m"]),
                max_range=float(cfg["max_range_m"]),
            )
            frames.append(sample)
            occupied.append(int((sample["gt_tristate"] == 2).sum()))
            if frame % 10 == 0:
                points = sum(len(pts) for pts in sample["points"])
                print(f"scene {scene} frame {frame} points {points} occupied {occupied[-1]}", flush=True)
        path = root / f"scene_{scene:02d}.npz"
        save_scene(scene, frames, path)
        print(
            f"wrote {path} median_occupied {int(np.median(occupied))} "
            f"min_occupied {min(occupied)} max_occupied {max(occupied)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
