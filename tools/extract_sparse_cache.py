"""Freeze the sender and write a sharded sparse cache. Not one file per frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from borrowedsight.data import Mini200
from borrowedsight.geometry import geo_stats
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import coarse_from_tristate
from borrowedsight.paths import CACHE_CLEAR, CHECKPOINTS, CONFIGS
from tools.ray_visibility import ray_visibility


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    device = torch.device("cuda")
    model = BorrowedSight("B1", feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"])).to(device)
    ckpt = torch.load(CHECKPOINTS / "sender_mini.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()
    dataset = Mini200(data_cfg["root"], "all", int(data_cfg["val_every"]))
    CACHE_CLEAR.mkdir(parents=True, exist_ok=True)
    shard_rows = 50
    buffer = []
    shard_id = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        for slot, (points, _labels) in enumerate(zip(sample["points"], sample["point_sem"])):
            buffer.append(encode_sender(model, sample, slot, points, device))
        if len(buffer) >= shard_rows:
            flush(buffer, shard_id)
            buffer = []
            shard_id += 1
            print(f"shard {shard_id} written", flush=True)
    if buffer:
        flush(buffer, shard_id)
    total = sum(path.stat().st_size for path in CACHE_CLEAR.glob("*.npz"))
    manifest = {"bytes": int(total), "shards": shard_id + (1 if buffer else 0), "rows_per_shard": shard_rows}
    (CACHE_CLEAR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    if total > 80 * (1024 ** 3):
        raise SystemExit("cache exceeded 80GB")


def encode_sender(model, sample, slot, points, device) -> dict:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        pts_t = torch.zeros(1, 1, 3, device=device)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        tri = torch.zeros(100, 100, 8, dtype=torch.uint8, device=device)
        coords = torch.zeros(0, 3, dtype=torch.long, device=device)
        stats = torch.zeros(0, 6, device=device)
        feat = torch.zeros(0, model.feat_dim, device=device)
        pred = torch.zeros(0, dtype=torch.long, device=device)
    else:
        pts_t = torch.as_tensor(pts, device=device).reshape(1, -1, 3)
        mask = torch.ones(1, pts_t.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.encoder(pts_t, mask)
        tri = ray_visibility(pts_t[0], torch.zeros(3, device=device))
        coords, stats, _lin = geo_stats(pts_t[0])
        if coords.numel() == 0:
            feat = torch.zeros(0, model.feat_dim, device=device)
            pred = torch.zeros(0, dtype=torch.long, device=device)
        else:
            feat_map = out["feat"][0].float()
            sem_map = out["sem"][0].argmax(dim=0)
            feat = feat_map[:, coords[:, 0], coords[:, 1], coords[:, 2]].transpose(0, 1).contiguous()
            pred = sem_map[coords[:, 0], coords[:, 1], coords[:, 2]]
            occupied = tri[coords[:, 0], coords[:, 1], coords[:, 2]] == 2
            coords, stats, feat, pred = coords[occupied], stats[occupied], feat[occupied], pred[occupied]
    envelope = coarse_from_tristate(tri.detach().cpu().numpy())
    return {
        "scene": int(sample["scene"]),
        "frame": int(sample["frame"]),
        "sender_id": int(sample["sender_ids"][slot]),
        "pose": np.asarray(sample["sender_pose"][slot], dtype=np.float32),
        "timestamp": float(sample["timestamp"]),
        "speed": float(sample["motion"][0]),
        "yaw_rate": float(sample["motion"][1]),
        "coords": coords.detach().cpu().numpy().astype(np.int16),
        "feats": feat.detach().float().cpu().numpy().astype(np.float16),
        "geo": stats.detach().float().cpu().numpy().astype(np.float16),
        "pred_class": pred.detach().cpu().numpy().astype(np.uint8),
        "envelope": envelope.astype(np.uint8),
    }


def flush(rows: list[dict], shard_id: int) -> None:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    coords, feats, geo, pred = [], [], [], []
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row["coords"])
        coords.append(row["coords"])
        feats.append(row["feats"])
        geo.append(row["geo"])
        pred.append(row["pred_class"])
    path = CACHE_CLEAR / f"shard_{shard_id:03d}.npz"
    np.savez_compressed(
        path,
        scene=np.array([row["scene"] for row in rows], dtype=np.int32),
        frame=np.array([row["frame"] for row in rows], dtype=np.int32),
        sender_id=np.array([row["sender_id"] for row in rows], dtype=np.int32),
        pose=np.stack([row["pose"] for row in rows]).astype(np.float32),
        timestamp=np.array([row["timestamp"] for row in rows], dtype=np.float32),
        speed=np.array([row["speed"] for row in rows], dtype=np.float32),
        yaw_rate=np.array([row["yaw_rate"] for row in rows], dtype=np.float32),
        offsets=offsets,
        coords=np.concatenate(coords, axis=0) if coords else np.zeros((0, 3), np.int16),
        feats=np.concatenate(feats, axis=0) if feats else np.zeros((0, 16), np.float16),
        geo=np.concatenate(geo, axis=0) if geo else np.zeros((0, 6), np.float16),
        pred_class=np.concatenate(pred, axis=0) if pred else np.zeros((0,), np.uint8),
        envelope=np.stack([row["envelope"] for row in rows]).astype(np.uint8),
    )


if __name__ == "__main__":
    main()
