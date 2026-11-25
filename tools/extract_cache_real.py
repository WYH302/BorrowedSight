"""Phase 3b (Real OPV2V): Freeze sender_opv2v.pt and extract sharded sparse feature caches.

Strictly separates Clear cache from LISA Fog and LISA Rain test caches (per CLAUDE.md):
- D:/bs_data/BorrowedSight/cache/opv2v_clear/{train,validate,test}/
- D:/bs_data/BorrowedSight/cache/opv2v_lisa_fog/{validate,test}/
- D:/bs_data/BorrowedSight/cache/opv2v_lisa_rain/{validate,test}/

Enforces Zero-Ego boundary: only neighbor CAVs (slot >= 1, sender_id != receiver_id)
are encoded into sender packets; EGO lidar is NEVER encoded or stored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from borrowedsight.geometry import geo_stats
from borrowedsight.grid import GRID_SIZE, transform_points
from borrowedsight.model import BorrowedSight
from borrowedsight.opv2v_data import RealOPV2VDataset
from borrowedsight.packer import coarse_from_tristate, envelope50_bitplanes
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT
from borrowedsight.weather import lisa_fog, lisa_rain, load_weather
from tools.ray_visibility import ray_visibility, union_tristate

CACHE_ROOT = DATA_ROOT / "cache"


def encode_one_sender(
    model: BorrowedSight,
    pts: np.ndarray,
    sender_id: int,
    pose: np.ndarray,
    timestamp: float,
    speed: float,
    device: torch.device,
) -> dict:
    pts_f = np.asarray(pts, dtype=np.float32)
    if len(pts_f) == 0:
        tri = torch.zeros(GRID_SIZE, dtype=torch.uint8, device=device)
        coords = torch.zeros((0, 3), dtype=torch.long, device=device)
        stats = torch.zeros((0, 6), dtype=torch.float32, device=device)
        feat = torch.zeros((0, model.feat_dim), dtype=torch.float32, device=device)
        pred = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        pts_t = torch.as_tensor(pts_f, device=device).reshape(1, -1, 3)
        mask = torch.ones(1, pts_t.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.encoder(pts_t, mask)
        tri = ray_visibility(pts_t[0], torch.zeros(3, device=device))
        coords, stats, _ = geo_stats(pts_t[0])
        if coords.numel() == 0:
            feat = torch.zeros((0, model.feat_dim), dtype=torch.float32, device=device)
            pred = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            feat_map = out["feat"][0].float()
            sem_map = out["sem"][0].argmax(dim=0)
            feat = feat_map[:, coords[:, 0], coords[:, 1], coords[:, 2]].transpose(0, 1).contiguous()
            pred = sem_map[coords[:, 0], coords[:, 1], coords[:, 2]]
            occupied = tri[coords[:, 0], coords[:, 1], coords[:, 2]] == 2
            coords = coords[occupied]
            stats = stats[occupied]
            feat = feat[occupied]
            pred = pred[occupied]
    tri_np = tri.detach().cpu().numpy()
    envelope = coarse_from_tristate(tri_np, cells=20)
    envelope25 = coarse_from_tristate(tri_np, cells=25)
    envelope50 = envelope50_bitplanes(tri_np)
    return {
        "sender_id": int(sender_id),
        "pose": np.asarray(pose, dtype=np.float32),
        "timestamp": float(timestamp),
        "speed": float(speed),
        "yaw_rate": 0.0,
        "coords": coords.detach().cpu().numpy().astype(np.int16),
        "feats": feat.detach().float().cpu().numpy().astype(np.float16),
        "geo": stats.detach().float().cpu().numpy().astype(np.float16),
        "pred_class": pred.detach().cpu().numpy().astype(np.uint8),
        "envelope": envelope.astype(np.uint8),
        "envelope25": envelope25.astype(np.uint8),
        "envelope50": envelope50.astype(np.uint8),
    }


def flush_shard(out_dir: Path, shard_id: int, frames: list[dict], sender_rows: list[dict]) -> int:
    # Pack sender rows
    s_offsets = np.zeros(len(sender_rows) + 1, dtype=np.int64)
    coords, feats, geo, pred = [], [], [], []
    for i, r in enumerate(sender_rows):
        s_offsets[i + 1] = s_offsets[i] + len(r["coords"])
        coords.append(r["coords"])
        feats.append(r["feats"])
        geo.append(r["geo"])
        pred.append(r["pred_class"])
    f_sender_offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    for i, fr in enumerate(frames):
        f_sender_offsets[i + 1] = f_sender_offsets[i] + fr["n_senders"]

    path = out_dir / f"shard_{shard_id:03d}.npz"
    np.savez_compressed(
        path,
        # Frame-level receiver metadata & GT (never contains ego lidar!)
        scene_name=np.array([fr["scene_name"] for fr in frames]),
        frame_idx=np.array([fr["frame"] for fr in frames], dtype=np.int32),
        stem=np.array([fr["stem"] for fr in frames]),
        timestamp=np.array([fr["timestamp"] for fr in frames], dtype=np.float32),
        ego_pose=np.stack([fr["ego_pose"] for fr in frames]).astype(np.float32),
        motion=np.stack([fr["motion"] for fr in frames]).astype(np.float32),
        receiver_id=np.array([fr["receiver_id"] for fr in frames], dtype=np.int32),
        n_vehicles=np.array([fr["n_vehicles"] for fr in frames], dtype=np.int16),
        gt_semantic=np.stack([fr["gt_semantic"] for fr in frames]).astype(np.uint8),
        gt_tristate=np.stack([fr["gt_tristate"] for fr in frames]).astype(np.uint8),
        frame_sender_offsets=f_sender_offsets,
        # Sender-level sparse token candidates
        sender_id=np.array([r["sender_id"] for r in sender_rows], dtype=np.int32),
        sender_pose=np.stack([r["pose"] for r in sender_rows]).astype(np.float32) if sender_rows else np.zeros((0, 6), np.float32),
        sender_timestamp=np.array([r["timestamp"] for r in sender_rows], dtype=np.float32),
        sender_speed=np.array([r["speed"] for r in sender_rows], dtype=np.float32),
        sender_yaw_rate=np.array([r["yaw_rate"] for r in sender_rows], dtype=np.float32),
        envelope=np.stack([r["envelope"] for r in sender_rows]).astype(np.uint8) if sender_rows else np.zeros((0, 20, 20), np.uint8),
        envelope25=np.stack([r["envelope25"] for r in sender_rows]).astype(np.uint8) if sender_rows else np.zeros((0, 25, 25), np.uint8),
        envelope50=np.stack([r["envelope50"] for r in sender_rows]).astype(np.uint8) if sender_rows else np.zeros((0, 939), np.uint8),
        token_offsets=s_offsets,
        coords=np.concatenate(coords, axis=0) if coords else np.zeros((0, 3), np.int16),
        feats=np.concatenate(feats, axis=0) if feats else np.zeros((0, 16), np.float16),
        geo=np.concatenate(geo, axis=0) if geo else np.zeros((0, 6), np.float16),
        pred_class=np.concatenate(pred, axis=0) if pred else np.zeros((0,), np.uint8),
    )
    return int(path.stat().st_size)


def extract_split(
    model: BorrowedSight,
    split: str,
    weather_mode: str,
    out_dir: Path,
    device: torch.device,
    stride: int = 1,
    seed: int = 10,
    frames_per_shard: int = 60,
) -> dict:
    weather_cfg = load_weather()
    rng = np.random.default_rng(seed)
    ds = RealOPV2VDataset(split, stride=stride)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.npz"):
        old.unlink()

    frame_buf = []
    sender_buf = []
    shard_id = 0
    total_bytes = 0
    t0 = time.perf_counter()

    for idx in range(len(ds)):
        sample = ds.__getitem__(idx, device=device, load_points=True)
        recv_id = int(sample["receiver_id"])
        ego_pose = sample["ego_pose"]
        rows_here = []
        tri_grids = []
        for slot in range(len(sample["sender_ids"])):
            sid = int(sample["sender_ids"][slot])
            if sid == recv_id:
                raise AssertionError("Zero-Ego violation during cache extraction!")
            pts = sample["points"][slot]
            if weather_mode == "fog" and len(pts) > 0:
                pts = lisa_fog(pts, rng, weather_cfg).astype(np.float32)
            elif weather_mode == "rain" and len(pts) > 0:
                pts, _ = lisa_rain(pts, rng, weather_cfg)
                pts = pts.astype(np.float32)
            row = encode_one_sender(
                model,
                pts,
                sender_id=sid,
                pose=sample["sender_pose"][slot],
                timestamp=sample["timestamp"],
                speed=float(sample["sender_speeds"][slot]),
                device=device,
            )
            rows_here.append(row)
            # For weather modes, also compute degraded neighbor tri-state?
            # Ground-truth tri-state stays the CLEAR physical neighbor-ray visibility (what is truly occupied/free/unknown),
            # while weather degrades the SENDER'S observations (points -> features & envelope)!
            # Wait: check CLAUDE.md: tri-state GT comes from sender raw clear OPV2V rays. sample["gt_tristate"] is exact clear GT!
        frame_buf.append(
            {
                "scene_name": sample["scene_name"],
                "frame": sample["frame"],
                "stem": sample["stem"],
                "timestamp": sample["timestamp"],
                "ego_pose": ego_pose,
                "motion": sample["motion"],
                "receiver_id": recv_id,
                "n_vehicles": sample["n_vehicles"],
                "gt_semantic": sample["gt_semantic"],
                "gt_tristate": sample["gt_tristate"],
                "n_senders": len(rows_here),
            }
        )
        sender_buf.extend(rows_here)
        if len(frame_buf) >= frames_per_shard:
            total_bytes += flush_shard(out_dir, shard_id, frame_buf, sender_buf)
            shard_id += 1
            frame_buf, sender_buf = [], []
            if shard_id % 5 == 0:
                print(
                    f"  [{weather_mode}/{split}] shard {shard_id} ({idx+1}/{len(ds)} frames, {total_bytes/1e6:.1f} MB)",
                    flush=True,
                )
    if frame_buf:
        total_bytes += flush_shard(out_dir, shard_id, frame_buf, sender_buf)
        shard_id += 1

    manifest = {
        "split": split,
        "weather": weather_mode,
        "scenes": [s["name"] for s in ds.scenes],
        "frames": len(ds),
        "shards": shard_id,
        "bytes": total_bytes,
        "seconds": time.perf_counter() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"DONE cache [{weather_mode}/{split}]: scenes={len(ds.scenes)} frames={len(ds)} "
        f"size={total_bytes/1e6:.1f}MB in {manifest['seconds']:.1f}s",
        flush=True,
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=1)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    model = BorrowedSight("B1", feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"])).to(device)
    ckpt = torch.load(CHECKPOINTS / "sender_opv2v.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.eval()

    total_cache_bytes = 0
    # Clear caches: train, validate, test
    for split, st in (("train", args.train_stride), ("validate", args.eval_stride), ("test", args.eval_stride)):
        m = extract_split(model, split, "clear", CACHE_ROOT / "opv2v_clear" / split, device, stride=st, seed=0)
        total_cache_bytes += m["bytes"]
    # Disjoint LISA test weather caches: validate & test ONLY (never train!)
    for mode, sd in (("fog", 101), ("rain", 202)):
        for split in ("validate", "test"):
            m = extract_split(
                model,
                split,
                mode,
                CACHE_ROOT / f"opv2v_lisa_{mode}" / split,
                device,
                stride=args.eval_stride,
                seed=sd,
            )
            total_cache_bytes += m["bytes"]

    print(f"TOTAL OPV2V SPARSE CACHE SIZE: {total_cache_bytes / 1e9:.3f} GB (cap 80 GB)", flush=True)
    if total_cache_bytes > 80 * 1024**3:
        raise SystemExit("Cache exceeded 80GB cap!")


if __name__ == "__main__":
    main()
