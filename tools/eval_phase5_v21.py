"""Phase 5 (BorrowedSight v2.1): Complete 6-axis stress evaluation, compute profiling,
per-class breakdown, and 5-CAV qualitative predictions saved to `outputs/real_opv2v/v21_phase5/`.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import TARGET_BYTES, drop_block, packet_layout, unpack_bytes
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.train_receivers_real import load_cached_split, move_sender

CACHE_ROOT = DATA_ROOT / "cache"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v2"
OUT = OUTPUTS / "real_opv2v" / "v21_phase5"

BANDWIDTHS = (2048, 4096, 8192, 16384, 24576, 32768)
POSES = ((0.0, 0.0), (0.5, 2.0), (1.0, 5.0))
DELAYS = (0, 1, 2, 5)
DROP_RATES = (0.0, 0.1, 0.25, 0.5)


def load_ckpt_v21(tag: str, seed: int, cfg: dict, device: torch.device) -> tuple[BorrowedSight, dict]:
    path = CKPT_DIR / f"{tag}_s{seed}.pt"
    blob = torch.load(path, map_location=device, weights_only=False)
    mkw = dict(blob.get("model_kwargs") or {})
    cb_size = int(mkw.pop("codebook_size", cfg["codebook_size"]))
    model = BorrowedSight(blob["method"], feat_dim=int(cfg["feat_dim"]), codebook_size=cb_size, **mkw).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def _pose_copy(row: dict, rng: np.random.Generator, xy_std: float, yaw_std: float) -> dict:
    if xy_std == 0.0 and yaw_std == 0.0:
        return row
    out = dict(row)
    pose = np.array(row["pose"], dtype=np.float32, copy=True)
    pose[0] += float(rng.normal(0.0, xy_std))
    pose[1] += float(rng.normal(0.0, xy_std))
    pose[4] += float(rng.normal(0.0, yaw_std))
    out["pose"] = pose
    return out


def _drop_blocks(blob: bytes, drop_p: float, rng: np.random.Generator) -> bytes:
    if not blob or drop_p <= 0.0:
        return blob
    n_blocks = int(unpack_bytes(blob)["n_blocks"])
    for index in range(n_blocks - 1, -1, -1):
        if rng.random() < drop_p:
            blob = drop_block(blob, index)
    return blob


def build_frame_lookup(frames: list[dict]) -> dict[tuple[str, int], dict]:
    return {(fr["scene_name"], fr["frame"]): fr for fr in frames}


def measure_stress(
    model: BorrowedSight,
    frames: list[dict],
    frame_lookup: dict[tuple[str, int], dict],
    target_bytes: int,
    device: torch.device,
    rng: np.random.Generator,
    sender_k: int | None = None,
    xy_std: float = 0.0,
    yaw_std: float = 0.0,
    delay: int = 0,
    drop_p: float = 0.0,
    filter_fn=None,
) -> dict:
    model.eval()
    budget = packet_layout(
        model.method,
        target=int(target_bytes),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    n_tokens = int(budget["n_tokens"])
    acc = new_counts()
    sizes = []
    n_eval = 0
    filtered = []
    for fr in frames:
        if filter_fn is not None and not filter_fn(fr):
            continue
        if sender_k is not None and len(fr["senders"]) < sender_k:
            continue
        filtered.append(fr)
    bs = 8
    with torch.no_grad():
        for start in range(0, len(filtered), bs):
            chunk = filtered[start : start + bs]
            pose_b = torch.as_tensor(np.stack([fr["ego_pose"] for fr in chunk]), dtype=torch.float32, device=device)
            stamp_b = torch.tensor([float(fr["timestamp"]) for fr in chunk], dtype=torch.float32, device=device)
            motion_b = torch.as_tensor(np.stack([fr["motion"] for fr in chunk]), dtype=torch.float32, device=device)
            blobs_b = []
            for fr in chunk:
                src = frame_lookup.get((fr["scene_name"], fr["frame"] - int(delay))) if delay > 0 else fr
                senders = list(src["senders"]) if src is not None else []
                ego_xy = fr["ego_pose"][:2]
                senders.sort(key=lambda r: float(np.linalg.norm(r["pose"][:2] - ego_xy)))
                if sender_k is not None:
                    senders = senders[: int(sender_k)]
                blobs = []
                for row in senders:
                    noisy = _pose_copy(row, rng, xy_std, yaw_std)
                    sent = model.transmit(
                        move_sender(noisy, device),
                        fr["ego_pose"],
                        int(fr["receiver_id"]),
                        n_tokens,
                        budget=budget,
                        serialize=True,
                    )
                    blob = _drop_blocks(sent["blob"], drop_p, rng)
                    if blob:
                        blobs.append(blob)
                        sizes.append(len(blob))
                blobs_b.append(blobs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(pose_b, stamp_b, motion_b, blobs_b)
            sems = pred["sem"].argmax(dim=1).detach().cpu().numpy()
            tris = pred["tri"].argmax(dim=1).detach().cpu().numpy()
            tri_probs_b = F.softmax(pred["tri"].float(), dim=1).detach().cpu().numpy()
            for bi, fr in enumerate(chunk):
                update_counts(acc, sems[bi], tris[bi], fr["gt_semantic"], fr["gt_tristate"], tri_probs=tri_probs_b[bi])
                n_eval += 1
    metrics = finalize_counts(acc)
    metrics["bytes_per_frame"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["bytes_min"] = float(np.min(sizes)) if sizes else 0.0
    metrics["bytes_max"] = float(np.max(sizes)) if sizes else 0.0
    metrics["n_packets"] = int(len(sizes))
    metrics["n_frames"] = n_eval
    return metrics


def benchmark_compute(models: dict[str, BorrowedSight], frames: list[dict], device: torch.device) -> dict:
    multi4 = [fr for fr in frames if len(fr["senders"]) >= 4]
    multi3 = [fr for fr in frames if len(fr["senders"]) >= 3]
    sample = multi4[0] if multi4 else (multi3[0] if multi3 else frames[0])
    out = {}
    for name in [t for t in ("B3_v2_32k", "B4_v21_32k", "B4_v21_32k_no_env") if t in models]:
        model = models[name]
        budget = packet_layout(
            model.method,
            TARGET_BYTES,
            stage=model.default_stage,
            include_envelope=model.use_envelope,
            compact=model.compact_wire,
        )
        n_tok = int(budget["n_tokens"])
        pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
        stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
        motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
        rows = [move_sender(r, device) for r in sample["senders"]]
        by_k = {}
        for k in (1, 2, 3, 4):
            if k > len(rows):
                continue
            use = rows[:k]
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(5):
                    blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                    model(pose, stamp, motion, [blobs])
            torch.cuda.synchronize()
            reps = 20
            tx_ms_acc, rx_ms_acc = 0.0, 0.0
            with torch.no_grad():
                for _ in range(reps):
                    ts0 = time.perf_counter()
                    blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                    torch.cuda.synchronize()
                    ts1 = time.perf_counter()
                    model(pose, stamp, motion, [blobs])
                    torch.cuda.synchronize()
                    ts2 = time.perf_counter()
                    tx_ms_acc += (ts1 - ts0) * 1000.0
                    rx_ms_acc += (ts2 - ts1) * 1000.0
            tot_ms = (tx_ms_acc + rx_ms_acc) / reps
            by_k[str(k)] = {
                "tx_pack_ms": round(tx_ms_acc / reps, 2),
                "rx_decode_ms": round(rx_ms_acc / reps, 2),
                "total_ms": round(tot_ms, 2),
                "fps": round(1000.0 / max(1e-3, tot_ms), 2),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "bytes_per_sender": len(blobs[0]) if blobs else 0,
                "total_frame_bytes": sum(len(b) for b in blobs),
            }
        out[name] = by_k
    return out


def predict_single(model: BorrowedSight, fr: dict, target_bytes: int, device: torch.device, sender_k=None, drop_p=0.0):
    budget = packet_layout(
        model.method,
        target_bytes,
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    n_tok = int(budget["n_tokens"])
    rng = np.random.default_rng(7)
    ego_xy = fr["ego_pose"][:2]
    senders = sorted(fr["senders"], key=lambda r: float(np.linalg.norm(r["pose"][:2] - ego_xy)))
    if sender_k is not None:
        senders = senders[:sender_k]
    pose = torch.as_tensor(fr["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
    stamp = torch.tensor([float(fr["timestamp"])], dtype=torch.float32, device=device)
    motion = torch.as_tensor(fr["motion"], dtype=torch.float32, device=device).reshape(1, 2)
    with torch.no_grad():
        blobs = []
        for row in senders:
            sent = model.transmit(move_sender(row, device), fr["ego_pose"], fr["receiver_id"], n_tok, budget=budget, serialize=True)
            b = _drop_blocks(sent["blob"], drop_p, rng)
            if b:
                blobs.append(b)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(pose, stamp, motion, [blobs])
        sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return sem, tri


def save_qualitative_5cav(models: dict[str, BorrowedSight], test_clear, test_fog, test_rain, device: torch.device) -> None:
    # Target the exact rich 5-CAV showcase frame used in Figs. 6 & 7
    idx = None
    for i, fr in enumerate(test_clear):
        if fr["scene_name"] == "2021_08_22_07_52_02" and str(fr["stem"]) == "000373":
            idx = i
            break
    if idx is None:
        multi = [i for i, fr in enumerate(test_clear) if len(fr["senders"]) >= 4 and fr["n_vehicles"] >= 4]
        idx = multi[len(multi) // 2] if multi else 0
    fr_c, fr_f, fr_r = test_clear[idx], test_fog[idx], test_rain[idx]
    save_d = {
        "scene_name": np.array(fr_c["scene_name"]),
        "stem": np.array(str(fr_c["stem"])),
        "gt_sem": fr_c["gt_semantic"].astype(np.uint8),
        "gt_tri": fr_c["gt_tristate"].astype(np.uint8),
    }
    for key, tag in (("b0", "B0_v2"), ("b1", "B1_v2_32k"), ("b3", "B3_v2_32k"), ("b4", "B4_v21_32k")):
        if tag in models:
            s, t = predict_single(models[tag], fr_c, TARGET_BYTES, device)
            save_d[f"{key}_sem"] = s
            save_d[f"{key}_tri"] = t
    if "B4_v21_32k" in models:
        m4 = models["B4_v21_32k"]
        for k, (fr_use, sk, dp) in (
            ("b4_fog", (fr_f, None, 0.0)),
            ("b4_rain", (fr_r, None, 0.0)),
            ("b4_drop50", (fr_c, None, 0.5)),
            ("b4_outage", (fr_c, 0, 0.0)),
        ):
            s, t = predict_single(m4, fr_use, TARGET_BYTES, device, sender_k=sk, drop_p=dp)
            save_d[f"{k}_sem"] = s
            save_d[f"{k}_tri"] = t
    np.savez_compressed(OUT / "qualitative_v21_5cav.npz", **save_d)
    print(f"Saved 5-CAV qualitative predictions for {fr_c['scene_name']} #{fr_c['stem']}", flush=True)


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    test_clear = load_cached_split(CACHE_ROOT / "opv2v_clear" / "test")
    test_fog = load_cached_split(CACHE_ROOT / "opv2v_lisa_fog" / "test")
    test_rain = load_cached_split(CACHE_ROOT / "opv2v_lisa_rain" / "test")
    test_lookup = build_frame_lookup(test_clear)

    load_tags = (
        "B0_v2", "B1_v2_32k", "B2_v2_32k", "B3_v2_32k", "B4_v21_32k",
        "B4_v21_32k_no_env", "B4_v21_32k_no_tdvf", "B4_v21_32k_no_cls",
        "B4_v21_32k_shared96", "B4_v21_32k_no_quota", "B4_v21_32k_stage1",
        "B4_v21_multibw",
    )
    models = {}
    for tag in load_tags:
        if (CKPT_DIR / f"{tag}_s0.pt").exists():
            models[tag], _ = load_ckpt_v21(tag, 0, cfg, device)
    print(f"Loaded {len(models)} v2.1 models for Phase 5 evaluation: {list(models.keys())}", flush=True)

    rows = []

    def push(split_name, model_name, family, level, metrics):
        rows.append(
            {
                "split": split_name,
                "model_name": model_name,
                "family": family,
                "level": str(level),
                "n_frames": metrics.get("n_frames", 0),
                "bytes_per_frame": metrics.get("bytes_per_frame"),
                "bytes_min": metrics.get("bytes_min"),
                "bytes_max": metrics.get("bytes_max"),
                "mIoU": metrics.get("mIoU"),
                "vehicle_IoU": metrics.get("vehicle_IoU"),
                "road_IoU": metrics.get("road_IoU"),
                "unknown_F1": metrics.get("unknown_F1"),
                "unobserved_as_free_err": metrics.get("unobserved_as_free_err"),
                "pred_unknown_frac": metrics.get("pred_unknown_frac"),
                "mean_uncertainty": metrics.get("mean_uncertainty"),
                "uncertainty_on_unknown": metrics.get("uncertainty_on_unknown"),
                "uncertainty_on_observed": metrics.get("uncertainty_on_observed"),
            }
        )

    print("\n--- 1. Bandwidth sweep (2-32 KB, test) ---", flush=True)
    for tag in [t for t in ("B1_v2_32k", "B2_v2_32k", "B3_v2_32k", "B4_v21_32k", "B4_v21_multibw") if t in models]:
        for bw in BANDWIDTHS:
            m = measure_stress(models[tag], test_clear, test_lookup, bw, device, np.random.default_rng(0))
            push("test", tag, "bandwidth", bw, m)
            print(f"  {tag:18s} bw={bw:5d} bytes={m['bytes_per_frame']:7.1f} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f}", flush=True)

    print("\n--- 2. LISA Weather + Per-Class Breakdown (test) ---", flush=True)
    per_class_summary = {}
    for tag in [t for t in load_tags if t != "B0_v2" and t in models]:
        for wname, wframes in (("clear", test_clear), ("fog", test_fog), ("rain", test_rain)):
            m = measure_stress(models[tag], wframes, build_frame_lookup(wframes), TARGET_BYTES, device, np.random.default_rng(0))
            push("test", tag, "weather", wname, m)
            if wname == "clear":
                per_class_summary[tag] = {
                    "mIoU": m.get("mIoU"),
                    "n_classes_present": m.get("n_classes_present"),
                    "per_class_iou": {str(k): v for k, v in (m.get("per_class_iou") or {}).items()},
                    "bytes_per_frame": m.get("bytes_per_frame"),
                }
            print(f"  {tag:20s} {wname:5s} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f} unc={m['mean_uncertainty']:.4f}", flush=True)

    print("\n--- 3. Multi-sender scaling (test_multi3 & test_multi4) ---", flush=True)
    multi3_test = [fr for fr in test_clear if len(fr["senders"]) >= 3]
    multi4_test = [fr for fr in test_clear if len(fr["senders"]) >= 4]
    for ms_split, ms_frames, max_k in (("test_multi3", multi3_test, 3), ("test_multi4", multi4_test, 4)):
        ms_lookup = build_frame_lookup(ms_frames)
        for tag in [t for t in ("B3_v2_32k", "B4_v21_32k") if t in models]:
            for k in range(1, max_k + 1):
                m = measure_stress(models[tag], ms_frames, ms_lookup, TARGET_BYTES, device, np.random.default_rng(0), sender_k=k)
                push(ms_split, tag, "senders", k, m)
                print(f"  [{ms_split}] {tag:12s} k={k} (n={m['n_frames']}) mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f}", flush=True)

    print("\n--- 4. Urban Congestion Stratification (test) ---", flush=True)
    veh_counts = [fr["n_vehicles"] for fr in test_clear]
    q33, q66 = float(np.percentile(veh_counts, 33)), float(np.percentile(veh_counts, 66))
    for tag in [t for t in ("B3_v2_32k", "B4_v21_32k") if t in models]:
        for dname, fn in (
            ("sparse", lambda fr, a=q33: fr["n_vehicles"] <= a),
            ("medium", lambda fr, a=q33, b=q66: a < fr["n_vehicles"] <= b),
            ("dense", lambda fr, b=q66: fr["n_vehicles"] > b),
        ):
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(0), filter_fn=fn)
            push("test", tag, "density", dname, m)
            print(f"  {tag:12s} density={dname:6s} (n={m['n_frames']:3d}) mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f}", flush=True)

    print("\n--- 5. Pose / Latency / Micro-Block Drop (test) ---", flush=True)
    for tag in [t for t in ("B3_v2_32k", "B4_v21_32k", "B4_v21_32k_no_env", "B4_v21_multibw") if t in models]:
        for xy_std, yaw_std in POSES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(1), xy_std=xy_std, yaw_std=yaw_std)
            push("test", tag, "pose", f"{xy_std}_{yaw_std}", m)
        for d in DELAYS:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(0), delay=d)
            push("test", tag, "latency", d, m)
        for dp in DROP_RATES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(2), drop_p=dp)
            push("test", tag, "packet_loss", dp, m)
        print(f"  DONE pose/latency/drop for {tag}", flush=True)

    print("\n--- 6. Compute / FPS / Peak VRAM Benchmark ---", flush=True)
    compute_bench = benchmark_compute(models, multi4_test or multi3_test or test_clear, device)
    print(json.dumps(compute_bench, indent=2), flush=True)

    print("\n--- 7. Saving 5-CAV Qualitative Predictions ---", flush=True)
    save_qualitative_5cav(models, test_clear, test_fog, test_rain, device)

    cols = list(rows[0].keys())
    with (OUT / "measurements.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (OUT / "compute_benchmark.json").write_text(json.dumps(compute_bench, indent=2), encoding="utf-8")
    (OUT / "per_class_iou.json").write_text(json.dumps(per_class_summary, indent=2), encoding="utf-8")
    print(f"\nDONE Phase 5 v2.1: wrote {len(rows)} rows to {OUT / 'measurements.csv'}", flush=True)


if __name__ == "__main__":
    main()
