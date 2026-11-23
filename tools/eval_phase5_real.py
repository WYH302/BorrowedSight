"""Phase 5 (Real OPV2V): Complete stress evaluation on real OPV2V validate & test splits.

Covers every A-level and B-level requirement in BorrowedSight_验收标准.txt:
- Bandwidth sweep (4KB, 8KB, 16KB, 24KB, 32KB) for B1, B2, B3, B4
- LISA physical weather (Clear, Fog, Rain) + Uncertainty & Unknown monotonicity (A9, B4)
- Multi-sender scaling (1, 2, 3+ senders) & Congestion density stratification (Sparse/Medium/Dense, A10, B2)
- V2X Robustness: Pose noise, Latency (0/1/2/5 frames), Micro-block Packet Loss (0/10/25/50%, A11, B3)
- Compute, Latency, and Peak VRAM benchmark (B6)
- 4-Scenario Qualitative BEV visualizations + Failure Case audit (Section 7 & B7)
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.labels import ROAD, VEHICLE
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import TARGET_BYTES, drop_block, packet_layout, unpack_bytes
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.train_receivers_real import load_cached_split, move_sender

CACHE_ROOT = DATA_ROOT / "cache"
CKPT_DIR = CHECKPOINTS / "real_opv2v"
OUT = OUTPUTS / "real_opv2v" / "phase5"
QUAL = OUTPUTS / "real_opv2v" / "qualitative"

BANDWIDTHS = (2048, 4096, 8192, 16384, 24576, 32768)
POSES = ((0.0, 0.0), (0.5, 2.0), (1.0, 5.0))
DELAYS = (0, 1, 2, 5)
DROP_RATES = (0.0, 0.1, 0.25, 0.5)


def load_ckpt_model(tag: str, seed: int, cfg: dict, device: torch.device) -> tuple[BorrowedSight, dict]:
    path = CKPT_DIR / f"{tag}_s{seed}.pt"
    blob = torch.load(path, map_location=device, weights_only=False)
    mkw = dict(blob.get("model_kwargs") or {})
    cb_size = int(mkw.pop("codebook_size", cfg["codebook_size"]))
    model = BorrowedSight(
        blob["method"],
        feat_dim=int(cfg["feat_dim"]),
        codebook_size=cb_size,
        **mkw,
    ).to(device)
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
    """Measure B6 compute cost: transmit+pack ms, receiver unpack+decode ms, peak VRAM vs sender count."""
    multi4 = [fr for fr in frames if len(fr["senders"]) >= 4]
    multi3 = [fr for fr in frames if len(fr["senders"]) >= 3]
    sample = multi4[0] if multi4 else (multi3[0] if multi3 else frames[0])
    out = {}
    bench_tags = [t for t in ("B3", "B4", "B4_env25") if t in models]
    for name in bench_tags:
        model = models[name]
        budget = packet_layout(model.method, TARGET_BYTES, stage=model.default_stage, include_envelope=model.use_envelope)
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
            # Warmup
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


def sem_tri_to_rgb(sem: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Render 100x100 BEV RGB where Unknown=dark charcoal, Free=light gray, Occupied=semantic color."""
    # Collapse Z: Unknown if all z==0; Occupied if any z==2; else Free
    has_occ = np.any(tri == 2, axis=2)
    has_free = np.any(tri == 1, axis=2)
    # Dominant non-zero class along Z with Vehicles prioritized for visibility
    sem_bev = np.max(sem, axis=2)
    has_veh = np.any((sem == VEHICLE) & (tri == 2), axis=2)
    has_road = np.any((sem == ROAD) & (tri == 2), axis=2)

    img = np.full((100, 100, 3), [35, 35, 48], dtype=np.uint8)  # Unknown = dark slate
    img[has_free & ~has_occ] = [235, 240, 245]  # Observed Free = crisp white/light-blue
    # Palette for occupied semantic classes
    palette = {
        1: [140, 140, 160],  # Building
        2: [180, 140, 90],   # Fence
        3: [152, 200, 120],  # Terrain
        4: [255, 220, 60],   # Pole
        5: [130, 120, 165],  # Road
        6: [175, 130, 115],  # SideWalk
        7: [60, 155, 75],    # Vegetation
        8: [230, 45, 55],    # Vehicles (bright red!)
        9: [110, 110, 130],  # Wall
        10: [210, 175, 70],  # GuardRail
        11: [255, 140, 0],   # TrafficSign
        12: [100, 160, 200], # Bridge
    }
    for cls_id, rgb in palette.items():
        mask = has_occ & (sem_bev == cls_id)
        img[mask] = rgb
    img[has_road & ~has_veh] = palette[5]
    img[has_veh] = palette[8]
    return np.transpose(img, (1, 0, 2))


def predict_single(model: BorrowedSight, fr: dict, target_bytes: int, device: torch.device, sender_k=None, drop_p=0.0):
    budget = packet_layout(model.method, target_bytes, stage=model.default_stage, include_envelope=model.use_envelope)
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
        pred = model(pose, stamp, motion, [blobs])
        sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
        tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
    return sem, tri


def render_qualitative_suite(models: dict[str, BorrowedSight], clear_frames: list[dict], fog_frames: list[dict], rain_frames: list[dict], device: torch.device) -> None:
    """Render the 4 required visual scenarios + failure case (Acceptance Section 7 & B7)."""
    QUAL.mkdir(parents=True, exist_ok=True)
    # Pick a rich multi-sender frame with multiple vehicles
    multi = [i for i, fr in enumerate(clear_frames) if len(fr["senders"]) >= 3 and fr["n_vehicles"] >= 4]
    if not multi:
        multi = [i for i, fr in enumerate(clear_frames) if len(fr["senders"]) >= 2]
    idx_main = multi[len(multi) // 2] if multi else 0
    fr_c = clear_frames[idx_main]
    fr_f = fog_frames[idx_main] if idx_main < len(fog_frames) else fr_c
    fr_r = rain_frames[idx_main] if idx_main < len(rain_frames) else fr_c

    # Scenario 1: Clear comparison (GT vs B0 vs B3 vs B4 at exact 32KB)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), constrained_layout=True)
    panels = [
        ("GT (Neighbor-Observable)", fr_c["gt_semantic"], fr_c["gt_tristate"]),
        ("B0 (No Comm, 0 B)", *predict_single(models["B0"], fr_c, TARGET_BYTES, device)),
        ("B3 (Shared VQ, 32 KB)", *predict_single(models["B3"], fr_c, TARGET_BYTES, device)),
        ("B4 (BorrowedSight, 32 KB)", *predict_single(models["B4"], fr_c, TARGET_BYTES, device)),
    ]
    for ax, (title, s, t) in zip(axes, panels):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=35, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Scenario 1 (Clear): Real OPV2V {fr_c['scene_name']} #{fr_c['stem']} (Dark=Unknown, White=Free, Red=Vehicle, Purple=Road)")
    fig.savefig(QUAL / "scenario1_clear_comparison.png", dpi=180)
    fig.savefig(QUAL / "scenario1_clear_comparison.pdf")
    plt.close(fig)

    # Scenario 2: Adverse LISA Weather (Clear vs Fog vs Rain for B4)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), constrained_layout=True)
    panels2 = [
        ("GT Clear", fr_c["gt_semantic"], fr_c["gt_tristate"]),
        ("B4 Clear", *predict_single(models["B4"], fr_c, TARGET_BYTES, device)),
        ("B4 LISA Fog", *predict_single(models["B4"], fr_f, TARGET_BYTES, device)),
        ("B4 LISA Rain", *predict_single(models["B4"], fr_r, TARGET_BYTES, device)),
    ]
    for ax, (title, s, t) in zip(axes, panels2):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=35, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_title(f"{title}\n(Unk={float((t==0).mean()):.2f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Scenario 2 (LISA Adverse Weather): Real OPV2V {fr_c['scene_name']} #{fr_c['stem']} — Honest Unknown Expansion")
    fig.savefig(QUAL / "scenario2_lisa_weather.png", dpi=180)
    fig.savefig(QUAL / "scenario2_lisa_weather.pdf")
    plt.close(fig)

    # Scenario 3: Multi-Sender Congestion & Occlusion Clearing (1 vs 2 vs 3 senders)
    n_s = min(3, len(fr_c["senders"]))
    fig, axes = plt.subplots(1, n_s + 1, figsize=(3.6 * (n_s + 1), 3.6), constrained_layout=True)
    axes[0].imshow(sem_tri_to_rgb(fr_c["gt_semantic"], fr_c["gt_tristate"]), origin="lower", interpolation="nearest")
    axes[0].scatter([50], [50], c="cyan", s=35, marker="^", edgecolors="black", linewidths=0.6)
    axes[0].set_title("GT (All Senders Union)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for k in range(1, n_s + 1):
        s, t = predict_single(models["B4"], fr_c, TARGET_BYTES, device, sender_k=k)
        axes[k].imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        axes[k].scatter([50], [50], c="cyan", s=35, marker="^", edgecolors="black", linewidths=0.6)
        axes[k].set_title(f"B4 ({k} Sender{'s' if k>1 else ''})\n(Unk={float((t==0).mean()):.2f})", fontsize=9)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    fig.suptitle(f"Scenario 3 (Urban Congestion & Multi-Sender Fusion): {fr_c['scene_name']} #{fr_c['stem']}")
    fig.savefig(QUAL / "scenario3_congestion_senders.png", dpi=180)
    fig.savefig(QUAL / "scenario3_congestion_senders.pdf")
    plt.close(fig)

    # Scenario 4: Packet Loss & Honest Failure Case (blind zone / 50% drop / 0 senders)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), constrained_layout=True)
    panels4 = [
        ("B4 0% Drop", *predict_single(models["B4"], fr_c, TARGET_BYTES, device, drop_p=0.0)),
        ("B4 25% Block Drop", *predict_single(models["B4"], fr_c, TARGET_BYTES, device, drop_p=0.25)),
        ("B4 50% Block Drop", *predict_single(models["B4"], fr_c, TARGET_BYTES, device, drop_p=0.50)),
        ("Failure: Link Outage (0 Senders)", *predict_single(models["B4"], fr_c, TARGET_BYTES, device, sender_k=0)),
    ]
    for ax, (title, s, t) in zip(axes, panels4):
        ax.imshow(sem_tri_to_rgb(s, t), origin="lower", interpolation="nearest")
        ax.scatter([50], [50], c="cyan", s=35, marker="^", edgecolors="black", linewidths=0.6)
        ax.set_title(f"{title}\n(Unk={float((t==0).mean()):.2f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Scenario 4 (Degraded V2X & Honest Outage Failure): {fr_c['scene_name']} #{fr_c['stem']}")
    fig.savefig(QUAL / "scenario4_degradation_failure.png", dpi=180)
    fig.savefig(QUAL / "scenario4_degradation_failure.pdf")
    plt.close(fig)


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")

    print("Loading Real OPV2V test & validate caches (Clear, LISA Fog, LISA Rain)...", flush=True)
    test_clear = load_cached_split(CACHE_ROOT / "opv2v_clear" / "test")
    test_fog = load_cached_split(CACHE_ROOT / "opv2v_lisa_fog" / "test")
    test_rain = load_cached_split(CACHE_ROOT / "opv2v_lisa_rain" / "test")
    val_clear = load_cached_split(CACHE_ROOT / "opv2v_clear" / "validate")
    print(f"Loaded test_clear={len(test_clear)}, test_fog={len(test_fog)}, test_rain={len(test_rain)}, val_clear={len(val_clear)}", flush=True)

    models = {}
    for tag in (
        "B0", "B1", "B2", "B3", "B4",
        "B4_no_envelope", "B4_shared_book", "B4_no_quota", "B4_stage2",
        "B4_env25", "B4_multibw", "B4_env25_multibw", "B4_no_cls_embed",
        "B4_K16", "B4_K64",
    ):
        if (CKPT_DIR / f"{tag}_s0.pt").exists():
            models[tag], _ = load_ckpt_model(tag, 0, cfg, device)

    test_lookup = build_frame_lookup(test_clear)
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

    # 1. Bandwidth sweep (2KB..32KB) for B1, B2, B3, B4 (+ B4_multibw, B4_env25, B4_env25_multibw) on held-out TEST
    print("\n--- 1. Bandwidth sweep (test) ---", flush=True)
    bw_tags = [t for t in ("B1", "B2", "B3", "B4", "B4_env25", "B4_multibw", "B4_env25_multibw") if t in models]
    for tag in bw_tags:
        for bw in BANDWIDTHS:
            m = measure_stress(models[tag], test_clear, test_lookup, bw, device, np.random.default_rng(0))
            push("test", tag, "bandwidth", bw, m)
            print(f"  {tag:16s} bw={bw:5d} bytes={m['bytes_per_frame']:7.1f} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f}", flush=True)

    # 2. LISA Weather (clear, fog, rain) on held-out TEST + exact per-class IoU table
    print("\n--- 2. LISA Weather (test) ---", flush=True)
    per_class_summary = {}
    w_tags = [t for t in ("B1", "B2", "B3", "B4", "B4_no_envelope", "B4_stage2", "B4_env25", "B4_multibw", "B4_env25_multibw", "B4_no_cls_embed") if t in models]
    for tag in w_tags:
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
            print(
                f"  {tag:16s} {wname:5s} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} "
                f"UFE={m['unobserved_as_free_err']:.4f} unkFrac={m['pred_unknown_frac']:.4f} unc={m['mean_uncertainty']:.4f}",
                flush=True,
            )

    # 3. Multi-sender scaling (1..4 senders on held-out TEST frames with >=3 and >=4 senders)
    print("\n--- 3. Multi-sender scaling ---", flush=True)
    multi4_test = [fr for fr in test_clear if len(fr["senders"]) >= 4]
    multi3_test = [fr for fr in test_clear if len(fr["senders"]) >= 3]
    ms_sets = []
    if len(multi3_test) >= 20:
        ms_sets.append(("test_multi3", multi3_test, 3))
    if len(multi4_test) >= 20:
        ms_sets.append(("test_multi4", multi4_test, 4))
    if not ms_sets:
        ms_sets.append(("test_all", test_clear, 2))
    for ms_split, ms_frames, max_k in ms_sets:
        ms_lookup = build_frame_lookup(ms_frames)
        for tag in [t for t in ("B3", "B4", "B4_env25") if t in models]:
            for k in range(1, max_k + 1):
                m = measure_stress(models[tag], ms_frames, ms_lookup, TARGET_BYTES, device, np.random.default_rng(0), sender_k=k)
                if m["n_frames"] == 0 or m["mIoU"] is None:
                    continue
                push(ms_split, tag, "senders", k, m)
                print(
                    f"  [{ms_split}] {tag:8s} senders={k} (n={m['n_frames']}) mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} "
                    f"unkFrac={m['pred_unknown_frac']:.4f} unc={m['mean_uncertainty']:.4f}",
                    flush=True,
                )

    # 4. Urban Congestion Density Stratification (sparse / medium / dense) on TEST
    print("\n--- 4. Traffic Density / Congestion Stratification (test) ---", flush=True)
    # Use data-driven tertiles if dense (>8) is small, or standard buckets
    veh_counts = [fr["n_vehicles"] for fr in test_clear]
    q33, q66 = float(np.percentile(veh_counts, 33)), float(np.percentile(veh_counts, 66))
    dens_filters = [
        ("sparse", lambda fr, a=q33: fr["n_vehicles"] <= a),
        ("medium", lambda fr, a=q33, b=q66: a < fr["n_vehicles"] <= b),
        ("dense", lambda fr, b=q66: fr["n_vehicles"] > b),
    ]
    for tag in ("B3", "B4"):
        for dname, fn in dens_filters:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(0), filter_fn=fn)
            if m["n_frames"] > 0:
                push("test", tag, "density", dname, m)
                print(f"  {tag:3s} density={dname:6s} (n={m['n_frames']:3d}) mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f}", flush=True)

    # 5. Pose noise, Latency, Packet Loss on held-out TEST
    print("\n--- 5. Pose / Latency / Packet Loss (test) ---", flush=True)
    for tag in [t for t in ("B3", "B4", "B4_no_envelope", "B4_env25") if t in models]:
        for xy_std, yaw_std in POSES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(1), xy_std=xy_std, yaw_std=yaw_std)
            push("test", tag, "pose", f"{xy_std}_{yaw_std}", m)
            print(f"  {tag:14s} pose={xy_std}_{yaw_std} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f} unc={m['mean_uncertainty']:.4f}", flush=True)
        for d in DELAYS:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(0), delay=d)
            push("test", tag, "latency", d, m)
            print(f"  {tag:14s} latency={d} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f} unc={m['mean_uncertainty']:.4f}", flush=True)
        for dp in DROP_RATES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(2), drop_p=dp)
            push("test", tag, "packet_loss", dp, m)
            print(f"  {tag:14s} drop={dp:.2f} bytes={m['bytes_per_frame']:7.1f} mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} UFE={m['unobserved_as_free_err']:.4f} unc={m['mean_uncertainty']:.4f}", flush=True)

    # 6. Compute & Latency Benchmark (B6)
    print("\n--- 6. Compute / Latency / Peak VRAM Benchmark (B6) ---", flush=True)
    compute_bench = benchmark_compute(models, multi4_test or multi3_test or test_clear, device)
    print(json.dumps(compute_bench, indent=2), flush=True)

    # 7. Qualitative 4-Scenario Visualizations + Failure Case
    print("\n--- 7. Rendering 4-Scenario Qualitative Visualizations ---", flush=True)
    render_qualitative_suite(models, test_clear, test_fog, test_rain, device)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (OUT / "measurements.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (OUT / "compute_benchmark.json").write_text(json.dumps(compute_bench, indent=2), encoding="utf-8")
    (OUT / "per_class_iou.json").write_text(json.dumps(per_class_summary, indent=2), encoding="utf-8")
    (OUT / "density_thresholds.json").write_text(
        json.dumps({"q33_vehicles": q33, "q66_vehicles": q66, "min": int(min(veh_counts)), "max": int(max(veh_counts))}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDONE Phase 5 Real OPV2V: wrote {len(rows)} rows to {OUT / 'measurements.csv'} and 4 qualitative figures to {QUAL}", flush=True)


if __name__ == "__main__":
    main()
