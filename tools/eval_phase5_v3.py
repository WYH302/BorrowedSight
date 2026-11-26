"""Phase 5 (BorrowedSight v3.0): Complete 6-axis stress evaluation, SS-KFW velocity-stratified
latency compensation analysis, 64-entry grouped codebook prototype audit, FP32+BF16 compute
profiling, per-class IoU breakdown, and 5-CAV qualitative predictions saved to `outputs/real_opv2v/v3_phase5/`.
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.codec import decode_tokens, demorton3
from borrowedsight.kinematics import sender_coords_to_world
from borrowedsight.labels import GROUND, N_CLASS, THINGS, VEHICLE, VERTICAL
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.packer import TARGET_BYTES, drop_block, packet_layout, unpack_bytes
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.eval_phase5_v21 import (
    BANDWIDTHS,
    DELAYS,
    DROP_RATES,
    POSES,
    _drop_blocks,
    _pose_copy,
    build_frame_lookup,
    measure_stress,
    predict_single,
)
from tools.train_receivers_real import load_cached_split, move_sender

CACHE_ROOT = DATA_ROOT / "cache"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v3"
OUT = OUTPUTS / "real_opv2v" / "v3_phase5"


def load_ckpt_v3(tag: str, seed: int, cfg: dict, device: torch.device) -> tuple[BorrowedSight, dict]:
    path = CKPT_DIR / f"{tag}_s{seed}.pt"
    blob = torch.load(path, map_location=device, weights_only=False)
    mkw = dict(blob.get("model_kwargs") or {})
    enc_kw = blob.get("encoder_kwargs") or {}
    cb_size = int(mkw.pop("codebook_size", 64))
    mkw.setdefault("sender_norm_pe", bool(enc_kw.get("use_norm_pe", False)))
    mkw.setdefault("sender_max_pool", bool(enc_kw.get("use_max_pool", False)))
    model = BorrowedSight(
        blob["method"],
        feat_dim=int(enc_kw.get("feat_dim", cfg["feat_dim"])),
        codebook_size=cb_size,
        **mkw,
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


def _extract_veh_world_from_blob(blob: bytes) -> tuple[np.ndarray, float]:
    view = unpack_bytes(blob)
    if view["method"] != "B4":
        return np.zeros((0, 3), dtype=np.float32), float(view["timestamp_s"])
    tokens = decode_tokens(view["method"], view["tokens"], stage=int(view["stage"]), compact=bool(view.get("compact", False)))
    morton = tokens["morton"]
    if morton.size == 0 or "cls" not in tokens:
        return np.zeros((0, 3), dtype=np.float32), float(view["timestamp_s"])
    veh_mask = np.asarray(tokens["cls"], dtype=np.int64) == VEHICLE
    if not bool(veh_mask.any()):
        return np.zeros((0, 3), dtype=np.float32), float(view["timestamp_s"])
    coords_np = demorton3(morton[veh_mask])
    pts_w = sender_coords_to_world(coords_np, np.asarray(view["pose"], dtype=np.float32))
    return pts_w, float(view["timestamp_s"])


def measure_latency_with_sskfw(
    model: BorrowedSight,
    frames: list[dict],
    frame_lookup: dict[tuple[str, int], dict],
    target_bytes: int,
    delay: int,
    device: torch.device,
    use_ss_kfw: bool = True,
) -> tuple[dict, dict]:
    """Strict wire-byte latency evaluation with optional zero-byte SS-KFW + velocity-stratified breakdown."""
    model.eval()
    budget = packet_layout(
        model.method,
        target=int(target_bytes),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    n_tokens = int(budget["n_tokens"])
    acc_all = new_counts()
    # Stratify by mean neighbor speed in frame (proxy for scene traffic velocity)
    bins = {"low_0_2mps": new_counts(), "med_2_8mps": new_counts(), "high_gt8mps": new_counts()}
    bin_counts = {k: 0 for k in bins}
    n_eval = 0

    with torch.no_grad():
        for fr in frames:
            src_idx = fr["frame"] - int(delay)
            src = frame_lookup.get((fr["scene_name"], src_idx)) if delay > 0 else fr
            if src is None or not src["senders"]:
                continue
            # Also retrieve 1-step (0.1s) and 2-step (0.2s) prior buffered packets per sender_id for SS-KFW
            prior_map: dict[int, list[tuple[np.ndarray, float]]] = {}
            if use_ss_kfw and delay > 0 and model.method == "B4":
                for lag in (2, 1):
                    prev_fr = frame_lookup.get((fr["scene_name"], src_idx - lag))
                    if prev_fr is None:
                        continue
                    for prow in prev_fr["senders"]:
                        psent = model.transmit(
                            move_sender(prow, device),
                            fr["ego_pose"],
                            int(fr["receiver_id"]),
                            n_tokens,
                            budget=budget,
                            serialize=True,
                        )
                        if psent["blob"]:
                            p_veh_w, p_ts = _extract_veh_world_from_blob(psent["blob"])
                            prior_map.setdefault(int(prow["sender_id"]), []).append((p_veh_w, p_ts))

            pose_t = torch.as_tensor(fr["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
            stamp_t = torch.tensor([float(fr["timestamp"])], dtype=torch.float32, device=device)
            motion_t = torch.as_tensor(fr["motion"], dtype=torch.float32, device=device).reshape(1, 2)

            placed_items = []
            speeds_fr = []
            for row in src["senders"]:
                speeds_fr.append(abs(float(row.get("speed", 0.0))))
                sent = model.transmit(
                    move_sender(row, device),
                    fr["ego_pose"],
                    int(fr["receiver_id"]),
                    n_tokens,
                    budget=budget,
                    serialize=True,
                )
                blob = sent["blob"]
                if not blob:
                    continue
                prev_list = None
                if use_ss_kfw and delay > 0 and model.method == "B4":
                    sid = int(row["sender_id"])
                    raw_priors = prior_map.get(sid, [])
                    cur_ts = float(row["timestamp"])
                    prev_list = [(pts, cur_ts - pts_ts) for (pts, pts_ts) in raw_priors if cur_ts - pts_ts > 1e-3]
                placed_items.append(
                    model._placed_from_blob(blob, pose_t[0], float(fr["timestamp"]), prev_packets_veh_w=prev_list)
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.render([placed_items], pose_t, stamp_t, motion_t)
            sem = pred["sem"].argmax(dim=1)[0].detach().cpu().numpy()
            tri = pred["tri"].argmax(dim=1)[0].detach().cpu().numpy()
            tri_p = F.softmax(pred["tri"].float(), dim=1)[0].detach().cpu().numpy()
            update_counts(acc_all, sem, tri, fr["gt_semantic"], fr["gt_tristate"], tri_probs=tri_p)
            n_eval += 1

            mean_spd = float(np.mean(speeds_fr)) if speeds_fr else 0.0
            bkey = "low_0_2mps" if mean_spd < 2.0 else ("med_2_8mps" if mean_spd <= 8.0 else "high_gt8mps")
            update_counts(bins[bkey], sem, tri, fr["gt_semantic"], fr["gt_tristate"], tri_probs=tri_p)
            bin_counts[bkey] += 1

    overall = finalize_counts(acc_all)
    overall["n_frames"] = n_eval
    stratified = {}
    for bk, acc_b in bins.items():
        if bin_counts[bk] > 0:
            mb = finalize_counts(acc_b)
            stratified[bk] = {
                "n_frames": bin_counts[bk],
                "mIoU": round(float(mb["mIoU"]), 4),
                "vehicle_IoU": round(float(mb["vehicle_IoU"]), 4),
                "road_IoU": round(float(mb["road_IoU"]), 4),
                "UFE": round(float(mb["unobserved_as_free_err"]), 4),
            }
    return overall, stratified


def analyze_codebooks(model_b4: BorrowedSight, model_b3: BorrowedSight | None, frames: list[dict], device: torch.device) -> dict:
    """Deep Mechanism Audit: per-group 64-entry utilization, normalized Shannon entropy,
    class x prototype activation heatmap, and inter-book cosine orthogonality."""
    out = {}
    for name, mod in [("B4_v3_32k", model_b4), ("B3_v3_32k", model_b3)]:
        if mod is None:
            continue
        mod.eval()
        K = int(mod.codebook_size)
        bgt = packet_layout(mod.method, TARGET_BYTES, stage=mod.default_stage, include_envelope=mod.use_envelope, compact=mod.compact_wire)
        n_tok = int(bgt["n_tokens"])
        hist_cls_c0 = np.zeros((N_CLASS, K), dtype=np.int64)
        hist_cls_c1 = np.zeros((N_CLASS, K), dtype=np.int64)
        with torch.no_grad():
            for fr in frames:
                for row in fr["senders"]:
                    sent = mod.transmit(move_sender(row, device), fr["ego_pose"], int(fr["receiver_id"]), n_tok, budget=bgt, serialize=False)
                    if sent["codes"] is None or sent["classes"].numel() == 0:
                        continue
                    c_np = sent["codes"].detach().cpu().numpy()
                    cls_np = sent["classes"].detach().cpu().numpy()
                    for ci, k0 in zip(cls_np, c_np[:, 0]):
                        if 0 <= ci < N_CLASS and 0 <= k0 < K:
                            hist_cls_c0[ci, k0] += 1
                    if c_np.shape[1] > 1:
                        for ci, k1 in zip(cls_np, c_np[:, 1]):
                            if 0 <= ci < N_CLASS and 0 <= k1 < K:
                                hist_cls_c1[ci, k1] += 1

        def _stats(counts_1d: np.ndarray) -> dict:
            tot = int(counts_1d.sum())
            active = int((counts_1d > 0).sum())
            if tot == 0:
                return {"active": 0, "total": K, "util": 0.0, "norm_entropy": 0.0, "perplexity": 0.0}
            p = counts_1d[counts_1d > 0] / float(tot)
            ent = float(-np.sum(p * np.log2(p)))
            return {
                "active": active,
                "total": K,
                "util": round(active / float(K), 4),
                "norm_entropy": round(ent / float(np.log2(K)), 4),
                "perplexity": round(float(2.0 ** ent), 2),
                "tokens": tot,
            }

        entry = {"K": K, "hist_cls_stage1": hist_cls_c0.tolist(), "hist_cls_stage2": hist_cls_c1.tolist()}
        if mod.method == "B4" and mod.use_grouped_books:
            entry["groups_stage1"] = {
                "things": _stats(hist_cls_c0[list(THINGS)].sum(axis=0)),
                "ground": _stats(hist_cls_c0[list(GROUND)].sum(axis=0)),
                "vertical": _stats(hist_cls_c0[list(VERTICAL)].sum(axis=0)),
            }
            entry["groups_stage2"] = {
                "things": _stats(hist_cls_c1[list(THINGS)].sum(axis=0)),
                "ground": _stats(hist_cls_c1[list(GROUND)].sum(axis=0)),
                "vertical": _stats(hist_cls_c1[list(VERTICAL)].sum(axis=0)),
            }
            # Mean absolute cosine similarity between prototypes within vs across grouped codebooks
            cbs = [F.normalize(book.layers[0].codebook.detach().float(), dim=-1) for book in mod.books]
            sim_mat = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    s = (cbs[i] @ cbs[j].T).abs()
                    if i == j:
                        s.fill_diagonal_(0.0)
                        sim_mat[i, j] = float(s.sum().item() / max(1, K * (K - 1)))
                    else:
                        sim_mat[i, j] = float(s.mean().item())
            entry["inter_book_abs_cos_sim"] = np.round(sim_mat, 4).tolist()
        else:
            entry["monolithic_stage1"] = _stats(hist_cls_c0.sum(axis=0))
            entry["monolithic_stage2"] = _stats(hist_cls_c1.sum(axis=0))
        out[name] = entry
    return out


def benchmark_compute_v3(models: dict[str, BorrowedSight], frames: list[dict], device: torch.device) -> dict:
    """Measure BOTH FP32 and BF16 RTX 3090 inference latency & FPS across K=1..4 senders."""
    multi4 = [fr for fr in frames if len(fr["senders"]) >= 4]
    multi3 = [fr for fr in frames if len(fr["senders"]) >= 3]
    sample = multi4[0] if multi4 else (multi3[0] if multi3 else frames[0])
    out = {}
    for name in [t for t in ("Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k") if t in models]:
        model = models[name]
        budget = packet_layout(model.method, TARGET_BYTES, stage=model.default_stage, include_envelope=model.use_envelope, compact=model.compact_wire)
        n_tok = int(budget["n_tokens"])
        pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
        stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
        motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
        rows = [move_sender(r, device) for r in sample["senders"]]
        out[name] = {"params": sum(p.numel() for p in model.parameters()), "fp32": {}, "bf16": {}}
        for prec, use_bf16 in (("fp32", False), ("bf16", True)):
            for k in (1, 2, 3, 4):
                if k > len(rows):
                    continue
                use = rows[:k]
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(5):
                        blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                        if use_bf16:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                model(pose, stamp, motion, [blobs])
                        else:
                            model(pose, stamp, motion, [blobs])
                torch.cuda.synchronize()
                reps = 20
                tx_acc, rx_acc = 0.0, 0.0
                with torch.no_grad():
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        if use_bf16:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                model(pose, stamp, motion, [blobs])
                        else:
                            model(pose, stamp, motion, [blobs])
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        tx_acc += (t1 - t0) * 1000.0
                        rx_acc += (t2 - t1) * 1000.0
                tot = (tx_acc + rx_acc) / reps
                out[name][prec][str(k)] = {
                    "tx_pack_ms": round(tx_acc / reps, 2),
                    "rx_decode_ms": round(rx_acc / reps, 2),
                    "total_ms": round(tot, 2),
                    "fps": round(1000.0 / max(1e-3, tot), 2),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                    "bytes_per_sender": len(blobs[0]) if blobs else 0,
                }
    return out


def save_qualitative_v3_5cav(models: dict[str, BorrowedSight], test_clear, test_fog, test_rain, device: torch.device) -> None:
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
    for key, tag in (
        ("b0", "B0_v3_32k"),
        ("b1", "B1_v3_32k"),
        ("w2c", "Where2comm_v3_32k"),
        ("cf", "CodeFilling_v3_32k"),
        ("b3", "B3_v3_32k"),
        ("b4", "B4_v3_32k"),
    ):
        if tag in models:
            s, t = predict_single(models[tag], fr_c, TARGET_BYTES, device)
            save_d[f"{key}_sem"] = s
            save_d[f"{key}_tri"] = t
    if "B4_v3_32k" in models:
        m4 = models["B4_v3_32k"]
        for k, (fr_use, sk, dp) in (
            ("b4_fog", (fr_f, None, 0.0)),
            ("b4_rain", (fr_r, None, 0.0)),
            ("b4_drop50", (fr_c, None, 0.5)),
            ("b4_outage", (fr_c, 0, 0.0)),
        ):
            s, t = predict_single(m4, fr_use, TARGET_BYTES, device, sender_k=sk, drop_p=dp)
            save_d[f"{k}_sem"] = s
            save_d[f"{k}_tri"] = t
    np.savez_compressed(OUT / "qualitative_v3_5cav.npz", **save_d)
    print(f"Saved v3 5-CAV qualitative predictions for {fr_c['scene_name']} #{fr_c['stem']}", flush=True)


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    test_clear = load_cached_split(CACHE_ROOT / "opv2v_v3_clear" / "test")
    test_fog = load_cached_split(CACHE_ROOT / "opv2v_v3_lisa_fog" / "test")
    test_rain = load_cached_split(CACHE_ROOT / "opv2v_v3_lisa_rain" / "test")
    test_lookup = build_frame_lookup(test_clear)

    load_tags = (
        "B0_v3_32k", "B1_v3_32k", "B2_v3_32k",
        "Where2comm_v3_32k", "CodeFilling_v3_32k",
        "B3_v3_32k", "B4_v3_32k", "B4_v3_32k_no_env", "B4_v3_multibw",
    )
    models = {}
    for tag in load_tags:
        if (CKPT_DIR / f"{tag}_s0.pt").exists():
            models[tag], _ = load_ckpt_v3(tag, 0, cfg, device)
    print(f"Loaded {len(models)} v3.0 models for Phase 5 evaluation: {list(models.keys())}", flush=True)

    rows = []

    def push(split_name, model_name, family, level, metrics):
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "family": family,
                "level": level,
                "n_frames": metrics.get("n_frames", 0),
                "bytes_mean": round(float(metrics.get("bytes_per_frame", 0.0)), 1),
                "mIoU": round(float(metrics.get("mIoU") or 0.0), 4),
                "fg_mIoU": round(float(metrics.get("fg_mIoU") or 0.0), 4),
                "vehicle_IoU": round(float(metrics.get("vehicle_IoU") or 0.0), 4),
                "road_IoU": round(float(metrics.get("road_IoU") or 0.0), 4),
                "building_IoU": round(float((metrics.get("per_class_iou") or {}).get("building", 0.0)), 4),
                "unknown_F1": round(float(metrics.get("unknown_F1") or 0.0), 4),
                "UFE": round(float(metrics.get("unobserved_as_free_err") or 0.0), 4),
                "tri_brier": round(float(metrics.get("tri_brier") or 0.0), 4),
            }
        )

    # 1. Per-class IoU table (clean 32 KB)
    per_class = {}
    for tag in ("B1_v3_32k", "B2_v3_32k", "Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k", "B4_v3_32k_no_env"):
        if tag in models:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(0))
            per_class[tag] = {
                "mIoU": round(float(m["mIoU"]), 4),
                "fg_mIoU": round(float(m.get("fg_mIoU", m["mIoU"])), 4),
                "unknown_F1": round(float(m["unknown_F1"]), 4),
                "UFE": round(float(m["unobserved_as_free_err"]), 4),
                "tri_brier": round(float(m.get("tri_brier", 0.0)), 4),
                "per_class_iou": {k: round(float(v), 4) for k, v in (m.get("per_class_iou") or {}).items()},
            }
            push("test", tag, "clean_32k", "32768", m)
    (OUT / "per_class_iou.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    print("Saved per_class_iou.json", flush=True)

    # 2. Bandwidth sweep (2 KB .. 32 KB)
    for tag in ("B1_v3_32k", "B2_v3_32k", "Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k", "B4_v3_multibw"):
        if tag not in models:
            continue
        for bw in BANDWIDTHS:
            m = measure_stress(models[tag], test_clear, test_lookup, bw, device, np.random.default_rng(1))
            push("test", tag, "bandwidth", str(bw), m)
    print("Finished bandwidth sweep", flush=True)

    # 3. Weather stress (LISA Fog & Rain)
    for w_name, frames_w in (("lisa_fog", test_fog), ("lisa_rain", test_rain)):
        lk_w = build_frame_lookup(frames_w)
        for tag in ("Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k", "B4_v3_32k_no_env"):
            if tag not in models:
                continue
            m = measure_stress(models[tag], frames_w, lk_w, TARGET_BYTES, device, np.random.default_rng(2))
            push("test", tag, "weather", w_name, m)
    print("Finished weather stress", flush=True)

    # 4. Block drop (0% .. 50%)
    for tag in ("Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k", "B4_v3_32k_no_env"):
        if tag not in models:
            continue
        for dp in DROP_RATES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(3), drop_p=dp)
            push("test", tag, "block_drop", f"{int(dp*100)}pct", m)
    print("Finished block drop stress", flush=True)

    # 5. Pose error
    for tag in ("Where2comm_v3_32k", "CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k"):
        if tag not in models:
            continue
        for xy_s, yaw_s in POSES:
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(4), xy_std=xy_s, yaw_std=yaw_s)
            push("test", tag, "pose_error", f"{xy_s}m_{yaw_s}deg", m)
    print("Finished pose error stress", flush=True)

    # 6. Multi-sender scaling (1..4) & Urban congestion
    for tag in ("CodeFilling_v3_32k", "B3_v3_32k", "B4_v3_32k"):
        if tag not in models:
            continue
        for k in (1, 2, 3, 4):
            m = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(5), sender_k=k)
            if m["n_frames"] > 0:
                push("test", tag, "senders", f"k{k}", m)
        m_low = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(6), filter_fn=lambda f: f["n_vehicles"] <= 2)
        m_hi = measure_stress(models[tag], test_clear, test_lookup, TARGET_BYTES, device, np.random.default_rng(6), filter_fn=lambda f: f["n_vehicles"] >= 4)
        push("test", tag, "congestion", "le2_veh", m_low)
        push("test", tag, "congestion", "ge4_veh", m_hi)
    print("Finished multi-sender & congestion", flush=True)

    # 7. Latency (0, 100, 200, 500 ms): B3, CodeFilling, B4 w/o SS-KFW, AND B4 w/ SS-KFW (+ velocity-stratified analysis!)
    ss_kfw_report = {}
    for d in DELAYS:
        for tag in ("CodeFilling_v3_32k", "B3_v3_32k"):
            if tag in models:
                m, _ = measure_latency_with_sskfw(models[tag], test_clear, test_lookup, TARGET_BYTES, d, device, use_ss_kfw=False)
                push("test", tag, "latency", f"{d*100}ms", m)
        if "B4_v3_32k" in models:
            m_no, strat_no = measure_latency_with_sskfw(models["B4_v3_32k"], test_clear, test_lookup, TARGET_BYTES, d, device, use_ss_kfw=False)
            push("test", "B4_v3_32k_noKFW", "latency", f"{d*100}ms", m_no)
            m_yes, strat_yes = measure_latency_with_sskfw(models["B4_v3_32k"], test_clear, test_lookup, TARGET_BYTES, d, device, use_ss_kfw=True)
            push("test", "B4_v3_32k_SSKFW", "latency", f"{d*100}ms", m_yes)
            ss_kfw_report[f"{d*100}ms"] = {
                "no_kfw": {"mIoU": round(m_no["mIoU"], 4), "veh": round(m_no["vehicle_IoU"], 4), "road": round(m_no["road_IoU"], 4), "UFE": round(m_no["unobserved_as_free_err"], 4), "stratified": strat_no},
                "ss_kfw": {"mIoU": round(m_yes["mIoU"], 4), "veh": round(m_yes["vehicle_IoU"], 4), "road": round(m_yes["road_IoU"], 4), "UFE": round(m_yes["unobserved_as_free_err"], 4), "stratified": strat_yes},
            }
            print(
                f"  [Latency {d*100:3d}ms] B4 no-KFW veh={m_no['vehicle_IoU']:.4f} mIoU={m_no['mIoU']:.4f} | "
                f"SS-KFW veh={m_yes['vehicle_IoU']:.4f} ({m_yes['vehicle_IoU']-m_no['vehicle_IoU']:+.4f}) mIoU={m_yes['mIoU']:.4f}",
                flush=True,
            )
    (OUT / "ss_kfw_analysis.json").write_text(json.dumps(ss_kfw_report, indent=2), encoding="utf-8")

    # Flush measurements.csv
    with (OUT / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUT / 'measurements.csv'}", flush=True)

    # 8. Deep Mechanism Analysis: 64-entry codebook prototype utilization & orthogonality
    if "B4_v3_32k" in models:
        cb_audit = analyze_codebooks(models["B4_v3_32k"], models.get("B3_v3_32k"), test_clear, device)
        (OUT / "codebook_analysis.json").write_text(json.dumps(cb_audit, indent=2), encoding="utf-8")
        print("Saved codebook_analysis.json", flush=True)

    # 9. Compute Benchmark (FP32 + BF16)
    comp = benchmark_compute_v3(models, test_clear, device)
    (OUT / "compute_benchmark.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print("Saved compute_benchmark.json", flush=True)

    # 10. Qualitative 5-CAV predictions
    save_qualitative_v3_5cav(models, test_clear, test_fog, test_rain, device)
    print("=== PHASE 5 V3 COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
