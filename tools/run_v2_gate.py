"""Step-2 Stop-Loss Fast Validation Gate for BorrowedSight v2 (Tier B).

Trains 5 single-seed (seed 0) models in isolated directories (`checkpoints/real_opv2v_v2`
and `outputs/real_opv2v/v2_gate`) without touching any existing v1 weights or results:
  1) B3_v2_32k        : Fair compact 4B/token Shared-VQ + GroupNorm/2D-PE + Lovasz/UFE + D4/impairment aug
  2) B4_v2_32k        : Full BorrowedSight v2 (5B/token w/ free Stage-2, true residual z-E_cls,
                        50x50 adaptive Zstd env + 25x25 z_max, GroupNorm/2D-PE, exp(-age/0.5),
                        road_floor=0.10, Lovasz+UFE, 8x D4 BEV aug + mild drop/delay/pose aug)
  3) B4_v2_32k_no_env : Exact same B4_v2_32k WITHOUT visibility envelope or z_max
  4) B3_v2_16k        : Fair compact 4B/token Shared-VQ at 16,384 B
  5) B4_v2_16k        : Full BorrowedSight v2 at 16,384 B

Immediately evaluates all 5 models on:
  - Clean Held-Out Test (exact 0.0% byte discrepancy)
  - 100 / 200 / 500 ms V2X Latency (real delayed frames)
  - 50% Independent Micro-Block Drop
  - 0.5m / 2.0 deg Pose Error
  - LISA Fog & LISA Rain held-out physical weather caches
  - RTX 3090 End-to-End Serialized FPS & VRAM
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.labels import IGNORE_INDEX
from borrowedsight.losses import asymmetric_ufe_loss, bev_d4_augment, lovasz_softmax_masked
from borrowedsight.metrics import finalize_counts, new_counts, update_counts
from borrowedsight.model import BorrowedSight, _force_unknown_where, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, TARGET_BYTES, drop_block, packet_layout, unpack_bytes
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.train_receivers_real import (
    _road_fraction,
    _usage,
    compute_class_weights,
    load_cached_split,
    move_sender,
    set_seed,
)

CACHE_ROOT = DATA_ROOT / "cache"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v2"
OUT = OUTPUTS / "real_opv2v" / "v2_gate"


def _pose_jitter(row: dict, rng: np.random.Generator, xy_std: float, yaw_std: float) -> dict:
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


def _drop_placed_tokens(item: dict, drop_p: float, rng: np.random.Generator) -> dict:
    """Fast in-memory block-like random dropout during training (multiples of 16 tokens)."""
    coords = item["coords_ego"]
    n = int(coords.shape[0])
    if n == 0 or drop_p <= 0.0:
        return item
    n_blocks = (n + 15) // 16
    keep_b = rng.random(n_blocks) >= drop_p
    if not np.any(keep_b):
        keep_b[int(rng.integers(0, n_blocks))] = True
    keep_tok = np.repeat(keep_b, 16)[:n]
    mask = torch.as_tensor(keep_tok, dtype=torch.bool, device=coords.device)
    out = dict(item)
    out["coords_ego"] = coords[mask]
    out["values"] = item["values"][mask]
    out["conf"] = item["conf"][mask]
    return out


def eval_serialized(
    model: BorrowedSight,
    frames: list[dict],
    frame_lookup: dict[tuple[str, int], dict],
    target_bytes: int,
    device: torch.device,
    rng: np.random.Generator,
    xy_std: float = 0.0,
    yaw_std: float = 0.0,
    delay: int = 0,
    drop_p: float = 0.0,
    eval_batch: int = 8,
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
    codes = []
    classes = []
    with torch.no_grad():
        for start in range(0, len(frames), eval_batch):
            chunk = frames[start : start + eval_batch]
            pose_b = torch.as_tensor(np.stack([fr["ego_pose"] for fr in chunk]), dtype=torch.float32, device=device)
            stamp_b = torch.tensor([float(fr["timestamp"]) for fr in chunk], dtype=torch.float32, device=device)
            motion_b = torch.as_tensor(np.stack([fr["motion"] for fr in chunk]), dtype=torch.float32, device=device)
            blobs_b = []
            for fr in chunk:
                src = frame_lookup.get((fr["scene_name"], fr["frame"] - int(delay))) if delay > 0 else fr
                senders = list(src["senders"]) if src is not None else []
                blobs = []
                for row in senders:
                    noisy = _pose_jitter(row, rng, xy_std, yaw_std)
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
                    if sent["code"] is not None and sent["classes"].numel():
                        codes.append(sent["code"].detach().cpu().numpy())
                        classes.append(sent["classes"].detach().cpu().numpy())
                blobs_b.append(blobs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(pose_b, stamp_b, motion_b, blobs_b)
            sems = pred["sem"].argmax(dim=1).detach().cpu().numpy()
            tris = pred["tri"].argmax(dim=1).detach().cpu().numpy()
            tri_probs_b = F.softmax(pred["tri"].float(), dim=1).detach().cpu().numpy()
            for bi, fr in enumerate(chunk):
                update_counts(acc, sems[bi], tris[bi], fr["gt_semantic"], fr["gt_tristate"], tri_probs=tri_probs_b[bi])
    m = finalize_counts(acc)
    m["bytes_per_frame"] = float(np.mean(sizes)) if sizes else 0.0
    m["bytes_min"] = float(np.min(sizes)) if sizes else 0.0
    m["bytes_max"] = float(np.max(sizes)) if sizes else 0.0
    m["n_packets"] = int(len(sizes))
    m["codebook_usage"] = _usage(model, codes, classes)
    m["road_token_fraction"] = _road_fraction(classes)
    return m


def train_v2_variant(
    tag: str,
    method: str,
    model_kwargs: dict,
    target_bytes: int,
    seed: int,
    train_frames: list[dict],
    val_frames: list[dict],
    test_frames: list[dict],
    class_weights: torch.Tensor,
    cfg: dict,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 4,
    lovasz_w: float = 0.5,
    ufe_w: float = 0.35,
) -> tuple[BorrowedSight, dict]:
    set_seed(seed)
    mkw = dict(model_kwargs)
    cb_size = int(mkw.pop("codebook_size", cfg["codebook_size"]))
    model = BorrowedSight(
        method,
        feat_dim=int(cfg["feat_dim"]),
        codebook_size=cb_size,
        **mkw,
    ).to(device)
    ckpt = torch.load(CHECKPOINTS / "sender_opv2v.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt["encoder"])
    for p in model.encoder.parameters():
        p.requires_grad = False

    budget = packet_layout(
        method,
        target=int(target_bytes),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    n_tokens = int(budget["n_tokens"])
    total_params = sum(p.numel() for p in model.parameters())
    train_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in train_frames}
    val_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in val_frames}
    test_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in test_frames}

    print(
        f"\n=== V2 GATE TRAIN {tag} (method={method}, params={total_params:,}, bytes={budget['serialized']}, "
        f"tok_w={budget['token_bytes']}B, n_tok={n_tokens}, env={model.use_envelope}({model.envelope_cells}), "
        f"stage={model.default_stage}, true_res={model.true_residual}, zmax={model.use_zmax_env}) ===",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    rng = np.random.default_rng(seed + 100)
    vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    best_score = -1.0
    best_state = None
    status = "running"

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_frames))
        ep_losses = []
        ep_t0 = time.perf_counter()
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            placed_b = []
            poses_b = []
            stamps_b = []
            motions_b = []
            sem_ts = []
            tri_ts = []
            vq_terms = []
            score_terms = []
            for idx in batch_ids:
                fr = train_frames[int(idx)]
                # Mild real-world V2X impairment exposure (~30% impaired, ~70% pure clean):
                # 12% real temporal lag (1-5 frames), 15% random block drop (10-35%), 6% mild pose jitter
                u = float(rng.random())
                src_fr = fr
                drop_p_step = 0.0
                xy_j, yaw_j = 0.0, 0.0
                if u < 0.12:
                    d_choice = int(rng.choice([1, 2, 3, 5]))
                    cand = train_lookup.get((fr["scene_name"], fr["frame"] - d_choice))
                    if cand is not None and len(cand["senders"]) > 0:
                        src_fr = cand
                elif u < 0.27:
                    drop_p_step = float(rng.uniform(0.10, 0.35))
                elif u < 0.33:
                    xy_j, yaw_j = 0.25, 1.0

                sent = []
                for row in src_fr["senders"]:
                    row_use = _pose_jitter(row, rng, xy_j, yaw_j) if xy_j > 0.0 else row
                    sent.append(
                        model.transmit(
                            move_sender(row_use, device),
                            fr["ego_pose"],
                            int(fr["receiver_id"]),
                            n_tokens,
                            budget=budget,
                            serialize=False,
                        )
                    )
                non_empty = [s for s in sent if s["blob"]]
                if not non_empty:
                    continue
                pose_i = torch.as_tensor(fr["ego_pose"], dtype=torch.float32, device=device)
                items_i = [model.placed_from_transmit(s, pose_i, float(fr["timestamp"])) for s in non_empty]
                if drop_p_step > 0.0:
                    items_i = [_drop_placed_tokens(it, drop_p_step, rng) for it in items_i]
                placed_b.append(items_i)
                poses_b.append(pose_i)
                stamps_b.append(float(fr["timestamp"]))
                motions_b.append(fr["motion"])
                sem_ts.append(fr["sem_target_t"].to(device, dtype=torch.long, non_blocking=True))
                tri_ts.append(fr["tri_target_t"].to(device, dtype=torch.long, non_blocking=True))
                for s in sent:
                    if torch.is_tensor(s["vq_loss"]):
                        vq_terms.append(s["vq_loss"])
                    if s["score_logit"].numel() > 0:
                        score_terms.append(score_bce(s["score_logit"], s["all_classes"]))
            if not placed_b:
                continue
            pose = torch.stack(poses_b, dim=0)
            stamp = torch.tensor(stamps_b, dtype=torch.float32, device=device)
            motion = torch.as_tensor(np.stack(motions_b), dtype=torch.float32, device=device)
            sem_t = torch.stack(sem_ts, dim=0)
            tri_t = torch.stack(tri_ts, dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feat, extra, nonempty = model.build_inputs(placed_b, pose, stamp, motion)
                if model.decouple_env and extra.shape[1] >= 8:
                    # Match ~16% Zstd z_max fallback + mild envelope dropout during training.
                    for bi in range(extra.shape[0]):
                        r_env = float(rng.random())
                        if r_env < 0.16:
                            extra[bi, 7] = 0.0
                        elif r_env < 0.24:
                            extra[bi, 0:2] = 0.0
                            extra[bi, 2] = 1.0
                            extra[bi, 7] = 0.0
                feat, extra, sem_t, tri_t = bev_d4_augment(feat, extra, sem_t, tri_t, rng)
                pred = _force_unknown_where(model.decoder(feat, extra), torch.logical_not(nonempty))

            vq = torch.stack(vq_terms).mean() if vq_terms else pose.new_zeros(())
            sc_loss = torch.stack(score_terms).mean() if score_terms else pose.new_zeros(())
            loss_sem_ce = F.cross_entropy(pred["sem"].float(), sem_t, weight=class_weights, ignore_index=IGNORE_INDEX)
            loss_sem_lov = lovasz_softmax_masked(pred["sem"].float(), sem_t, ignore_index=IGNORE_INDEX)
            loss_tri = F.cross_entropy(pred["tri"].float(), tri_t)
            loss_ufe = asymmetric_ufe_loss(pred["tri"].float(), tri_t)
            loss = loss_sem_ce + lovasz_w * loss_sem_lov + 0.5 * loss_tri + ufe_w * loss_ufe + vq + 0.05 * sc_loss
            if not torch.isfinite(loss):
                status = "nan"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
            if torch.cuda.max_memory_allocated() >= vram_cap:
                status = "vram_abort"
                break
        if status in {"nan", "vram_abort"}:
            break
        sched.step()
        ep_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        if ep == 1 or ep % 4 == 0 or ep >= epochs - 2:
            val_m = eval_serialized(model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
            score = float(val_m["mIoU"] or 0.0)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  [{tag}] ep {ep:02d}/{epochs} loss={ep_mean:.4f} "
                f"val_mIoU={val_m['mIoU']:.4f} veh={val_m['vehicle_IoU']:.4f} "
                f"road={val_m['road_IoU']:.4f} unkF1={val_m['unknown_F1']:.4f} "
                f"UFE={val_m['unobserved_as_free_err']:.4f} "
                f"usage={val_m['codebook_usage']} roadTok={val_m['road_token_fraction']} "
                f"dt={time.perf_counter()-ep_t0:.1f}s",
                flush=True,
            )

    if status == "running":
        status = "converged"
    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = eval_serialized(model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
    test_metrics = eval_serialized(model, test_frames, test_lookup, target_bytes, device, np.random.default_rng(0))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "tag": tag,
            "method": method,
            "model_kwargs": model_kwargs,
            "target_bytes": int(target_bytes),
            "seed": int(seed),
            "status": status,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        CKPT_DIR / f"{tag}_s{seed}.pt",
    )
    rec = {
        "tag": tag,
        "method": method,
        "params": total_params,
        "target_bytes": int(target_bytes),
        "token_bytes": int(budget["token_bytes"]),
        "n_tokens": n_tokens,
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(float(torch.cuda.max_memory_allocated() / 1024**3), 3),
        "status": status,
        "val": val_metrics,
        "test": test_metrics,
    }
    print(
        f"DONE [{tag}] time={rec['seconds']}s vram={rec['peak_vram_gb']}GB | "
        f"VAL mIoU={val_metrics['mIoU']:.4f} | TEST mIoU={test_metrics['mIoU']:.4f} "
        f"veh={test_metrics['vehicle_IoU']:.4f} road={test_metrics['road_IoU']:.4f} "
        f"unkF1={test_metrics['unknown_F1']:.4f} UFE={test_metrics['unobserved_as_free_err']:.4f} "
        f"bytes={test_metrics['bytes_per_frame']:.0f}",
        flush=True,
    )
    return model, rec


def bench_speed(model: BorrowedSight, sample: dict, target_bytes: int, device: torch.device) -> dict:
    budget = packet_layout(
        model.method,
        target=int(target_bytes),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    n_tok = int(budget["n_tokens"])
    pose = torch.as_tensor(sample["ego_pose"], dtype=torch.float32, device=device).reshape(1, 6)
    stamp = torch.tensor([float(sample["timestamp"])], dtype=torch.float32, device=device)
    motion = torch.as_tensor(sample["motion"], dtype=torch.float32, device=device).reshape(1, 2)
    rows = [move_sender(r, device) for r in sample["senders"]]
    out = {}
    for k in (1, 4):
        if k > len(rows):
            continue
        use = rows[:k]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                model(pose, stamp, motion, [blobs])
        torch.cuda.synchronize()
        reps = 15
        t_acc = 0.0
        with torch.no_grad():
            for _ in range(reps):
                ts0 = time.perf_counter()
                blobs = [model.transmit(r, sample["ego_pose"], sample["receiver_id"], n_tok, budget=budget)["blob"] for r in use]
                model(pose, stamp, motion, [blobs])
                torch.cuda.synchronize()
                t_acc += (time.perf_counter() - ts0) * 1000.0
        ms = t_acc / reps
        out[f"{k}_sender"] = {
            "total_ms": round(ms, 2),
            "fps": round(1000.0 / max(1e-3, ms), 1),
            "vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--mode", choices=("all", "v21"), default="all")
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    train_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "train")
    val_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "validate")
    test_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "test")
    fog_test = load_cached_split(CACHE_ROOT / "opv2v_lisa_fog" / "test")
    rain_test = load_cached_split(CACHE_ROOT / "opv2v_lisa_rain" / "test")
    class_weights = compute_class_weights(train_frames, device)
    print(
        f"Loaded caches in {time.perf_counter()-t_load:.1f}s: "
        f"train={len(train_frames)} val={len(val_frames)} test={len(test_frames)} "
        f"env50_present={train_frames[0]['senders'][0].get('envelope50') is not None}",
        flush=True,
    )

    b3_kw = dict(compact_wire=True, use_norm_pe=True, use_time_decay=True)
    b4_kw = dict(
        use_envelope=True,
        use_grouped_books=True,
        use_quota=True,
        default_stage=2,
        envelope_cells=50,
        use_cls_embed=True,
        compact_wire=True,
        true_residual=True,
        use_zmax_env=True,
        use_norm_pe=True,
        use_time_decay=True,
        road_floor=0.10,
    )
    b4_noenv_kw = dict(b4_kw)
    b4_noenv_kw["use_envelope"] = False
    b4_noenv_kw["use_zmax_env"] = False

    b4_v21_kw = dict(
        use_envelope=True,
        use_grouped_books=True,
        use_quota=True,
        default_stage=2,
        envelope_cells=50,
        use_cls_embed=True,
        compact_wire=True,
        true_residual=False,
        use_zmax_env=True,
        use_norm_pe=True,
        use_time_decay=True,
        road_floor=0.10,
        decouple_env=True,
    )

    if args.mode == "v21":
        variants = [
            ("B4_v21_32k", "B4", b4_v21_kw, TARGET_BYTES),
            ("B4_v21_16k", "B4", b4_v21_kw, FAIR_BUDGET_BYTES),
        ]
        stress_tags = ("B4_v21_32k", "B4_v21_16k")
        out_name = "v21_verify_summary.json"
    else:
        variants = [
            ("B3_v2_32k", "B3", b3_kw, TARGET_BYTES),
            ("B4_v2_32k", "B4", b4_kw, TARGET_BYTES),
            ("B4_v2_32k_no_env", "B4", b4_noenv_kw, TARGET_BYTES),
            ("B3_v2_16k", "B3", b3_kw, FAIR_BUDGET_BYTES),
            ("B4_v2_16k", "B4", b4_kw, FAIR_BUDGET_BYTES),
        ]
        stress_tags = ("B3_v2_32k", "B4_v2_32k", "B4_v2_32k_no_env")
        out_name = "v2_gate_summary.json"

    results = {}
    trained_models = {}
    for tag, method, mkw, tbytes in variants:
        model, rec = train_v2_variant(
            tag,
            method,
            mkw,
            tbytes,
            seed=0,
            train_frames=train_frames,
            val_frames=val_frames,
            test_frames=test_frames,
            class_weights=class_weights,
            cfg=cfg,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            batch_size=args.batch_size,
        )
        trained_models[tag] = (model, tbytes)
        results[tag] = rec

    test_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in test_frames}
    fog_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in fog_test}
    rain_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in rain_test}
    multi4 = [fr for fr in test_frames if len(fr["senders"]) >= 4]
    bench_sample = multi4[0] if multi4 else test_frames[0]

    stress = {}
    for tag in stress_tags:
        m, tbytes = trained_models[tag]
        st = {}
        for d in (1, 2, 5):
            st[f"delay_{d*100}ms"] = eval_serialized(m, test_frames, test_lookup, tbytes, device, np.random.default_rng(7), delay=d)
        st["drop_50pct"] = eval_serialized(m, test_frames, test_lookup, tbytes, device, np.random.default_rng(7), drop_p=0.5)
        st["pose_0.5m_2deg"] = eval_serialized(m, test_frames, test_lookup, tbytes, device, np.random.default_rng(7), xy_std=0.5, yaw_std=2.0)
        st["lisa_fog"] = eval_serialized(m, fog_test, fog_lookup, tbytes, device, np.random.default_rng(7))
        st["lisa_rain"] = eval_serialized(m, rain_test, rain_lookup, tbytes, device, np.random.default_rng(7))
        st["speed"] = bench_speed(m, bench_sample, tbytes, device)
        stress[tag] = st
        print(
            f"\nSTRESS [{tag}]: "
            f"delay500ms mIoU={st['delay_500ms']['mIoU']:.4f} veh={st['delay_500ms']['vehicle_IoU']:.4f} | "
            f"drop50% mIoU={st['drop_50pct']['mIoU']:.4f} | "
            f"pose0.5m mIoU={st['pose_0.5m_2deg']['mIoU']:.4f} | "
            f"fog={st['lisa_fog']['mIoU']:.4f} rain={st['lisa_rain']['mIoU']:.4f} | "
            f"speed={st['speed']}",
            flush=True,
        )

    summary = {"models": results, "stress": stress}
    (OUT / out_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved complete summary to {OUT / out_name}", flush=True)


if __name__ == "__main__":
    main()
