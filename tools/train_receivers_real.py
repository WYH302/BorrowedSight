"""Phase 4 (Real OPV2V): Train B0-B4, all 4 core ablations, and 3 seeds
at exact serialized byte budgets (32,768 B and 16,384 B), enforcing every
CLAUDE.md Gate 3 and stop rule.
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

from borrowedsight.data import loss_targets
from borrowedsight.labels import GROUND, IGNORE_INDEX, N_CLASS, ROAD, THINGS, VERTICAL
from borrowedsight.metrics import finalize_counts, new_counts, semantic_target, update_counts
from borrowedsight.model import BorrowedSight, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, TARGET_BYTES, packet_layout
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from borrowedsight.registry import log_run

CACHE_ROOT = DATA_ROOT / "cache" / "opv2v_clear"
CKPT_DIR = CHECKPOINTS / "real_opv2v"
OUT = OUTPUTS / "real_opv2v" / "phase4_gate3"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_split(cache_dir: Path) -> list[dict]:
    """Load sharded sparse cache frames + sender rows into RAM (each array decompressed ONCE per shard)."""
    frames = []
    for path in sorted(cache_dir.glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as blob:
            frame_idx = blob["frame_idx"]
            n_f = len(frame_idx)
            f_off = blob["frame_sender_offsets"]
            t_off = blob["token_offsets"]
            scene_name = blob["scene_name"]
            stem = blob["stem"]
            timestamp = blob["timestamp"]
            ego_pose = blob["ego_pose"]
            motion = blob["motion"]
            receiver_id = blob["receiver_id"]
            n_vehicles = blob["n_vehicles"]
            gt_semantic = blob["gt_semantic"]
            gt_tristate = blob["gt_tristate"]
            sender_id = blob["sender_id"]
            sender_pose = blob["sender_pose"]
            sender_timestamp = blob["sender_timestamp"]
            sender_speed = blob["sender_speed"]
            sender_yaw_rate = blob["sender_yaw_rate"]
            coords_all = blob["coords"]
            feats_all = blob["feats"]
            geo_all = blob["geo"]
            pred_all = blob["pred_class"]
            envelope_all = blob["envelope"]
            envelope25_all = blob["envelope25"] if "envelope25" in blob.files else None
            for i in range(n_f):
                s0, s1 = int(f_off[i]), int(f_off[i + 1])
                senders = []
                for si in range(s0, s1):
                    k0, k1 = int(t_off[si]), int(t_off[si + 1])
                    senders.append(
                        {
                            "sender_id": int(sender_id[si]),
                            "pose": np.asarray(sender_pose[si], dtype=np.float32),
                            "timestamp": float(sender_timestamp[si]),
                            "speed": float(sender_speed[si]),
                            "yaw_rate": float(sender_yaw_rate[si]),
                            "coords": torch.from_numpy(coords_all[k0:k1].astype(np.int64)),
                            "feats": torch.from_numpy(feats_all[k0:k1].astype(np.float32)),
                            "geo": torch.from_numpy(geo_all[k0:k1].astype(np.float32)),
                            "pred_class": torch.from_numpy(pred_all[k0:k1].astype(np.int64)),
                            "envelope": np.asarray(envelope_all[si], dtype=np.uint8),
                            "envelope25": np.asarray(envelope25_all[si], dtype=np.uint8) if envelope25_all is not None else None,
                        }
                    )
                gt_sem = np.asarray(gt_semantic[i], dtype=np.uint8)
                gt_tri = np.asarray(gt_tristate[i], dtype=np.uint8)
                sem_tgt = semantic_target(gt_sem, gt_tri).astype(np.int16)
                frames.append(
                    {
                        "scene_name": str(scene_name[i]),
                        "frame": int(frame_idx[i]),
                        "stem": str(stem[i]),
                        "timestamp": float(timestamp[i]),
                        "ego_pose": np.asarray(ego_pose[i], dtype=np.float32),
                        "motion": np.asarray(motion[i], dtype=np.float32),
                        "receiver_id": int(receiver_id[i]),
                        "n_vehicles": int(n_vehicles[i]),
                        "gt_semantic": gt_sem,
                        "gt_tristate": gt_tri,
                        "sem_target_t": torch.from_numpy(sem_tgt),
                        "tri_target_t": torch.from_numpy(gt_tri),
                        "senders": senders,
                    }
                )
    return frames


def compute_class_weights(train_frames: list[dict], device: torch.device) -> torch.Tensor:
    counts = np.zeros(N_CLASS, dtype=np.float64)
    for fr in train_frames:
        tgt = semantic_target(fr["gt_semantic"], fr["gt_tristate"])
        valid = tgt != IGNORE_INDEX
        if np.any(valid):
            bc = np.bincount(tgt[valid].reshape(-1), minlength=N_CLASS)
            counts += bc[:N_CLASS]
    counts = np.maximum(counts, 1.0)
    # Mild inverse-sqrt weighting shared identically across all methods B1-B4
    inv = 1.0 / np.sqrt(counts)
    inv = inv / np.mean(inv[counts > 1.0])
    inv = np.clip(inv, 0.3, 6.0)
    return torch.as_tensor(inv, dtype=torch.float32, device=device)


def move_sender(row: dict, device: torch.device) -> dict:
    out = dict(row)
    for key in ("coords", "feats", "geo", "pred_class"):
        out[key] = row[key].to(device, non_blocking=True)
    return out


def _usage(model: BorrowedSight, codes: list[np.ndarray], classes: list[np.ndarray]) -> float | None:
    if model.method not in {"B3", "B4"} or not codes:
        return None
    code = np.concatenate(codes)
    cls = np.concatenate(classes)
    if model.method == "B4" and model.use_grouped_books:
        groups = [THINGS, GROUND, VERTICAL]
        denom = float(model.codebook_size)
    else:
        groups = [None]
        denom = float(model.codebook_size * (3 if (model.method == "B4" and not model.use_grouped_books) else 1))
    used = []
    for group in groups:
        mask = np.ones(len(code), dtype=bool) if group is None else np.isin(cls, group)
        if not np.any(mask):
            continue
        used.append(len(np.unique(code[mask])) / denom)
    return float(np.mean(used)) if used else None


def _road_fraction(classes: list[np.ndarray]) -> float | None:
    if not classes:
        return None
    cls = np.concatenate(classes)
    if len(cls) == 0:
        return None
    return float(np.mean(cls == ROAD))


def evaluate_frames(
    model: BorrowedSight,
    frames: list[dict],
    budget: dict,
    device: torch.device,
    eval_batch: int = 8,
) -> dict:
    """Strict wire-byte evaluation: EVERY prediction goes through pack_bytes -> unpack_bytes."""
    model.eval()
    acc = new_counts()
    sizes = []
    codes = []
    classes = []
    n_tokens = int(budget["n_tokens"])
    bs = max(1, int(eval_batch))
    with torch.no_grad():
        for start in range(0, len(frames), bs):
            chunk = frames[start : start + bs]
            pose_b = torch.as_tensor(np.stack([fr["ego_pose"] for fr in chunk]), dtype=torch.float32, device=device)
            stamp_b = torch.tensor([float(fr["timestamp"]) for fr in chunk], dtype=torch.float32, device=device)
            motion_b = torch.as_tensor(np.stack([fr["motion"] for fr in chunk]), dtype=torch.float32, device=device)
            blobs_b = []
            for fr in chunk:
                blobs = []
                for row in fr["senders"]:
                    sent = model.transmit(
                        move_sender(row, device),
                        fr["ego_pose"],
                        int(fr["receiver_id"]),
                        n_tokens,
                        budget=budget,
                        serialize=True,
                    )
                    if sent["blob"]:
                        blobs.append(sent["blob"])
                        sizes.append(len(sent["blob"]))
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
    metrics = finalize_counts(acc)
    metrics["bytes_per_frame"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["bytes_min"] = float(np.min(sizes)) if sizes else 0.0
    metrics["bytes_max"] = float(np.max(sizes)) if sizes else 0.0
    metrics["n_packets"] = int(len(sizes))
    metrics["codebook_usage"] = _usage(model, codes, classes)
    metrics["road_token_fraction"] = _road_fraction(classes)
    return metrics


def train_variant(
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
    train_budgets: tuple[int, ...] | None = None,
    batch_size: int = 4,
) -> dict:
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
    )
    budget_pool = (
        [
            packet_layout(method, target=int(b), stage=model.default_stage, include_envelope=model.use_envelope)
            for b in train_budgets
        ]
        if (train_budgets and method != "B0")
        else [budget]
    )
    n_tokens = int(budget["n_tokens"])
    print(
        f"\n=== TRAIN {tag} (method={method}, seed={seed}, bytes={budget['serialized']}, "
        f"tokens={n_tokens}, env={model.use_envelope}({model.envelope_cells}x{model.envelope_cells}), "
        f"grouped={model.use_grouped_books}, quota={model.use_quota}, stage={model.default_stage}, "
        f"train_budgets={train_budgets or (target_bytes,)}) ===",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    rng = np.random.default_rng(seed)
    vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    status = "no_communication" if method == "B0" else "running"
    losses = []
    best_score = -1.0
    best_state = None

    if method != "B0":
        bs = max(1, int(batch_size))
        for ep in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(train_frames))
            ep_losses = []
            ep_t0 = time.perf_counter()
            for start in range(0, len(order), bs):
                batch_ids = order[start : start + bs]
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
                    step_budget = budget_pool[int(rng.integers(0, len(budget_pool)))] if len(budget_pool) > 1 else budget
                    step_tokens = int(step_budget["n_tokens"])
                    sent = []
                    for row in fr["senders"]:
                        sent.append(
                            model.transmit(
                                move_sender(row, device),
                                fr["ego_pose"],
                                int(fr["receiver_id"]),
                                step_tokens,
                                budget=step_budget,
                                serialize=False,
                            )
                        )
                    non_empty = [s for s in sent if s["blob"]]
                    if not non_empty:
                        continue
                    pose_i = torch.as_tensor(fr["ego_pose"], dtype=torch.float32, device=device)
                    placed_b.append([model.placed_from_transmit(s, pose_i, float(fr["timestamp"])) for s in non_empty])
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model.render(placed_b, pose, stamp, motion)
                vq = torch.stack(vq_terms).mean() if vq_terms else pose.new_zeros(())
                sc_loss = torch.stack(score_terms).mean() if score_terms else pose.new_zeros(())
                sem_t = torch.stack(sem_ts, dim=0)
                tri_t = torch.stack(tri_ts, dim=0)
                loss_sem = F.cross_entropy(pred["sem"].float(), sem_t, weight=class_weights, ignore_index=IGNORE_INDEX)
                loss_tri = F.cross_entropy(pred["tri"].float(), tri_t)
                loss = loss_sem + 0.5 * loss_tri + vq + 0.05 * sc_loss
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
            losses.append(ep_mean)
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                val_m = evaluate_frames(model, val_frames, budget, device)
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

    val_metrics = evaluate_frames(model, val_frames, budget, device) if status != "nan" else None
    test_metrics = evaluate_frames(model, test_frames, budget, device) if (status != "nan" and test_frames) else None

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{tag}_s{seed}.pt"
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
        ckpt_path,
    )
    record = {
        "tag": tag,
        "method": method,
        "seed": int(seed),
        "target_bytes": int(target_bytes),
        "n_tokens": n_tokens,
        "model_kwargs": model_kwargs,
        "status": status,
        "epochs": epochs,
        "seconds": time.perf_counter() - t0,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        "losses": losses,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if val_metrics is not None:
        log_run(
            exp_id=f"real_{tag}_s{seed}_val",
            model_name=tag,
            seed=seed,
            bytes_per_frame=val_metrics["bytes_per_frame"],
            metrics=val_metrics,
            status=f"real_val:{status}",
        )
    if test_metrics is not None:
        log_run(
            exp_id=f"real_{tag}_s{seed}_test",
            model_name=tag,
            seed=seed,
            bytes_per_frame=test_metrics["bytes_per_frame"],
            metrics=test_metrics,
            status=f"real_test:{status}",
        )
    return record


def evaluate_gate3(records: dict[str, dict], split_key: str = "val_metrics") -> dict:
    def m(k: str, metric_name: str):
        rec = records.get(k, {})
        met = rec.get(split_key) or {}
        return met.get(metric_name)

    sizes = [m(k, "bytes_per_frame") for k in ("B1", "B2", "B3", "B4") if m(k, "bytes_per_frame") is not None]
    byte_error = None
    if sizes and np.mean(sizes) > 0:
        byte_error = float(max(abs(a - b) for a in sizes for b in sizes) / np.mean(sizes))
    b4_iou, b3_iou = m("B4", "mIoU"), m("B3", "mIoU")
    b4_veh, b3_veh = m("B4", "vehicle_IoU"), m("B3", "vehicle_IoU")
    b4_ufe, b3_ufe = m("B4", "unobserved_as_free_err"), m("B3", "unobserved_as_free_err")
    usage = m("B4", "codebook_usage")
    road = m("B4", "road_token_fraction")
    reasons = []
    if usage is None or usage < 0.40:
        reasons.append(f"codebook usage {usage} < 0.40")
    if road is None or road > 0.60:
        reasons.append(f"road token fraction {road} > 0.60")
    if byte_error is None or byte_error > 0.02:
        reasons.append(f"byte error {byte_error} > 0.02")
    win = False
    if not reasons and None not in (b4_iou, b3_iou):
        if b4_iou >= b3_iou + 0.015:
            win = True
        elif None not in (b4_veh, b3_veh, b4_ufe, b3_ufe) and b4_veh >= b3_veh + 0.05 and b4_ufe <= 0.7 * b3_ufe:
            win = True
        elif abs(b4_iou - b3_iou) <= 0.005 and (m("B4", "bytes_per_frame") or 1e9) <= 0.7 * (m("B3", "bytes_per_frame") or 1):
            win = True
        else:
            reasons.append("B4 did not clear a Gate 3 win criterion against B3")
    return {
        "split": split_key,
        "pass": bool(win and not reasons),
        "reasons": reasons,
        "byte_error": byte_error,
        "b4_mIoU": b4_iou,
        "b3_mIoU": b3_iou,
        "b4_vehicle_IoU": b4_veh,
        "b3_vehicle_IoU": b3_veh,
        "b4_ufe": b4_ufe,
        "b3_ufe": b3_ufe,
        "codebook_usage": usage,
        "road_token_fraction": road,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--skip-16k", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")

    print("Loading pre-extracted Real OPV2V sparse caches...", flush=True)
    train_frames = load_cached_split(CACHE_ROOT / "train")
    val_frames = load_cached_split(CACHE_ROOT / "validate")
    test_frames = load_cached_split(CACHE_ROOT / "test")
    print(f"Loaded train={len(train_frames)}, validate={len(val_frames)}, test={len(test_frames)} real frames", flush=True)
    class_weights = compute_class_weights(train_frames, device)
    print("Shared train class weights:", np.round(class_weights.cpu().numpy(), 3).tolist(), flush=True)

    all_records: dict[str, dict] = {}

    # 1. Core Gate 3 @ 32,768 bytes (official CLAUDE.md Phase 4 budget, seed 0)
    gate3_specs = [
        ("B0", "B0", {}, TARGET_BYTES, 0),
        ("B1", "B1", {}, TARGET_BYTES, 0),
        ("B2", "B2", {}, TARGET_BYTES, 0),
        ("B3", "B3", {}, TARGET_BYTES, 0),
        ("B4", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1), TARGET_BYTES, 0),
    ]
    # 2. All Core Ablations + High-ROI Upgrades (cls_embed, 25x25 Zstd envelope, multi-budget, K=16/64 sensitivity) @ 32,768 B (seed 0)
    ablation_specs_32k = [
        ("B4_no_envelope", "B4", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1), TARGET_BYTES, 0, None),
        ("B4_shared_book", "B4", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1), TARGET_BYTES, 0, None),
        ("B4_no_quota", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=False, default_stage=1), TARGET_BYTES, 0, None),
        ("B4_stage2", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=2), TARGET_BYTES, 0, None),
        ("B4_env25", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, envelope_cells=25), TARGET_BYTES, 0, None),
        ("B4_multibw", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1), TARGET_BYTES, 0, (4096, 8192, 16384, 32768)),
        ("B4_env25_multibw", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, envelope_cells=25), TARGET_BYTES, 0, (4096, 8192, 16384, 32768)),
        ("B4_no_cls_embed", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, use_cls_embed=False), TARGET_BYTES, 0, None),
        ("B4_K16", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, codebook_size=16), TARGET_BYTES, 0, None),
        ("B4_K64", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, codebook_size=64), TARGET_BYTES, 0, None),
    ]
    # 3. Tight-bandwidth (16,384 B = 1.31 Mbps @ 10 Hz) comparison + ablations (seed 0)
    tight_specs_16k = [] if args.skip_16k else [
        ("B1_16k", "B1", {}, FAIR_BUDGET_BYTES, 0, None),
        ("B2_16k", "B2", {}, FAIR_BUDGET_BYTES, 0, None),
        ("B3_16k", "B3", {}, FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1), FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k_no_envelope", "B4", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1), FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k_shared_book", "B4", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1), FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k_no_quota", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=False, default_stage=1), FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k_stage2", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=2), FAIR_BUDGET_BYTES, 0, None),
        ("B4_16k_env25", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, envelope_cells=25), FAIR_BUDGET_BYTES, 0, None),
    ]
    # 4. Multi-seed stability (seeds 1 and 2) on B3, B4, B4_env25, and 2 key ablations @ 32KB
    seed_specs = []
    for sd in (1, 2):
        seed_specs.extend([
            ("B3", "B3", {}, TARGET_BYTES, sd, None),
            ("B4", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1), TARGET_BYTES, sd, None),
            ("B4_env25", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1, envelope_cells=25), TARGET_BYTES, sd, None),
            ("B4_no_envelope", "B4", dict(use_envelope=False, use_grouped_books=True, use_quota=True, default_stage=1), TARGET_BYTES, sd, None),
            ("B4_shared_book", "B4", dict(use_envelope=True, use_grouped_books=False, use_quota=True, default_stage=1), TARGET_BYTES, sd, None),
        ])

    gate3_specs_6 = [(t, m, kw, tb, sd, None) for (t, m, kw, tb, sd) in gate3_specs]
    for tag, method, mkw, tbytes, sd, tpool in gate3_specs_6 + ablation_specs_32k + tight_specs_16k + seed_specs:
        ckpt_path = CKPT_DIR / f"{tag}_s{sd}.pt"
        if args.resume and ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if saved.get("val_metrics") is not None and saved.get("test_metrics") is not None:
                bud = packet_layout(
                    method,
                    target=int(tbytes),
                    stage=int(mkw.get("default_stage", 1)),
                    include_envelope=(method == "B4") if mkw.get("use_envelope") is None else bool(mkw["use_envelope"]),
                )
                print(
                    f"=== RESUME SKIP {tag}_s{sd} (val_mIoU={saved['val_metrics']['mIoU']:.4f}, "
                    f"test_mIoU={saved['test_metrics']['mIoU']:.4f}) ===",
                    flush=True,
                )
                all_records[f"{tag}_s{sd}"] = {
                    "tag": tag,
                    "method": method,
                    "seed": int(sd),
                    "target_bytes": int(tbytes),
                    "n_tokens": int(bud["n_tokens"]),
                    "model_kwargs": mkw,
                    "status": saved.get("status", "converged"),
                    "epochs": args.epochs,
                    "seconds": 0.0,
                    "peak_vram_gb": 0.0,
                    "losses": [],
                    "val_metrics": saved["val_metrics"],
                    "test_metrics": saved["test_metrics"],
                }
                continue
        rec = train_variant(
            tag=tag,
            method=method,
            model_kwargs=mkw,
            target_bytes=tbytes,
            seed=sd,
            train_frames=train_frames,
            val_frames=val_frames,
            test_frames=test_frames,
            class_weights=class_weights,
            cfg=cfg,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            train_budgets=tpool,
            batch_size=args.batch_size,
        )
        all_records[f"{tag}_s{sd}"] = rec

    gate_32k_val = evaluate_gate3({k: all_records[f"{k}_s0"] for k in ("B0", "B1", "B2", "B3", "B4")}, "val_metrics")
    gate_32k_test = evaluate_gate3({k: all_records[f"{k}_s0"] for k in ("B0", "B1", "B2", "B3", "B4")}, "test_metrics")
    gate_16k_val = None
    gate_16k_test = None
    if not args.skip_16k:
        rec16 = {k: all_records[f"{k}_16k_s0"] for k in ("B1", "B2", "B3", "B4")}
        rec16["B0"] = all_records["B0_s0"]
        gate_16k_val = evaluate_gate3(rec16, "val_metrics")
        gate_16k_test = evaluate_gate3(rec16, "test_metrics")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_32k_val": gate_32k_val,
        "gate_32k_test": gate_32k_test,
        "gate_16k_val": gate_16k_val,
        "gate_16k_test": gate_16k_test,
        "records": all_records,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Write clean CSV + Markdown summary
    csv_cols = [
        "run_key", "tag", "method", "seed", "split", "target_bytes", "bytes_per_frame",
        "mIoU", "vehicle_IoU", "road_IoU", "unknown_F1", "unobserved_as_free_err",
        "pred_unknown_frac", "mean_uncertainty", "uncertainty_on_unknown", "uncertainty_on_observed",
        "codebook_usage", "road_token_fraction", "peak_vram_gb", "status",
    ]
    rows = []
    for rkey, rec in all_records.items():
        for sp_name, sp_key in (("validate", "val_metrics"), ("test", "test_metrics")):
            met = rec.get(sp_key)
            if not met:
                continue
            rows.append(
                {
                    "run_key": rkey,
                    "tag": rec["tag"],
                    "method": rec["method"],
                    "seed": rec["seed"],
                    "split": sp_name,
                    "target_bytes": rec["target_bytes"],
                    "bytes_per_frame": met.get("bytes_per_frame"),
                    "mIoU": met.get("mIoU"),
                    "vehicle_IoU": met.get("vehicle_IoU"),
                    "road_IoU": met.get("road_IoU"),
                    "unknown_F1": met.get("unknown_F1"),
                    "unobserved_as_free_err": met.get("unobserved_as_free_err"),
                    "pred_unknown_frac": met.get("pred_unknown_frac"),
                    "mean_uncertainty": met.get("mean_uncertainty"),
                    "uncertainty_on_unknown": met.get("uncertainty_on_unknown"),
                    "uncertainty_on_observed": met.get("uncertainty_on_observed"),
                    "codebook_usage": met.get("codebook_usage"),
                    "road_token_fraction": met.get("road_token_fraction"),
                    "peak_vram_gb": rec.get("peak_vram_gb"),
                    "status": rec["status"],
                }
            )
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=csv_cols)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Real OPV2V + Semantic-OPV2V Gate 3 & Ablations Summary",
        "",
        f"- Gate 3 @ 32KB (validate): **Pass={gate_32k_val['pass']}** (byte_err={gate_32k_val['byte_error']}, usage={gate_32k_val['codebook_usage']}, roadTok={gate_32k_val['road_token_fraction']})",
        f"- Gate 3 @ 32KB (test): **Pass={gate_32k_test['pass']}**",
    ]
    if gate_16k_val:
        md.append(f"- Gate 3 @ 16KB (validate): **Pass={gate_16k_val['pass']}** (test Pass={gate_16k_test['pass']})")
    md.extend([
        "",
        "| run | split | bytes | mIoU | veh IoU | road IoU | unk F1 | UFE | usage | roadTok |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        def f(v):
            return "--" if v is None else f"{float(v):.4f}"
        md.append(
            f"| {r['run_key']} | {r['split']} | {r['bytes_per_frame']:.0f} | "
            f"{f(r['mIoU'])} | {f(r['vehicle_IoU'])} | {f(r['road_IoU'])} | "
            f"{f(r['unknown_F1'])} | {f(r['unobserved_as_free_err'])} | "
            f"{f(r['codebook_usage'])} | {f(r['road_token_fraction'])} |"
        )
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT / 'summary.md'} | Gate3 32KB val={gate_32k_val['pass']} test={gate_32k_test['pass']}", flush=True)


if __name__ == "__main__":
    main()
