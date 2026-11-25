"""Step 3 (BorrowedSight v2.1 Full Suite): Resume-safe 3-seed, fair baselines,
orthogonal ablations, and MultiBW training on 25-scene Real OPV2V.

Reuses already-converged checkpoints in `checkpoints/real_opv2v_v2` automatically
and saves incremental results after every single model to `outputs/real_opv2v/v21_full/`.
Zero overwriting of v1 checkpoints or v1 outputs.
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
from borrowedsight.model import BorrowedSight, _force_unknown_where, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, TARGET_BYTES, packet_layout
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.run_v2_gate import _drop_placed_tokens, _pose_jitter, eval_serialized
from tools.train_receivers_real import compute_class_weights, load_cached_split, move_sender, set_seed

CACHE_ROOT = DATA_ROOT / "cache"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v2"
OUT = OUTPUTS / "real_opv2v" / "v21_full"


def train_or_load_v21(
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
    train_budgets: tuple[int, ...] | None = None,
    lovasz_w: float = 0.5,
    ufe_w: float = 0.35,
) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{tag}_s{seed}.pt"

    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (
            blob.get("status") in {"converged", "no_communication"}
            and blob.get("method") == method
            and int(blob.get("target_bytes", -1)) == int(target_bytes)
            and blob.get("val_metrics") is not None
            and blob.get("test_metrics") is not None
        ):
            mkw_saved = blob.get("model_kwargs") or {}
            if mkw_saved == model_kwargs:
                mkw_tmp = dict(model_kwargs)
                cb_sz = int(mkw_tmp.pop("codebook_size", cfg["codebook_size"]))
                m_tmp = BorrowedSight(method, feat_dim=int(cfg["feat_dim"]), codebook_size=cb_sz, **mkw_tmp)
                total_params = sum(p.numel() for p in m_tmp.parameters())
                budget = packet_layout(
                    method,
                    target=int(target_bytes),
                    stage=m_tmp.default_stage,
                    include_envelope=m_tmp.use_envelope,
                    compact=m_tmp.compact_wire,
                )
                vm, tm = blob["val_metrics"], blob["test_metrics"]
                print(
                    f"[SKIP/REUSE] {tag}_s{seed} | VAL mIoU={vm['mIoU']:.4f} | "
                    f"TEST mIoU={tm['mIoU']:.4f} veh={tm['vehicle_IoU']:.4f} "
                    f"road={tm['road_IoU']:.4f} unkF1={tm['unknown_F1']:.4f} "
                    f"UFE={tm['unobserved_as_free_err']:.4f}",
                    flush=True,
                )
                return {
                    "tag": tag,
                    "method": method,
                    "seed": int(seed),
                    "params": total_params,
                    "target_bytes": int(target_bytes),
                    "token_bytes": int(budget["token_bytes"]),
                    "n_tokens": int(budget["n_tokens"]),
                    "model_kwargs": model_kwargs,
                    "train_budgets": list(train_budgets) if train_budgets else [int(target_bytes)],
                    "seconds": float(blob.get("seconds", 0.0)),
                    "peak_vram_gb": float(blob.get("peak_vram_gb", 0.0)),
                    "status": blob["status"],
                    "val": vm,
                    "test": tm,
                }

    set_seed(seed)
    mkw = dict(model_kwargs)
    cb_size = int(mkw.pop("codebook_size", cfg["codebook_size"]))
    model = BorrowedSight(method, feat_dim=int(cfg["feat_dim"]), codebook_size=cb_size, **mkw).to(device)
    ckpt_s = torch.load(CHECKPOINTS / "sender_opv2v.pt", map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt_s["encoder"])
    for p in model.encoder.parameters():
        p.requires_grad = False

    budget = packet_layout(
        method,
        target=int(target_bytes),
        stage=model.default_stage,
        include_envelope=model.use_envelope,
        compact=model.compact_wire,
    )
    budget_pool = (
        [
            packet_layout(
                method,
                target=int(b),
                stage=model.default_stage,
                include_envelope=model.use_envelope,
                compact=model.compact_wire,
            )
            for b in train_budgets
        ]
        if (train_budgets and method != "B0")
        else [budget]
    )
    total_params = sum(p.numel() for p in model.parameters())
    train_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in train_frames}
    val_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in val_frames}
    test_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in test_frames}

    print(
        f"\n=== V2.1 TRAIN {tag}_s{seed} (method={method}, params={total_params:,}, bytes={budget['serialized']}, "
        f"tok_w={budget['token_bytes']}B, n_tok={budget['n_tokens']}, env={model.use_envelope}({model.envelope_cells}), "
        f"stage={model.default_stage}, tdvf={model.decouple_env}, multibw={bool(train_budgets)}) ===",
        flush=True,
    )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    status = "no_communication" if method == "B0" else "running"

    if method != "B0":
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
        rng = np.random.default_rng(seed + 100)
        vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
        best_score = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(train_frames))
            ep_losses = []
            ep_t0 = time.perf_counter()
            for start in range(0, len(order), batch_size):
                batch_ids = order[start : start + batch_size]
                opt.zero_grad(set_to_none=True)
                placed_b, poses_b, stamps_b, motions_b, sem_ts, tri_ts = [], [], [], [], [], []
                vq_terms, score_terms = [], []
                for idx in batch_ids:
                    fr = train_frames[int(idx)]
                    step_budget = budget_pool[int(rng.integers(0, len(budget_pool)))] if len(budget_pool) > 1 else budget
                    step_tokens = int(step_budget["n_tokens"])

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
                                step_tokens,
                                budget=step_budget,
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
                    f"  [{tag}_s{seed}] ep {ep:02d}/{epochs} loss={ep_mean:.4f} "
                    f"val_mIoU={val_m['mIoU']:.4f} veh={val_m['vehicle_IoU']:.4f} "
                    f"road={val_m['road_IoU']:.4f} unkF1={val_m['unknown_F1']:.4f} "
                    f"UFE={val_m['unobserved_as_free_err']:.4f} usage={val_m['codebook_usage']} "
                    f"dt={time.perf_counter()-ep_t0:.1f}s",
                    flush=True,
                )
        if status == "running":
            status = "converged"
        if best_state is not None:
            model.load_state_dict(best_state)

    val_metrics = eval_serialized(model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
    test_metrics = eval_serialized(model, test_frames, test_lookup, target_bytes, device, np.random.default_rng(0))
    secs = round(time.perf_counter() - t0, 1)
    peak_gb = round(float(torch.cuda.max_memory_allocated() / 1024**3), 3)

    torch.save(
        {
            "model": model.state_dict(),
            "tag": tag,
            "method": method,
            "model_kwargs": model_kwargs,
            "target_bytes": int(target_bytes),
            "seed": int(seed),
            "status": status,
            "seconds": secs,
            "peak_vram_gb": peak_gb,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    print(
        f"DONE [{tag}_s{seed}] time={secs}s vram={peak_gb}GB | "
        f"VAL mIoU={val_metrics['mIoU']:.4f} | TEST mIoU={test_metrics['mIoU']:.4f} "
        f"veh={test_metrics['vehicle_IoU']:.4f} road={test_metrics['road_IoU']:.4f} "
        f"unkF1={test_metrics['unknown_F1']:.4f} UFE={test_metrics['unobserved_as_free_err']:.4f} "
        f"bytes={test_metrics['bytes_per_frame']:.0f}",
        flush=True,
    )
    return {
        "tag": tag,
        "method": method,
        "seed": int(seed),
        "params": total_params,
        "target_bytes": int(target_bytes),
        "token_bytes": int(budget["token_bytes"]),
        "n_tokens": int(budget["n_tokens"]),
        "model_kwargs": model_kwargs,
        "train_budgets": list(train_budgets) if train_budgets else [int(target_bytes)],
        "seconds": secs,
        "peak_vram_gb": peak_gb,
        "status": status,
        "val": val_metrics,
        "test": test_metrics,
    }


def flush_outputs(records: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    fields = [
        "tag", "method", "seed", "target_bytes", "token_bytes", "n_tokens", "params",
        "val_mIoU", "test_mIoU", "test_veh", "test_road", "test_unkF1", "test_UFE",
        "test_bytes", "cb_usage", "road_tok", "seconds", "peak_vram_gb", "status",
    ]
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in records:
            t, v = r["test"], r["val"]
            w.writerow(
                {
                    "tag": r["tag"],
                    "method": r["method"],
                    "seed": r["seed"],
                    "target_bytes": r["target_bytes"],
                    "token_bytes": r["token_bytes"],
                    "n_tokens": r["n_tokens"],
                    "params": r["params"],
                    "val_mIoU": round(float(v["mIoU"] or 0.0), 4),
                    "test_mIoU": round(float(t["mIoU"] or 0.0), 4),
                    "test_veh": round(float(t["vehicle_IoU"] or 0.0), 4),
                    "test_road": round(float(t["road_IoU"] or 0.0), 4),
                    "test_unkF1": round(float(t["unknown_F1"] or 0.0), 4),
                    "test_UFE": round(float(t["unobserved_as_free_err"] or 0.0), 4),
                    "test_bytes": round(float(t["bytes_per_frame"] or 0.0), 1),
                    "cb_usage": None if t.get("codebook_usage") is None else round(float(t["codebook_usage"]), 4),
                    "road_tok": None if t.get("road_token_fraction") is None else round(float(t["road_token_fraction"]), 4),
                    "seconds": r["seconds"],
                    "peak_vram_gb": r["peak_vram_gb"],
                    "status": r["status"],
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    train_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "train")
    val_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "validate")
    test_frames = load_cached_split(CACHE_ROOT / "opv2v_clear" / "test")
    class_weights = compute_class_weights(train_frames, device)

    b12_kw = dict(compact_wire=False, use_norm_pe=True, use_time_decay=True)
    b3_kw = dict(compact_wire=True, use_norm_pe=True, use_time_decay=True)
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

    def _ablate(**overrides) -> dict:
        d = dict(b4_v21_kw)
        d.update(overrides)
        return d

    # Complete v2.1 publication matrix (reuses already-converged s0 models automatically!)
    jobs = [
        # 1. Zero-comm lower bound
        ("B0_v2", "B0", dict(use_norm_pe=True), TARGET_BYTES, 0, None),
        # 2. 3-seed Shared VQ (fair 4B/token) & BorrowedSight v2.1 (5B/token + 50x50 TDVF) at 32 KB
        ("B3_v2_32k", "B3", b3_kw, TARGET_BYTES, 0, None),
        ("B3_v2_32k", "B3", b3_kw, TARGET_BYTES, 1, None),
        ("B3_v2_32k", "B3", b3_kw, TARGET_BYTES, 2, None),
        ("B4_v21_32k", "B4", b4_v21_kw, TARGET_BYTES, 0, None),
        ("B4_v21_32k", "B4", b4_v21_kw, TARGET_BYTES, 1, None),
        ("B4_v21_32k", "B4", b4_v21_kw, TARGET_BYTES, 2, None),
        # 3. 3-seed Shared VQ & BorrowedSight v2.1 at 16 KB (Half Bandwidth)
        ("B3_v2_16k", "B3", b3_kw, FAIR_BUDGET_BYTES, 0, None),
        ("B3_v2_16k", "B3", b3_kw, FAIR_BUDGET_BYTES, 1, None),
        ("B3_v2_16k", "B3", b3_kw, FAIR_BUDGET_BYTES, 2, None),
        ("B4_v21_16k", "B4", b4_v21_kw, FAIR_BUDGET_BYTES, 0, None),
        ("B4_v21_16k", "B4", b4_v21_kw, FAIR_BUDGET_BYTES, 1, None),
        ("B4_v21_16k", "B4", b4_v21_kw, FAIR_BUDGET_BYTES, 2, None),
        # 4. Strictly orthogonal ablations at 32 KB (changing ONE factor at a time vs B4_v21_32k)
        ("B4_v21_32k_no_env", "B4", _ablate(use_envelope=False, use_zmax_env=False, decouple_env=False), TARGET_BYTES, 0, None),
        ("B4_v21_32k_no_env", "B4", _ablate(use_envelope=False, use_zmax_env=False, decouple_env=False), TARGET_BYTES, 1, None),
        ("B4_v21_32k_no_env", "B4", _ablate(use_envelope=False, use_zmax_env=False, decouple_env=False), TARGET_BYTES, 2, None),
        ("B4_v21_32k_no_tdvf", "B4", _ablate(decouple_env=False), TARGET_BYTES, 0, None),
        ("B4_v21_32k_no_cls", "B4", _ablate(use_cls_embed=False), TARGET_BYTES, 0, None),
        ("B4_v21_32k_shared96", "B4", _ablate(use_grouped_books=False), TARGET_BYTES, 0, None),
        ("B4_v21_32k_no_quota", "B4", _ablate(use_quota=False), TARGET_BYTES, 0, None),
        ("B4_v21_32k_stage1", "B4", _ablate(default_stage=1), TARGET_BYTES, 0, None),
        # 5. Multi-budget rate-adaptive checkpoint (4-32 KB randomized training)
        ("B4_v21_multibw", "B4", b4_v21_kw, TARGET_BYTES, 0, (4096, 8192, 16384, 32768)),
        # 6. Fair continuous baselines (32 KB & 16 KB)
        ("B1_v2_32k", "B1", b12_kw, TARGET_BYTES, 0, None),
        ("B2_v2_32k", "B2", b12_kw, TARGET_BYTES, 0, None),
        ("B1_v2_16k", "B1", b12_kw, FAIR_BUDGET_BYTES, 0, None),
        ("B2_v2_16k", "B2", b12_kw, FAIR_BUDGET_BYTES, 0, None),
    ]

    records = []
    t_all = time.perf_counter()
    for tag, method, mkw, tbytes, seed, tbudgets in jobs:
        rec = train_or_load_v21(
            tag=tag,
            method=method,
            model_kwargs=mkw,
            target_bytes=tbytes,
            seed=seed,
            train_frames=train_frames,
            val_frames=val_frames,
            test_frames=test_frames,
            class_weights=class_weights,
            cfg=cfg,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            batch_size=args.batch_size,
            train_budgets=tbudgets,
        )
        records.append(rec)
        flush_outputs(records)

    print(f"\nALL {len(records)} V2.1 MODELS COMPLETE in {(time.perf_counter()-t_all)/3600:.2f} h -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
