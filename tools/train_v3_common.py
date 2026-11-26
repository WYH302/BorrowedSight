"""Shared resume-safe training engine for BorrowedSight v3.0 (IEEE Transactions Flagship).

Upgrades over v2.1:
- Frozen SenderEncoder v3 (`sender_opv2v_v3.pt`: 7-ch D4-equivariant scatter-mean/absmax/log-density + GroupNorm + 2D PE).
- Full 6-bit codebook capacity (`K=64`) at 0 extra wire bytes.
- Linear LR warmup (`warmup_epochs=2`) + Cosine Annealing + Model EMA (`ema_decay=0.995`, val-selected).
- Configurable `ufe_w`, `lovasz_w`, `use_sem_film`, `use_code_fill`, `use_w2c_gate`, `fusion_mode`.
- Logs full per-epoch convergence `history` into every checkpoint & JSON record.
- Strictly isolated checkpoints under `checkpoints/real_opv2v_v3/` (never modifies v1/v2.1 artifacts).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from borrowedsight.labels import IGNORE_INDEX
from borrowedsight.losses import asymmetric_ufe_loss, bev_d4_augment, lovasz_softmax_masked
from borrowedsight.model import BorrowedSight, _force_unknown_where, score_bce
from borrowedsight.packer import packet_layout
from borrowedsight.paths import CHECKPOINTS
from tools.run_v2_gate import _drop_placed_tokens, _pose_jitter, eval_serialized
from tools.train_receivers_real import move_sender, set_seed


@torch.no_grad()
def update_ema(ema_mod: torch.nn.Module, src_mod: torch.nn.Module, decay: float) -> None:
    for p_ema, p_src in zip(ema_mod.parameters(), src_mod.parameters()):
        if p_src.requires_grad:
            p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)
    for b_ema, b_src in zip(ema_mod.buffers(), src_mod.buffers()):
        b_ema.data.copy_(b_src.data)


def build_v3_model(
    method: str,
    model_kwargs: dict,
    cfg: dict,
    sender_ckpt_path: Path,
    device: torch.device,
) -> tuple[BorrowedSight, dict]:
    ckpt_s = torch.load(sender_ckpt_path, map_location=device, weights_only=False)
    enc_kw = ckpt_s.get("encoder_kwargs", {})
    mkw = dict(model_kwargs)
    cb_size = int(mkw.pop("codebook_size", 64))
    mkw.setdefault("sender_norm_pe", bool(enc_kw.get("use_norm_pe", False)))
    mkw.setdefault("sender_max_pool", bool(enc_kw.get("use_max_pool", False)))
    model = BorrowedSight(
        method,
        feat_dim=int(enc_kw.get("feat_dim", cfg["feat_dim"])),
        codebook_size=cb_size,
        **mkw,
    ).to(device)
    model.encoder.load_state_dict(ckpt_s["encoder"])
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model, enc_kw


def train_or_load_v3(
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
    ckpt_dir: Path,
    sender_ckpt_name: str = "sender_opv2v_v3.pt",
    epochs: int = 30,
    lr: float = 8.0e-4,
    warmup_epochs: int = 2,
    ema_decay: float = 0.995,
    device: torch.device | None = None,
    batch_size: int = 4,
    train_budgets: tuple[int, ...] | None = None,
    lovasz_w: float = 0.5,
    ufe_w: float = 0.25,
) -> dict:
    if device is None:
        device = torch.device("cuda")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}_s{seed}.pt"
    sender_ckpt_path = CHECKPOINTS / sender_ckpt_name

    # Exact config fingerprint for safe resume/reuse
    train_cfg = {
        "epochs": int(epochs),
        "lr": float(lr),
        "warmup_epochs": int(warmup_epochs),
        "ema_decay": float(ema_decay),
        "lovasz_w": float(lovasz_w),
        "ufe_w": float(ufe_w),
        "sender_ckpt": str(sender_ckpt_name),
        "train_budgets": list(train_budgets) if train_budgets else [int(target_bytes)],
    }

    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (
            blob.get("status") in {"converged", "no_communication"}
            and blob.get("method") == method
            and int(blob.get("target_bytes", -1)) == int(target_bytes)
            and blob.get("model_kwargs") == model_kwargs
            and blob.get("train_cfg") == train_cfg
            and blob.get("val_metrics") is not None
            and (blob.get("test_metrics") is not None or not test_frames)
        ):
            vm, tm = blob["val_metrics"], blob.get("test_metrics")
            print(
                f"[SKIP/REUSE] {tag}_s{seed} | VAL mIoU={vm['mIoU']:.4f} "
                + (
                    f"| TEST mIoU={tm['mIoU']:.4f} veh={tm['vehicle_IoU']:.4f} "
                    f"road={tm['road_IoU']:.4f} unkF1={tm['unknown_F1']:.4f} "
                    f"UFE={tm['unobserved_as_free_err']:.4f}"
                    if tm is not None
                    else ""
                ),
                flush=True,
            )
            return {
                "tag": tag,
                "method": method,
                "seed": int(seed),
                "params": int(blob.get("params", 0)),
                "target_bytes": int(target_bytes),
                "token_bytes": int(blob.get("token_bytes", 0)),
                "n_tokens": int(blob.get("n_tokens", 0)),
                "model_kwargs": model_kwargs,
                "train_cfg": train_cfg,
                "best_src": blob.get("best_src", "raw"),
                "seconds": float(blob.get("seconds", 0.0)),
                "peak_vram_gb": float(blob.get("peak_vram_gb", 0.0)),
                "status": blob["status"],
                "val": vm,
                "test": tm,
                "history": blob.get("history", []),
            }

    set_seed(seed)
    model, enc_kw = build_v3_model(method, model_kwargs, cfg, sender_ckpt_path, device)
    ema_model = copy.deepcopy(model).eval() if (ema_decay > 0.0 and method != "B0") else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

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
        f"\n=== V3 TRAIN {tag}_s{seed} (method={method}, K={model.codebook_size}, params={total_params:,}, "
        f"bytes={budget['serialized']}, tok_w={budget['token_bytes']}B, n_tok={budget['n_tokens']}, "
        f"env={model.use_envelope}({model.envelope_cells}), tdvf={model.decouple_env}, film={model.use_sem_film}, "
        f"fuse={model.fusion_mode}, ufe_w={ufe_w}, warmup={warmup_epochs}, ema={ema_decay}, ep={epochs}) ===",
        flush=True,
    )

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    status = "no_communication" if method == "B0" else "running"
    history = []
    best_src = "raw"

    if method != "B0":
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(seed + 100)
        vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
        best_score = -1.0
        best_state = None
        nan_skips = 0

        steps_per_ep = max(1, (len(train_frames) + batch_size - 1) // batch_size)
        total_steps = epochs * steps_per_ep
        warmup_steps = warmup_epochs * steps_per_ep
        global_step = 0
        eta_min = lr * 0.05

        for ep in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(train_frames))
            ep_losses = []
            ep_t0 = time.perf_counter()
            for start in range(0, len(order), batch_size):
                global_step += 1
                if warmup_steps > 0 and global_step <= warmup_steps:
                    cur_lr = lr * (global_step / float(warmup_steps))
                else:
                    prog = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    cur_lr = eta_min + 0.5 * (lr - eta_min) * (1.0 + np.cos(np.pi * min(1.0, prog)))
                for pg in opt.param_groups:
                    pg["lr"] = float(cur_lr)

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
                    cls_prior = model.build_cls_prior(placed_b, pose)
                    if model.decouple_env and extra.shape[1] >= 8:
                        for bi in range(extra.shape[0]):
                            r_env = float(rng.random())
                            if r_env < 0.16:
                                extra[bi, 7] = 0.0
                            elif r_env < 0.24:
                                extra[bi, 0:2] = 0.0
                                extra[bi, 2] = 1.0
                                extra[bi, 7] = 0.0
                    if cls_prior is not None:
                        feat, extra, sem_t, tri_t, cls_prior = bev_d4_augment(feat, extra, sem_t, tri_t, rng, cls_prior=cls_prior)
                    else:
                        feat, extra, sem_t, tri_t = bev_d4_augment(feat, extra, sem_t, tri_t, rng)
                    pred = _force_unknown_where(model.decoder(feat, extra, cls_prior=cls_prior), torch.logical_not(nonempty))

                vq = torch.stack(vq_terms).mean() if vq_terms else pose.new_zeros(())
                sc_loss = torch.stack(score_terms).mean() if score_terms else pose.new_zeros(())
                loss_sem_ce = F.cross_entropy(pred["sem"].float(), sem_t, weight=class_weights, ignore_index=IGNORE_INDEX)
                loss_sem_lov = lovasz_softmax_masked(pred["sem"].float(), sem_t, ignore_index=IGNORE_INDEX)
                loss_tri = F.cross_entropy(pred["tri"].float(), tri_t)
                loss_ufe = asymmetric_ufe_loss(pred["tri"].float(), tri_t)
                loss = loss_sem_ce + lovasz_w * loss_sem_lov + 0.5 * loss_tri + ufe_w * loss_ufe + vq + 0.05 * sc_loss
                if not torch.isfinite(loss):
                    nan_skips += 1
                    opt.zero_grad(set_to_none=True)
                    if nan_skips > 20:
                        status = "nan"
                        break
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
                if not torch.isfinite(grad_norm):
                    nan_skips += 1
                    opt.zero_grad(set_to_none=True)
                    if nan_skips > 20:
                        status = "nan"
                        break
                    continue
                opt.step()
                if ema_model is not None:
                    update_ema(ema_model, model, float(ema_decay))
                ep_losses.append(float(loss.detach().cpu()))
                if torch.cuda.max_memory_allocated() >= vram_cap:
                    status = "vram_abort"
                    break
            if status in {"nan", "vram_abort"}:
                break

            ep_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
            h_row = {
                "epoch": ep,
                "train_loss": ep_mean,
                "lr": float(opt.param_groups[0]["lr"]),
                "seconds": round(time.perf_counter() - ep_t0, 2),
            }
            if ep == 1 or ep % 5 == 0 or ep >= epochs - 3:
                val_raw = eval_serialized(model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
                h_row["val_raw_mIoU"] = float(val_raw["mIoU"] or 0.0)
                h_row["val_raw_veh"] = float(val_raw["vehicle_IoU"] or 0.0)
                h_row["val_raw_ufe"] = float(val_raw["unobserved_as_free_err"] or 0.0)
                if h_row["val_raw_mIoU"] > best_score:
                    best_score = h_row["val_raw_mIoU"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_src = f"raw_ep{ep}"

                val_ema = None
                if ema_model is not None and ep >= 5:
                    val_ema = eval_serialized(ema_model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
                    h_row["val_ema_mIoU"] = float(val_ema["mIoU"] or 0.0)
                    h_row["val_ema_veh"] = float(val_ema["vehicle_IoU"] or 0.0)
                    h_row["val_ema_ufe"] = float(val_ema["unobserved_as_free_err"] or 0.0)
                    if h_row["val_ema_mIoU"] > best_score:
                        best_score = h_row["val_ema_mIoU"]
                        best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
                        best_src = f"ema_ep{ep}"
                msg = (
                    f"  [{tag}_s{seed}] ep {ep:02d}/{epochs} loss={ep_mean:.4f} "
                    f"raw={val_raw['mIoU']:.4f}(veh={val_raw['vehicle_IoU']:.4f},UFE={val_raw['unobserved_as_free_err']:.4f}) "
                )
                if val_ema is not None:
                    msg += f"ema={val_ema['mIoU']:.4f}(UFE={val_ema['unobserved_as_free_err']:.4f}) "
                msg += f"best={best_score:.4f}({best_src}) usage={val_raw['codebook_usage']} dt={h_row['seconds']:.1f}s"
                print(msg, flush=True)
            history.append(h_row)

        if status == "running":
            status = "converged"
        if best_state is not None:
            model.load_state_dict(best_state)

    val_metrics = eval_serialized(model, val_frames, val_lookup, target_bytes, device, np.random.default_rng(0))
    test_metrics = (
        eval_serialized(model, test_frames, test_lookup, target_bytes, device, np.random.default_rng(0))
        if test_frames
        else None
    )
    secs = round(time.perf_counter() - t0, 1)
    peak_gb = round(float(torch.cuda.max_memory_allocated() / 1024**3), 3)

    torch.save(
        {
            "model": model.state_dict(),
            "tag": tag,
            "method": method,
            "model_kwargs": model_kwargs,
            "encoder_kwargs": enc_kw,
            "train_cfg": train_cfg,
            "params": int(total_params),
            "target_bytes": int(target_bytes),
            "token_bytes": int(budget["token_bytes"]),
            "n_tokens": int(budget["n_tokens"]),
            "seed": int(seed),
            "status": status,
            "best_src": best_src,
            "seconds": secs,
            "peak_vram_gb": peak_gb,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": history,
        },
        ckpt_path,
    )
    print(
        f"DONE {tag}_s{seed} ({secs/60:.1f}m, {best_src}) | VAL mIoU={val_metrics['mIoU']:.4f} "
        + (
            f"| TEST mIoU={test_metrics['mIoU']:.4f} veh={test_metrics['vehicle_IoU']:.4f} "
            f"road={test_metrics['road_IoU']:.4f} unkF1={test_metrics['unknown_F1']:.4f} "
            f"UFE={test_metrics['unobserved_as_free_err']:.4f}"
            if test_metrics is not None
            else ""
        ),
        flush=True,
    )
    return {
        "tag": tag,
        "method": method,
        "seed": int(seed),
        "params": int(total_params),
        "target_bytes": int(target_bytes),
        "token_bytes": int(budget["token_bytes"]),
        "n_tokens": int(budget["n_tokens"]),
        "model_kwargs": model_kwargs,
        "train_cfg": train_cfg,
        "best_src": best_src,
        "seconds": secs,
        "peak_vram_gb": peak_gb,
        "status": status,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
