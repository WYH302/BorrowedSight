"""Phase 3 v3.0 (Real OPV2V): Train upgraded pure-PyTorch SenderEncoder v3.

Upgrades over v1 SenderEncoder (which was locked at 0.7150 val mIoU):
1. Scatter-mean + scatter-max + log1p(density) 7-channel voxel pooling (`use_max_pool=True`).
2. GroupNorm(8, C) inside every ResBlock + fixed 2D sinusoidal PE inside CoarseSDPA (`use_norm_pe=True`).
3. Exact 8-way D4 dihedral augmentation applied directly on raw 3D point clouds & semantic grids.
4. Lovasz-Softmax (0.5) + class-weighted CE + 2-epoch linear warmup + Cosine LR + Model EMA (0.995).

Saves isolated checkpoint to `D:/bs_data/BorrowedSight/checkpoints/sender_opv2v_v3.pt`
and never overwrites v1's `sender_opv2v.pt`.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.labels import IGNORE_INDEX, N_CLASS
from borrowedsight.losses import lovasz_softmax_masked
from borrowedsight.metrics import finalize_counts, new_counts, semantic_target, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.opv2v_data import RealOPV2VDataset
from borrowedsight.paths import CHECKPOINTS, CONFIGS, OUTPUTS
from borrowedsight.weather import load_weather, train_degrade
from tools.ray_visibility import ray_visibility
from tools.train_sender_real import build_sweep_index, collate_sweeps, evaluate_sender, set_seed, weighted_ce

OUT = OUTPUTS / "real_opv2v" / "phase3_sender_v3"


def apply_d4_pooled_and_target(pooled: torch.Tensor, tgt: torch.Tensor, rng: np.random.Generator):
    """Algebraically exact 8-way D4 dihedral transform on pre-pooled voxel tensor and semantic grid.

    pooled: [B, 7, GZ, GX, GY] with channels:
        0: mean_dx, 1: mean_dy, 2: mean_dz, 3: max_|dx|, 4: max_|dy|, 5: max_dz, 6: log1p(cnt)/4
    tgt:    [B, GX, GY, GZ]
    Applying D4 after voxelization eliminates any off-by-one boundary clipping from CoHFF's 40/99 voxel spacing
    while maintaining 100% exact correspondence between offset channels and spatial indices.
    """
    k = int(rng.integers(0, 4))
    flip_x = bool(rng.integers(0, 2))
    p = pooled
    g = tgt
    if k > 0:
        p = torch.rot90(p, k=k, dims=(-2, -1)).clone()
        g = torch.rot90(g, k=k, dims=(1, 2))
        dx, dy = p[:, 0].clone(), p[:, 1].clone()
        ax, ay = p[:, 3].clone(), p[:, 4].clone()
        if k == 1:
            # rot90(k=1) maps old (i, j) -> new (GY-1-j, i), so (x', y') = (-y, +x)
            p[:, 0] = -dy
            p[:, 1] = dx
            p[:, 3] = ay
            p[:, 4] = ax
        elif k == 2:
            p[:, 0] = -dx
            p[:, 1] = -dy
        elif k == 3:
            p[:, 0] = dy
            p[:, 1] = -dx
            p[:, 3] = ay
            p[:, 4] = ax
    if flip_x:
        p = torch.flip(p, dims=(-2,)).clone()
        g = torch.flip(g, dims=(1,))
        p[:, 0] = -p[:, 0]
    return p.contiguous(), g.contiguous()


@torch.no_grad()
def update_ema(ema_mod: torch.nn.Module, src_mod: torch.nn.Module, decay: float) -> None:
    for p_ema, p_src in zip(ema_mod.parameters(), src_mod.parameters()):
        p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)
    for b_ema, b_src in zip(ema_mod.buffers(), src_mod.buffers()):
        b_ema.data.copy_(b_src.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--lovasz-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sweeps-per-epoch", type=int, default=600)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    weather_cfg = load_weather()
    set_seed(args.seed)
    device = torch.device("cuda")

    train_ds = RealOPV2VDataset("train", stride=1)
    val_ds = RealOPV2VDataset("validate", stride=2)
    train_sweeps = build_sweep_index(train_ds)

    model = BorrowedSight(
        "B1",
        feat_dim=int(cfg["feat_dim"]),
        codebook_size=64,
        sender_norm_pe=True,
        sender_max_pool=True,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(
        f"Real OPV2V Sender v3 Training: enc_params={enc_params:,}, "
        f"train_scenes={len(train_ds.scenes)} ({len(train_sweeps)} sweeps), "
        f"epochs={args.epochs}, lr={args.lr}, warmup={args.warmup_epochs}, ema={args.ema_decay}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
    torch.cuda.reset_peak_memory_stats()

    history = []
    best_miou = -1.0
    best_state = None
    best_src = "raw"
    t_start = time.perf_counter()
    status = "running"

    total_steps = args.epochs * ((min(len(train_sweeps), args.max_sweeps_per_epoch) + args.batch_size - 1) // args.batch_size)
    warmup_steps = args.warmup_epochs * ((min(len(train_sweeps), args.max_sweeps_per_epoch) + args.batch_size - 1) // args.batch_size)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_sweeps))
        if args.max_sweeps_per_epoch > 0:
            order = order[: args.max_sweeps_per_epoch]
        ep_losses = []
        ep_t0 = time.perf_counter()
        for i in range(0, len(order), args.batch_size):
            global_step += 1
            if global_step <= warmup_steps:
                cur_lr = args.lr * (global_step / max(1, warmup_steps))
            else:
                prog = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cur_lr = 2e-5 + 0.5 * (args.lr - 2e-5) * (1.0 + np.cos(np.pi * min(1.0, prog)))
            for pg in opt.param_groups:
                pg["lr"] = float(cur_lr)

            batch_ids = order[i : i + args.batch_size]
            loaded = []
            for idx in batch_ids:
                s_idx, fi, ki = train_sweeps[int(idx)]
                pts, sem_vox = train_ds.get_sender_sweep(s_idx, fi, ki)
                if len(pts) > 0:
                    pts = train_degrade(pts, rng, weather_cfg).astype(np.float32)
                loaded.append((pts, sem_vox))
            pts_t, mask_t, tgt_t = collate_sweeps(loaded, device)
            with torch.no_grad():
                pooled = model.encoder._pool(pts_t, mask_t)
                pooled, tgt_t = apply_d4_pooled_and_target(pooled, tgt_t, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.encoder.forward_from_pooled(pooled)
            loss_ce = weighted_ce(out["sem"], tgt_t)
            loss_lov = lovasz_softmax_masked(out["sem"].float(), tgt_t, ignore_index=IGNORE_INDEX)
            loss = loss_ce + float(args.lovasz_w) * loss_lov
            if not torch.isfinite(loss):
                status = "nan"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
            opt.step()
            update_ema(ema_model.encoder, model.encoder, float(args.ema_decay))
            ep_losses.append(float(loss.detach().cpu()))
            if torch.cuda.max_memory_allocated() >= vram_cap:
                status = "vram_abort"
                break
        if status in {"nan", "vram_abort"}:
            break

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        do_eval = (epoch == 1) or (epoch % 5 == 0) or (epoch >= args.epochs - 3)
        val_raw = evaluate_sender(model, val_ds, device, max_sweeps=80) if do_eval else None
        val_ema = evaluate_sender(ema_model, val_ds, device, max_sweeps=80) if (do_eval and epoch >= 5) else None
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - ep_t0,
            "peak_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        }
        if val_raw is not None:
            row["val_raw_mIoU"] = val_raw["mIoU"]
            row["val_raw_veh"] = val_raw["vehicle_IoU"]
            row["val_raw_road"] = val_raw["road_IoU"]
            if (val_raw["mIoU"] or 0.0) > best_miou:
                best_miou = float(val_raw["mIoU"] or 0.0)
                best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
                best_src = f"raw_ep{epoch}"
        if val_ema is not None:
            row["val_ema_mIoU"] = val_ema["mIoU"]
            row["val_ema_veh"] = val_ema["vehicle_IoU"]
            if (val_ema["mIoU"] or 0.0) > best_miou:
                best_miou = float(val_ema["mIoU"] or 0.0)
                best_state = {k: v.detach().cpu().clone() for k, v in ema_model.encoder.state_dict().items()}
                best_src = f"ema_ep{epoch}"
        history.append(row)
        msg = f"Epoch {epoch:02d}/{args.epochs} loss={mean_loss:.4f} lr={row['lr']:.2e} "
        if val_raw is not None:
            msg += f"raw={val_raw['mIoU']:.4f} (veh={val_raw['vehicle_IoU']:.4f}) "
        if val_ema is not None:
            msg += f"ema={val_ema['mIoU']:.4f} (veh={val_ema['vehicle_IoU']:.4f}) "
        msg += f"best={best_miou:.4f}({best_src}) t={row['seconds']:.1f}s peak={row['peak_gb']:.2f}GB"
        print(msg, flush=True)

    if status == "running":
        status = "converged_v3"
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.encoder.load_state_dict(best_state)
    final_val = evaluate_sender(model, val_ds, device, max_sweeps=200)
    ckpt_path = CHECKPOINTS / "sender_opv2v_v3.pt"
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "encoder_kwargs": {"feat_dim": int(cfg["feat_dim"]), "use_norm_pe": True, "use_max_pool": True},
            "enc_params": enc_params,
            "status": status,
            "best_src": best_src,
            "epochs": args.epochs,
            "seed": args.seed,
            "metrics": final_val,
            "history": history,
        },
        ckpt_path,
    )
    summary = {
        "status": status,
        "best_src": best_src,
        "enc_params": enc_params,
        "epochs": args.epochs,
        "seconds": time.perf_counter() - t_start,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        "final_val_metrics": final_val,
        "history": history,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\nDONE SENDER V3 -> {ckpt_path} | final_val_mIoU={final_val['mIoU']:.4f} "
        f"veh={final_val['vehicle_IoU']:.4f} road={final_val['road_IoU']:.4f} "
        f"(vs v1 0.7150) in {summary['seconds']/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
