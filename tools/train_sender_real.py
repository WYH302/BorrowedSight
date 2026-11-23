"""Phase 3 (Real OPV2V): Train pure-PyTorch SenderEncoder on real OPV2V raw lidar
supervised by exact Semantic-OPV2V 4LidarSurround voxels + sender ray tri-state.

Uses only disjoint lightweight train augmentation (train_degrade: 5% drop + 1cm jitter),
never LISA fog/rain coefficients. Unknown & Unlabeled voxels use ignore_index = -100.
"""
from __future__ import annotations

import argparse
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

from borrowedsight.labels import IGNORE_INDEX, N_CLASS
from borrowedsight.metrics import finalize_counts, new_counts, semantic_target, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.opv2v_data import RealOPV2VDataset
from borrowedsight.paths import CHECKPOINTS, CONFIGS, OUTPUTS
from borrowedsight.registry import log_run
from borrowedsight.weather import load_weather, train_degrade
from tools.ray_visibility import ray_visibility

OUT = OUTPUTS / "real_opv2v" / "phase3_sender"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sweep_index(ds: RealOPV2VDataset) -> list[tuple[int, int, int]]:
    items = []
    for s_idx, fi in ds.index:
        n_cav = len(ds.scenes[s_idx]["cav_ids"])
        for ki in range(n_cav):
            items.append((s_idx, fi, ki))
    return items


def collate_sweeps(sweeps: list[tuple[np.ndarray, np.ndarray]], device: torch.device):
    max_pts = max((len(pts) for pts, _ in sweeps), default=1)
    max_pts = max(1, min(max_pts, 120000))
    b = len(sweeps)
    pts_t = torch.zeros(b, max_pts, 3, dtype=torch.float32, device=device)
    mask_t = torch.zeros(b, max_pts, dtype=torch.bool, device=device)
    targets = []
    for i, (pts, sem_vox) in enumerate(sweeps):
        n = min(len(pts), max_pts)
        if n > 0:
            p = torch.as_tensor(pts[:n], dtype=torch.float32, device=device)
            pts_t[i, :n] = p
            mask_t[i, :n] = True
            tri = ray_visibility(p, torch.zeros(3, device=device)).detach().cpu().numpy()
        else:
            tri = np.zeros((100, 100, 8), dtype=np.uint8)
        tgt = semantic_target(sem_vox, tri)
        targets.append(torch.as_tensor(tgt, dtype=torch.long, device=device))
    return pts_t, mask_t, torch.stack(targets, dim=0)


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, N_CLASS).float()
    flat_target = target.reshape(-1)
    valid = flat_target != IGNORE_INDEX
    if int(valid.sum().item()) == 0:
        return logits.sum() * 0.0
    if class_weights is None:
        counts = torch.bincount(flat_target[valid], minlength=N_CLASS).float().clamp(min=1.0)
        class_weights = (counts.mean() / counts.sqrt()).clamp(min=0.2, max=8.0)
    return F.cross_entropy(flat_logits, flat_target, weight=class_weights, ignore_index=IGNORE_INDEX)


def evaluate_sender(model: BorrowedSight, val_ds: RealOPV2VDataset, device: torch.device, max_sweeps: int = 120) -> dict:
    model.eval()
    acc = new_counts()
    all_sweeps = build_sweep_index(val_ds)
    step = max(1, len(all_sweeps) // max_sweeps)
    picked = all_sweeps[::step][:max_sweeps]
    with torch.no_grad():
        for i in range(0, len(picked), 4):
            chunk = picked[i : i + 4]
            loaded = [val_ds.get_sender_sweep(s, f, k) for s, f, k in chunk]
            pts, mask, target = collate_sweeps(loaded, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.encoder(pts, mask)
            pred_sem = out["sem"].argmax(dim=1).detach().cpu().numpy()
            tgt_np = target.detach().cpu().numpy()
            for b in range(pred_sem.shape[0]):
                gt_sem = np.where(tgt_np[b] > 0, tgt_np[b], 0).astype(np.int64)
                gt_tri = np.zeros_like(gt_sem, dtype=np.uint8)
                gt_tri[tgt_np[b] == 0] = 1
                gt_tri[tgt_np[b] > 0] = 2
                pred_tri = np.where(pred_sem[b] == 0, 1, 2).astype(np.uint8)
                update_counts(acc, pred_sem[b], pred_tri, gt_sem, gt_tri)
    return finalize_counts(acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=2)
    ap.add_argument("--max-sweeps-per-epoch", type=int, default=400)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    weather_cfg = load_weather()
    set_seed(args.seed)
    device = torch.device("cuda")

    train_ds = RealOPV2VDataset("train", stride=args.train_stride)
    val_ds = RealOPV2VDataset("validate", stride=args.val_stride)
    train_sweeps = build_sweep_index(train_ds)
    print(
        f"Real OPV2V Sender Training: train_scenes={len(train_ds.scenes)} ({len(train_sweeps)} sweeps), "
        f"val_scenes={len(val_ds.scenes)} ({len(val_ds)} frames), epochs={args.epochs}",
        flush=True,
    )

    model = BorrowedSight("B1", feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"])).to(device)
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=1e-5)

    rng = np.random.default_rng(args.seed)
    vram_cap = float(cfg["vram_abort_gb"]) * (1024**3)
    torch.cuda.reset_peak_memory_stats()

    history = []
    best_miou = -1.0
    best_state = None
    best_metrics = None
    t_start = time.perf_counter()
    status = "running"

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_sweeps))
        if args.max_sweeps_per_epoch > 0:
            order = order[: args.max_sweeps_per_epoch]
        ep_losses = []
        ep_t0 = time.perf_counter()
        for i in range(0, len(order), args.batch_size):
            batch_ids = order[i : i + args.batch_size]
            loaded = []
            for idx in batch_ids:
                s_idx, fi, ki = train_sweeps[int(idx)]
                pts, sem_vox = train_ds.get_sender_sweep(s_idx, fi, ki)
                # Disjoint lightweight training augmentation only (drop + jitter)
                if len(pts) > 0:
                    pts = train_degrade(pts, rng, weather_cfg).astype(np.float32)
                loaded.append((pts, sem_vox))
            pts_t, mask_t, tgt_t = collate_sweeps(loaded, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.encoder(pts_t, mask_t)
            loss = weighted_ce(out["sem"], tgt_t)
            if not torch.isfinite(loss):
                status = "nan"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
            if torch.cuda.max_memory_allocated() >= vram_cap:
                status = "vram_abort"
                break
        if status in {"nan", "vram_abort"}:
            break
        sched.step()
        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        do_eval = (epoch == 1) or (epoch % 5 == 0) or (epoch == args.epochs)
        val_m = evaluate_sender(model, val_ds, device, max_sweeps=80) if do_eval else None
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lr": float(sched.get_last_lr()[0]),
            "seconds": time.perf_counter() - ep_t0,
            "peak_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        }
        if val_m is not None:
            row["val_mIoU"] = val_m["mIoU"]
            row["val_vehicle_IoU"] = val_m["vehicle_IoU"]
            row["val_road_IoU"] = val_m["road_IoU"]
            if (val_m["mIoU"] or 0.0) > best_miou:
                best_miou = float(val_m["mIoU"] or 0.0)
                best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
                best_metrics = val_m
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} loss={mean_loss:.4f} "
            + (
                f"val_mIoU={row['val_mIoU']:.4f} veh={row['val_vehicle_IoU']:.4f} road={row['val_road_IoU']:.4f} "
                if val_m
                else ""
            )
            + f"time={row['seconds']:.1f}s peak={row['peak_gb']:.2f}GB",
            flush=True,
        )

    if status == "running":
        status = "converged_30ep" if history and history[-1]["train_loss"] < history[0]["train_loss"] else "stalled"

    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.encoder.load_state_dict(best_state)
    final_val = evaluate_sender(model, val_ds, device, max_sweeps=200)
    ckpt_path = CHECKPOINTS / "sender_opv2v.pt"
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "status": status,
            "epochs": args.epochs,
            "seed": args.seed,
            "train_scenes": [s["name"] for s in train_ds.scenes],
            "val_scenes": [s["name"] for s in val_ds.scenes],
            "metrics": final_val,
            "cfg": cfg,
        },
        ckpt_path,
    )
    summary = {
        "status": status,
        "epochs": args.epochs,
        "seconds": time.perf_counter() - t_start,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        "train_scenes": [s["name"] for s in train_ds.scenes],
        "val_scenes": [s["name"] for s in val_ds.scenes],
        "final_val_metrics": final_val,
        "history": history,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot training loss & val mIoU curve
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], "-o", markersize=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train weighted CE loss")
    eval_h = [h for h in history if "val_mIoU" in h]
    if eval_h:
        axes[1].plot([h["epoch"] for h in eval_h], [h["val_mIoU"] for h in eval_h], "-o", label="mIoU")
        axes[1].plot([h["epoch"] for h in eval_h], [h["val_vehicle_IoU"] for h in eval_h], "-s", label="Vehicle IoU")
        axes[1].plot([h["epoch"] for h in eval_h], [h["val_road_IoU"] for h in eval_h], "-^", label="Road IoU")
        axes[1].legend(frameon=False)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation IoU")
    fig.suptitle(f"Real OPV2V Sender Training ({len(train_ds.scenes)} train scenes, {args.epochs} epochs)")
    fig.savefig(OUT / "sender_curve.png", dpi=150)
    plt.close(fig)

    log_run(
        exp_id=f"sender_opv2v_ep{args.epochs}_s{args.seed}",
        model_name="sender_opv2v",
        seed=args.seed,
        bytes_per_frame=0.0,
        metrics=final_val,
        status=status,
    )
    print(f"SAVED {ckpt_path} | final_val={json.dumps(final_val)}", flush=True)


if __name__ == "__main__":
    main()
