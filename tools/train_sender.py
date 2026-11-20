"""Train the sender on Mini-200 until the loss drops, or for at most 15 minutes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.data import Mini200
from borrowedsight.grid import voxel_index_torch
from borrowedsight.labels import IGNORE_INDEX, N_CLASS
from borrowedsight.metrics import finalize_counts, new_counts, semantic_target, update_counts
from borrowedsight.model import BorrowedSight
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from borrowedsight.registry import log_run
from tools.ray_visibility import ray_visibility


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sweeps(sample: dict):
    for points, labels in zip(sample["points"], sample["point_sem"]):
        yield np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def collate(bucket, device):
    width = max((len(points) for points, _labels in bucket), default=1)
    width = max(width, 1)
    pts = torch.zeros(len(bucket), width, 3, device=device)
    mask = torch.zeros(len(bucket), width, dtype=torch.bool, device=device)
    for i, (points, _labels) in enumerate(bucket):
        if len(points) == 0:
            continue
        pts[i, : len(points)] = torch.as_tensor(points, device=device)
        mask[i, : len(points)] = True
    return pts, mask


def targets_for(bucket, device):
    out = []
    for points, labels in bucket:
        if len(points) == 0:
            tri = torch.zeros(100, 100, 8, dtype=torch.uint8, device=device)
            sem = np.zeros((100, 100, 8), dtype=np.uint8)
        else:
            tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
            tri = ray_visibility(tensor, torch.zeros(3, device=device))
            sem = np.zeros((100, 100, 8), dtype=np.uint8)
            idx, valid = voxel_index_torch(tensor)
            if int(valid.sum().item()) > 0:
                kept = idx[valid].detach().cpu().numpy()
                lab = labels[valid.detach().cpu().numpy()]
                sem[kept[:, 0], kept[:, 1], kept[:, 2]] = lab.astype(np.uint8)
        target = semantic_target(sem, tri.detach().cpu().numpy())
        out.append(torch.as_tensor(target, dtype=torch.long, device=device))
    return torch.stack(out, dim=0)


def weighted_ce(logits, target):
    flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, N_CLASS).float()
    flat_target = target.reshape(-1)
    valid = flat_target != IGNORE_INDEX
    if int(valid.sum().item()) == 0:
        return logits.sum() * 0.0
    counts = torch.bincount(flat_target[valid], minlength=N_CLASS).float().clamp(min=1)
    weight = counts.mean() / counts.sqrt()
    return F.cross_entropy(flat_logits, flat_target, weight=weight, ignore_index=IGNORE_INDEX)


def evaluate(model, dataset, device, limit: int = 12):
    model.eval()
    acc = new_counts()
    with torch.no_grad():
        seen = 0
        bucket = []
        metas = []
        for index in range(len(dataset)):
            for sweep in sweeps(dataset[index]):
                bucket.append(sweep)
                if len(bucket) < 2:
                    continue
                pts, mask = collate(bucket, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model.encoder(pts, mask)
                target_sem = targets_for(bucket, device)
                sem = pred["sem"].argmax(dim=1).detach().cpu().numpy()
                tri = target_sem.detach().cpu().numpy()
                for b in range(sem.shape[0]):
                    gt_sem = np.zeros(sem[b].shape, dtype=np.int64)
                    positive = tri[b] > 0
                    gt_sem[positive] = tri[b][positive]
                    gt_tri = np.zeros(sem[b].shape, dtype=np.uint8)
                    gt_tri[tri[b] == 0] = 1
                    gt_tri[positive] = 2
                    pred_tri = np.where(sem[b] == 0, 1, 2).astype(np.uint8)
                    update_counts(acc, sem[b], pred_tri, gt_sem, gt_tri)
                bucket = []
                seen += 1
                if seen >= limit:
                    return finalize_counts(acc)
        if bucket:
            pts, mask = collate(bucket, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model.encoder(pts, mask)
            target_sem = targets_for(bucket, device)
            sem = pred["sem"].argmax(dim=1).detach().cpu().numpy()
            tri = target_sem.detach().cpu().numpy()
            for b in range(sem.shape[0]):
                gt_sem = np.zeros(sem[b].shape, dtype=np.int64)
                positive = tri[b] > 0
                gt_sem[positive] = tri[b][positive]
                gt_tri = np.zeros(sem[b].shape, dtype=np.uint8)
                gt_tri[tri[b] == 0] = 1
                gt_tri[positive] = 2
                pred_tri = np.where(sem[b] == 0, 1, 2).astype(np.uint8)
                update_counts(acc, sem[b], pred_tri, gt_sem, gt_tri)
    return finalize_counts(acc)


def main() -> None:
    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with (CONFIGS / "mini200.yaml").open("r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda")
    train = Mini200(data_cfg["root"], "train", int(data_cfg["val_every"]))
    val = Mini200(data_cfg["root"], "val", int(data_cfg["val_every"]))
    model = BorrowedSight("B1", feat_dim=int(cfg["feat_dim"]), codebook_size=int(cfg["codebook_size"])).to(device)
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    indices = np.arange(len(train))
    rng = np.random.default_rng(int(cfg["seed"]))
    cursor = 0
    bucket = []
    losses = []
    started = time.perf_counter()
    limit_s = float(cfg["max_minutes_sender"]) * 60.0
    vram_cap = float(cfg["vram_abort_gb"]) * (1024 ** 3)
    status = "running"
    step = 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > limit_s:
            status = "time_cap"
            break
        if cursor >= len(indices):
            rng.shuffle(indices)
            cursor = 0
        for sweep in sweeps(train[int(indices[cursor])]):
            bucket.append(sweep)
        cursor += 1
        if len(bucket) < int(cfg["batch_size"]):
            continue
        batch = bucket[: int(cfg["batch_size"])]
        bucket = bucket[int(cfg["batch_size"]) :]
        pts, mask = collate(batch, device)
        target = targets_for(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model.encoder(pts, mask)
        loss = weighted_ce(pred["sem"], target)
        if not torch.isfinite(loss):
            status = "nan"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        step += 1
        peak = torch.cuda.max_memory_allocated()
        if peak >= vram_cap:
            status = "vram_abort"
            break
        if step >= int(cfg["min_steps"]) and len(losses) >= 40:
            base = float(np.mean(losses[:20]))
            recent = float(np.mean(losses[-20:]))
            if base > 0 and recent < float(cfg["loss_drop_ratio"]) * base:
                status = "loss_dropped"
                break
        if step % 20 == 0:
            print(f"step {step} loss {losses[-1]:.4f} elapsed {elapsed:.1f}s peak_gb {peak/1024**3:.2f}", flush=True)
    out_dir = OUTPUTS / "phase3_sender"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loss.json").write_text(
        json.dumps({"status": status, "steps": step, "losses": losses, "seconds": time.perf_counter() - started}, indent=2),
        encoding="utf-8",
    )
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINTS / "sender_mini.pt"
    torch.save({"encoder": model.encoder.state_dict(), "status": status, "steps": step, "cfg": cfg}, ckpt)
    metrics = None
    if status != "nan":
        metrics = evaluate(model, val, device)
    log_run(
        exp_id=f"sender_mini_seed{cfg['seed']}",
        model_name="sender_mini",
        seed=int(cfg["seed"]),
        bytes_per_frame=0.0,
        metrics=metrics,
        status=status,
    )
    print(json.dumps({"status": status, "steps": step, "metrics": metrics}, default=str), flush=True)
    if status == "nan":
        raise SystemExit(2)
    if status == "vram_abort":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
