"""Scene-Grouped 5-Fold Cross-Validation on Real OPV2V + Semantic-OPV2V.

Strictly groups entire driving scenarios (scenes) into 5 disjoint folds so that
no two temporally adjacent frames from the same scenario ever span train and
validation sets (`train_scenes & val_scenes == empty`).

On every fold, trains B3 (shared codebook, no envelope, top-k) and B4
(BorrowedSight: grouped codebooks, quota, 512 B visibility envelope) from
scratch on identical frozen sender features, computes fold-wise class weights
strictly from that fold's training scenes, evaluates strictly through
`pack_bytes -> unpack_bytes` at matched byte budgets, and reports 5-fold
mean +- std plus paired fold-by-fold statistical significance (t-test).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from borrowedsight.data import loss_targets
from borrowedsight.labels import IGNORE_INDEX
from borrowedsight.model import BorrowedSight, score_bce
from borrowedsight.packer import FAIR_BUDGET_BYTES, TARGET_BYTES, packet_layout
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.train_receivers_real import (
    compute_class_weights,
    evaluate_frames,
    load_cached_split,
    move_sender,
    set_seed,
)

CACHE_ROOT = DATA_ROOT / "cache" / "opv2v_clear"
OUT = OUTPUTS / "real_opv2v" / "cv5"


def make_scene_folds(all_frames: list[dict], k_folds: int = 5) -> list[list[str]]:
    """Deterministically partition unique scenes into K balanced disjoint folds."""
    by_scene: dict[str, int] = defaultdict(int)
    for fr in all_frames:
        by_scene[fr["scene_name"]] += 1
    ranked = sorted(by_scene.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < k_folds:
        raise RuntimeError(f"Need at least {k_folds} distinct scenes for {k_folds}-fold CV, found {len(ranked)}")
    folds: list[list[str]] = [[] for _ in range(k_folds)]
    fold_loads = [0] * k_folds
    # Greedy load-balanced assignment by scene frame count
    for scene_name, count in ranked:
        target = int(np.argmin(fold_loads))
        folds[target].append(scene_name)
        fold_loads[target] += count
    for f in folds:
        f.sort()
    return folds


def paired_ttest_one_sided(diffs: list[float]) -> dict:
    """Paired one-sided t-test for mean(B4 - B3) > 0 (and two-sided p-value)."""
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n <= 1 or std <= 1e-12:
        return {"mean_diff": mean, "std_diff": std, "t_stat": None, "p_two_sided": None, "p_one_sided": None}
    t_stat = mean / (std / math.sqrt(n))
    try:
        from scipy import stats

        p_two = float(stats.t.sf(abs(t_stat), df=n - 1) * 2.0)
        p_one = float(stats.t.sf(t_stat, df=n - 1))
    except Exception:
        if n - 1 == 4:
            def _t_sf_df4(val: float) -> float:
                u = val * val + 4.0
                return 0.5 - (val / (2.0 * math.sqrt(u))) * (1.0 + 2.0 / u)
            p_one = float(_t_sf_df4(t_stat))
            p_two = float(2.0 * _t_sf_df4(abs(t_stat)))
        else:
            p_two, p_one = None, None
    return {
        "mean_diff": mean,
        "std_diff": std,
        "t_stat": float(t_stat),
        "p_two_sided": p_two,
        "p_one_sided": p_one,
    }


def train_fold_model(
    method: str,
    model_kwargs: dict,
    target_bytes: int,
    fold_id: int,
    train_frames: list[dict],
    val_frames: list[dict],
    class_weights: torch.Tensor,
    sender_state: dict,
    cfg: dict,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    set_seed(100 + fold_id)
    model = BorrowedSight(
        method,
        feat_dim=int(cfg["feat_dim"]),
        codebook_size=int(cfg["codebook_size"]),
        **model_kwargs,
    ).to(device)
    model.encoder.load_state_dict(sender_state)
    for p in model.encoder.parameters():
        p.requires_grad = False

    budget = packet_layout(method, target=int(target_bytes), stage=model.default_stage, include_envelope=model.use_envelope)
    n_tokens = int(budget["n_tokens"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    rng = np.random.default_rng(100 + fold_id)

    best_score = -1.0
    best_state = None
    t0 = time.perf_counter()
    bs = max(1, int(batch_size))

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_frames))
        for start in range(0, len(order), bs):
            batch_ids = order[start : start + bs]
            opt.zero_grad(set_to_none=True)
            placed_b, poses_b, stamps_b, motions_b, sem_ts, tri_ts, vq_terms, score_terms = [], [], [], [], [], [], [], []
            for idx in batch_ids:
                fr = train_frames[int(idx)]
                sent = [
                    model.transmit(
                        move_sender(row, device),
                        fr["ego_pose"],
                        int(fr["receiver_id"]),
                        n_tokens,
                        budget=budget,
                        serialize=False,
                    )
                    for row in fr["senders"]
                ]
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
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        if ep == epochs or ep % 5 == 0:
            val_m = evaluate_frames(model, val_frames, budget, device)
            sc = float(val_m["mIoU"] or 0.0)
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_m = evaluate_frames(model, val_frames, budget, device)
    final_m["seconds"] = time.perf_counter() - t0
    return final_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--budgets", type=int, nargs="+", default=[TARGET_BYTES])
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "validate", "test"])
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    ckpt = torch.load(CHECKPOINTS / "sender_opv2v.pt", map_location=device, weights_only=False)
    sender_state = ckpt["encoder"]

    all_frames = []
    for sp in args.splits:
        sp_dir = CACHE_ROOT / sp
        if sp_dir.is_dir():
            all_frames.extend(load_cached_split(sp_dir))
    folds = make_scene_folds(all_frames, k_folds=args.folds)
    print(f"Scene-Grouped {args.folds}-Fold CV over {len(all_frames)} frames across {sum(len(f) for f in folds)} unique scenes:", flush=True)
    for fi, fl in enumerate(folds):
        nf = sum(1 for fr in all_frames if fr["scene_name"] in set(fl))
        print(f"  Fold {fi}: {len(fl)} scenes ({nf} frames) -> {fl}", flush=True)

    rows = []
    for budget_b in args.budgets:
        for fi in range(args.folds):
            val_scenes = set(folds[fi])
            train_scenes = {sc for fj, fl in enumerate(folds) if fj != fi for sc in fl}
            assert train_scenes.isdisjoint(val_scenes), "Scene leakage detected across CV folds!"
            tr_f = [fr for fr in all_frames if fr["scene_name"] in train_scenes]
            va_f = [fr for fr in all_frames if fr["scene_name"] in val_scenes]
            cw = compute_class_weights(tr_f, device)
            for tag, method, mkw in (
                ("B3", "B3", {}),
                ("B4", "B4", dict(use_envelope=True, use_grouped_books=True, use_quota=True, default_stage=1)),
            ):
                m = train_fold_model(
                    method=method,
                    model_kwargs=mkw,
                    target_bytes=budget_b,
                    fold_id=fi,
                    train_frames=tr_f,
                    val_frames=va_f,
                    class_weights=cw,
                    sender_state=sender_state,
                    cfg=cfg,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                    batch_size=args.batch_size,
                )
                row = {
                    "budget_bytes": int(budget_b),
                    "fold": int(fi),
                    "method": tag,
                    "n_train_scenes": len(train_scenes),
                    "n_val_scenes": len(val_scenes),
                    "n_train_frames": len(tr_f),
                    "n_val_frames": len(va_f),
                    "bytes_per_frame": m["bytes_per_frame"],
                    "mIoU": m["mIoU"],
                    "vehicle_IoU": m["vehicle_IoU"],
                    "road_IoU": m["road_IoU"],
                    "unknown_F1": m["unknown_F1"],
                    "unobserved_as_free_err": m["unobserved_as_free_err"],
                    "codebook_usage": m["codebook_usage"],
                    "road_token_fraction": m["road_token_fraction"],
                    "seconds": m["seconds"],
                }
                rows.append(row)
                print(
                    f"  [B={budget_b} Fold {fi} {tag}] mIoU={m['mIoU']:.4f} veh={m['vehicle_IoU']:.4f} "
                    f"road={m['road_IoU']:.4f} unkF1={m['unknown_F1']:.4f} UFE={m['unobserved_as_free_err']:.4f} "
                    f"({m['seconds']:.1f}s)",
                    flush=True,
                )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"folds": folds, "epochs": args.epochs, "splits_included": args.splits, "by_budget": {}}
    md = [
        f"# Scene-Grouped {args.folds}-Fold Cross-Validation Summary (Real OPV2V)",
        "",
        f"- Total unique scenes: {sum(len(f) for f in folds)}, total frames: {len(all_frames)}, epochs/fold: {args.epochs}",
        "- Strict scene-level disjointness verified on every fold (`train_scenes & val_scenes == empty`).",
        "",
    ]
    for budget_b in args.budgets:
        sub = [r for r in rows if r["budget_bytes"] == int(budget_b)]
        b3_rows = sorted([r for r in sub if r["method"] == "B3"], key=lambda r: r["fold"])
        b4_rows = sorted([r for r in sub if r["method"] == "B4"], key=lambda r: r["fold"])
        agg = {}
        for metric in ("mIoU", "vehicle_IoU", "road_IoU", "unknown_F1", "unobserved_as_free_err"):
            v3 = [float(r[metric]) for r in b3_rows if r[metric] is not None]
            v4 = [float(r[metric]) for r in b4_rows if r[metric] is not None]
            diffs = [a - b for a, b in zip(v4, v3)]
            agg[metric] = {
                "B3_mean": float(np.mean(v3)),
                "B3_std": float(np.std(v3, ddof=1)) if len(v3) > 1 else 0.0,
                "B4_mean": float(np.mean(v4)),
                "B4_std": float(np.std(v4, ddof=1)) if len(v4) > 1 else 0.0,
                "paired_B4_minus_B3": paired_ttest_one_sided(diffs),
            }
        summary["by_budget"][str(budget_b)] = agg
        md.append(f"## Budget = {budget_b:,} B")
        md.append("| Metric | B3 (mean ± std) | B4 (mean ± std) | Paired Δ (B4 − B3) | One-sided p |")
        md.append("| --- | ---: | ---: | ---: | ---: |")
        for metric, st in agg.items():
            p_str = "--" if st["paired_B4_minus_B3"]["p_one_sided"] is None else f"{st['paired_B4_minus_B3']['p_one_sided']:.4g}"
            md.append(
                f"| {metric} | {st['B3_mean']:.4f} ± {st['B3_std']:.4f} | {st['B4_mean']:.4f} ± {st['B4_std']:.4f} | "
                f"{st['paired_B4_minus_B3']['mean_diff']:+.4f} ± {st['paired_B4_minus_B3']['std_diff']:.4f} | {p_str} |"
            )
        md.append("")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWROTE {OUT / 'summary.md'} and {OUT / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
