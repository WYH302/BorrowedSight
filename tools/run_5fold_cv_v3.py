"""Synchronized BorrowedSight v3.0 Scene-Grouped 5-Fold Cross-Validation across all 25 Real OPV2V scenarios.

Eliminates any version mismatch between Table II/III and Table IV:
- Uses the EXACT same frozen SenderEncoder v3 sparse caches (`opv2v_v3_clear/{train,validate,test}` pooled across 25 scenes).
- Uses the EXACT same final v3.0 architecture (`B4-TDVF v3` vs `B3-Compact v3`, K=64, Lovasz+UFE+D4+Warmup+EMA).
- Strictly disjoint 5-scene folds (`train_scenes & val_scenes == empty`); class weights recomputed per fold on train scenes ONLY.
- Runs BOTH 32,768 B and 16,384 B (20 models total), resume-safe per fold, with paired one-sided & two-sided t-tests!
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
import yaml

from borrowedsight.packer import FAIR_BUDGET_BYTES, TARGET_BYTES
from borrowedsight.paths import CHECKPOINTS, CONFIGS, DATA_ROOT, OUTPUTS
from tools.run_5fold_cv_real import make_scene_folds, paired_ttest_one_sided
from tools.train_receivers_real import compute_class_weights, load_cached_split
from tools.train_v3_common import train_or_load_v3

CACHE_DIR = DATA_ROOT / "cache" / "opv2v_v3_clear"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v3_cv5"
OUT = OUTPUTS / "real_opv2v" / "v3_cv5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=8.0e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ufe-w", type=float, default=0.25)
    ap.add_argument("--use-sem-film", action="store_true")
    ap.add_argument("--fusion-mode", type=str, default="mean", choices=["mean", "softmax"])
    ap.add_argument("--softmax-temp", type=float, default=4.0)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    all_frames = (
        load_cached_split(CACHE_DIR / "train")
        + load_cached_split(CACHE_DIR / "validate")
        + load_cached_split(CACHE_DIR / "test")
    )
    folds = make_scene_folds(all_frames, k_folds=5)
    print(f"V3 5-Fold CV pooled {len(all_frames)} frames across 25 scenes into 5 disjoint folds:", flush=True)
    for fi, scs in enumerate(folds):
        n_fr = sum(1 for fr in all_frames if fr["scene_name"] in set(scs))
        print(f"  Fold {fi}: {len(scs)} scenes, {n_fr} frames -> {scs}", flush=True)

    b4_kw = dict(
        codebook_size=64,
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
        use_sarc=True,
    )
    if args.use_sem_film:
        b4_kw["use_sem_film"] = True
    if args.fusion_mode == "softmax":
        b4_kw["fusion_mode"] = "softmax"
        b4_kw["softmax_temp"] = float(args.softmax_temp)

    b3_kw = dict(
        codebook_size=64,
        use_envelope=False,
        default_stage=2,
        compact_wire=True,
        use_norm_pe=True,
        use_time_decay=True,
    )

    results_by_budget = {}
    t_all = time.perf_counter()

    for tb in (TARGET_BYTES, FAIR_BUDGET_BYTES):
        fold_rows = []
        for fi, val_scenes in enumerate(folds):
            val_set = set(val_scenes)
            tr_fr = [fr for fr in all_frames if fr["scene_name"] not in val_set]
            va_fr = [fr for fr in all_frames if fr["scene_name"] in val_set]
            assert not ({fr["scene_name"] for fr in tr_fr} & {fr["scene_name"] for fr in va_fr})
            cw = compute_class_weights(tr_fr, device)

            r_b3 = train_or_load_v3(
                tag=f"cv5_f{fi}_B3_{tb}",
                method="B3",
                model_kwargs=dict(b3_kw),
                target_bytes=tb,
                seed=100 + fi,
                train_frames=tr_fr,
                val_frames=va_fr,
                test_frames=[],
                class_weights=cw,
                cfg=cfg,
                ckpt_dir=CKPT_DIR,
                epochs=args.epochs,
                lr=args.lr,
                warmup_epochs=2,
                ema_decay=0.995,
                device=device,
                batch_size=args.batch_size,
                lovasz_w=0.5,
                ufe_w=args.ufe_w,
            )
            r_b4 = train_or_load_v3(
                tag=f"cv5_f{fi}_B4_{tb}",
                method="B4",
                model_kwargs=dict(b4_kw),
                target_bytes=tb,
                seed=100 + fi,
                train_frames=tr_fr,
                val_frames=va_fr,
                test_frames=[],
                class_weights=cw,
                cfg=cfg,
                ckpt_dir=CKPT_DIR,
                epochs=args.epochs,
                lr=args.lr,
                warmup_epochs=2,
                ema_decay=0.995,
                device=device,
                batch_size=args.batch_size,
                lovasz_w=0.5,
                ufe_w=args.ufe_w,
            )
            fold_rows.append(
                {
                    "fold": fi,
                    "val_scenes": val_scenes,
                    "n_val_frames": len(va_fr),
                    "B3": r_b3["val"],
                    "B4": r_b4["val"],
                    "seconds": round(r_b3["seconds"] + r_b4["seconds"], 1),
                }
            )
            # Incremental save
            (OUT / f"cv5_{tb}_partial.json").write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")

        agg = {}
        for metric in ("mIoU", "vehicle_IoU", "road_IoU", "unknown_F1", "unobserved_as_free_err"):
            v3 = [r["B3"][metric] for r in fold_rows]
            v4 = [r["B4"][metric] for r in fold_rows]
            diffs = [a - b for a, b in zip(v4, v3)]
            agg[metric] = {
                "B3_mean": float(np.mean(v3)),
                "B3_std": float(np.std(v3, ddof=1)),
                "B4_mean": float(np.mean(v4)),
                "B4_std": float(np.std(v4, ddof=1)),
                "paired_ttest": paired_ttest_one_sided(diffs),
            }
        results_by_budget[str(tb)] = {"folds": fold_rows, "aggregate": agg}

    summary = {
        "config": vars(args),
        "folds": folds,
        "results": results_by_budget,
        "total_seconds": round(time.perf_counter() - t_all, 1),
    }
    (OUT / "cv5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUT / "cv5_folds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "fold", "n_val", "B3_mIoU", "B4_mIoU", "d_mIoU", "B3_veh", "B4_veh", "B3_UFE", "B4_UFE", "B3_unkF1", "B4_unkF1"])
        for tb_str, block in results_by_budget.items():
            for r in block["folds"]:
                b3, b4 = r["B3"], r["B4"]
                w.writerow([
                    tb_str, r["fold"], r["n_val_frames"],
                    round(b3["mIoU"], 4), round(b4["mIoU"], 4), round(b4["mIoU"] - b3["mIoU"], 4),
                    round(b3["vehicle_IoU"], 4), round(b4["vehicle_IoU"], 4),
                    round(b3["unobserved_as_free_err"], 4), round(b4["unobserved_as_free_err"], 4),
                    round(b3["unknown_F1"], 4), round(b4["unknown_F1"], 4),
                ])

    print(f"\n=== V3 5-FOLD CV COMPLETE ({(time.perf_counter()-t_all)/3600:.2f} h) ===", flush=True)
    for tb_str, block in results_by_budget.items():
        agg = block["aggregate"]
        print(f"Budget {tb_str} B:")
        for met, vals in agg.items():
            tt = vals["paired_ttest"]
            print(
                f"  {met:24s}: B3={vals['B3_mean']:.4f}+-{vals['B3_std']:.4f} | "
                f"B4={vals['B4_mean']:.4f}+-{vals['B4_std']:.4f} | "
                f"diff={tt['mean_diff']:+.4f}, t={tt['t_stat']}, p1={tt['p_one_sided']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
