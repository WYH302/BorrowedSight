"""BorrowedSight v3.0 Full Experimental Suite (IEEE Transactions Flagship).

Trains (or safely resumes from v3_gate checkpoints in `checkpoints/real_opv2v_v3/`):
1. Main 32,768 B Comparison (100% Equal-Byte, Frozen SenderEncoder v3):
   - B0 (Zero-Ego No-Comm)
   - B1 (Dense FP16 36 B/tok)
   - B2 (Thin FP16 20 B/tok)
   - Where2comm-Occ (NeurIPS'22 Confidence-Gated Continuous 20 B/tok, 3 seeds)
   - CodeFilling-Occ (CVPR'24 Monolithic Shared VQ + Codebook Spatial Completion 4/5 B/tok, K=64, 3 seeds)
   - B3-Compact (Monolithic VQ 5 B/tok, K=64, 3 seeds)
   - B4-TDVF v3 (BorrowedSight Flagship: 5 B/tok, K=64, Grouped Books, Quota, 50x50+z_max Zstd Env, TDVF, 3 seeds)
2. Tight Half-Bandwidth 16,384 B Comparison:
   - B1, B2, Where2comm-Occ, CodeFilling-Occ (3 seeds), B3-Compact (3 seeds), B4-TDVF v3 (3 seeds)
3. Strictly Orthogonal 32,768 B Ablations (1 factor changed at a time from B4-TDVF v3):
   - w/o Visibility Envelope (`use_envelope=False`)
   - w/o TDVF (Early Concatenation, `decouple_env=False`)
   - w/o Class Anchor (`use_cls_embed=False`)
   - Shared 192-Entry Book (`use_grouped_books=False`)
   - w/o Category Quota (`use_quota=False`)
   - Stage-1 Only (`default_stage=1`)
   - K=32 vs K=64 (`codebook_size=32`)
   - w/o Warmup+EMA (`warmup_epochs=0, ema_decay=0.0`)
4. Multi-Bandwidth Single-Checkpoint Model (35 epochs):
   - B4-TDVF-MultiBW v3 trained over {4, 8, 16, 32} KB.
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
from tools.train_receivers_real import compute_class_weights, load_cached_split
from tools.train_v3_common import train_or_load_v3

CACHE_DIR = DATA_ROOT / "cache" / "opv2v_v3_clear"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v3"
OUT = OUTPUTS / "real_opv2v" / "v3_full"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--multibw-epochs", type=int, default=35)
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

    train_frames = load_cached_split(CACHE_DIR / "train")
    val_frames = load_cached_split(CACHE_DIR / "validate")
    test_frames = load_cached_split(CACHE_DIR / "test")
    class_weights = compute_class_weights(train_frames, device)
    print(
        f"V3 Full loaded train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)} | "
        f"ufe_w={args.ufe_w}, film={args.use_sem_film}, fusion={args.fusion_mode}",
        flush=True,
    )

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
    cf_kw = {**b3_kw, "use_code_fill": True}
    w2c_kw = dict(use_envelope=False, use_norm_pe=True, use_time_decay=True, use_w2c_gate=True)
    b12_kw = dict(use_envelope=False, use_norm_pe=True, use_time_decay=True)

    # Specs: (tag, method, kwargs, target_bytes, seed, train_budgets, ep_override, warmup_override, ema_override)
    specs = [
        # --- 32 KB Core + Public SOTA Equal-Byte Baselines ---
        ("B0_v3_32k", "B0", {}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B1_v3_32k", "B1", dict(b12_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B2_v3_32k", "B2", dict(b12_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("Where2comm_v3_32k", "B2", dict(w2c_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("Where2comm_v3_32k", "B2", dict(w2c_kw), TARGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("Where2comm_v3_32k", "B2", dict(w2c_kw), TARGET_BYTES, 2, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_32k", "B3", dict(cf_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_32k", "B3", dict(cf_kw), TARGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_32k", "B3", dict(cf_kw), TARGET_BYTES, 2, None, args.epochs, 2, 0.995),
        ("B3_v3_32k", "B3", dict(b3_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B3_v3_32k", "B3", dict(b3_kw), TARGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("B3_v3_32k", "B3", dict(b3_kw), TARGET_BYTES, 2, None, args.epochs, 2, 0.995),
        ("B4_v3_32k", "B4", dict(b4_kw), TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k", "B4", dict(b4_kw), TARGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("B4_v3_32k", "B4", dict(b4_kw), TARGET_BYTES, 2, None, args.epochs, 2, 0.995),
        # --- 16 KB Tight Half-Bandwidth ---
        ("B1_v3_16k", "B1", dict(b12_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B2_v3_16k", "B2", dict(b12_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("Where2comm_v3_16k", "B2", dict(w2c_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_16k", "B3", dict(cf_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_16k", "B3", dict(cf_kw), FAIR_BUDGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("CodeFilling_v3_16k", "B3", dict(cf_kw), FAIR_BUDGET_BYTES, 2, None, args.epochs, 2, 0.995),
        ("B3_v3_16k", "B3", dict(b3_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B3_v3_16k", "B3", dict(b3_kw), FAIR_BUDGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("B3_v3_16k", "B3", dict(b3_kw), FAIR_BUDGET_BYTES, 2, None, args.epochs, 2, 0.995),
        ("B4_v3_16k", "B4", dict(b4_kw), FAIR_BUDGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_16k", "B4", dict(b4_kw), FAIR_BUDGET_BYTES, 1, None, args.epochs, 2, 0.995),
        ("B4_v3_16k", "B4", dict(b4_kw), FAIR_BUDGET_BYTES, 2, None, args.epochs, 2, 0.995),
        # --- Strictly Orthogonal 32 KB Ablations (1 factor changed from B4_v3_32k) ---
        ("B4_v3_32k_no_env", "B4", {**{k: v for k, v in b4_kw.items() if k != "use_sem_film"}, "use_envelope": False, "use_zmax_env": False, "decouple_env": False}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_no_tdvf", "B4", {**{k: v for k, v in b4_kw.items() if k != "use_sem_film"}, "decouple_env": False}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_no_cls", "B4", {**b4_kw, "use_cls_embed": False, "use_sarc": False, "true_residual": False}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_no_sarc", "B4", {**{k: v for k, v in b4_kw.items() if k != "use_sarc"}}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_shared", "B4", {**b4_kw, "use_grouped_books": False}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_no_quota", "B4", {**b4_kw, "use_quota": False}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_stage1", "B4", {**b4_kw, "default_stage": 1}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_K32", "B4", {**b4_kw, "codebook_size": 32}, TARGET_BYTES, 0, None, args.epochs, 2, 0.995),
        ("B4_v3_32k_no_ema", "B4", dict(b4_kw), TARGET_BYTES, 0, None, args.epochs, 0, 0.0),
        # --- Multi-Bandwidth Single-Checkpoint Flagship (35 ep) ---
        ("B4_v3_multibw", "B4", dict(b4_kw), TARGET_BYTES, 0, (4096, 8192, 16384, 32768), args.multibw_epochs, 2, 0.995),
    ]

    records = []
    t_all = time.perf_counter()
    for tag, method, mkw, tb, sd, t_bgt, ep_n, wu, ema in specs:
        rec = train_or_load_v3(
            tag=tag,
            method=method,
            model_kwargs=mkw,
            target_bytes=tb,
            seed=sd,
            train_frames=train_frames,
            val_frames=val_frames,
            test_frames=test_frames,
            class_weights=class_weights,
            cfg=cfg,
            ckpt_dir=CKPT_DIR,
            epochs=ep_n,
            lr=args.lr,
            warmup_epochs=wu,
            ema_decay=ema,
            device=device,
            batch_size=args.batch_size,
            train_budgets=t_bgt,
            lovasz_w=0.5,
            ufe_w=args.ufe_w,
        )
        records.append(rec)

        # Incremental flush after every model
        (OUT / "summary.json").write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "records": records,
                    "total_wall_seconds": round(time.perf_counter() - t_all, 1),
                    "total_train_seconds": round(sum(r["seconds"] for r in records), 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag", "method", "seed", "params", "bytes", "tok_bytes", "n_tokens",
                    "val_mIoU", "val_veh", "val_UFE",
                    "test_mIoU", "test_fg_mIoU", "test_veh", "test_road", "test_bldg",
                    "test_unkF1", "test_UFE", "test_Brier", "usage", "road_tok_frac",
                    "best_src", "seconds", "peak_vram_gb",
                ]
            )
            for r in records:
                vm, tm = r["val"], r["test"]
                w.writerow(
                    [
                        r["tag"], r["method"], r["seed"], r["params"], r["target_bytes"],
                        r["token_bytes"], r["n_tokens"],
                        round(vm["mIoU"], 4), round(vm["vehicle_IoU"], 4), round(vm["unobserved_as_free_err"], 4),
                        round(tm["mIoU"], 4), round(tm.get("fg_mIoU", tm["mIoU"]), 4),
                        round(tm["vehicle_IoU"], 4), round(tm["road_IoU"], 4),
                        round(tm.get("per_class_iou", {}).get("building", 0.0), 4),
                        round(tm["unknown_F1"], 4), round(tm["unobserved_as_free_err"], 4),
                        round(tm.get("tri_brier", 0.0), 4),
                        round(tm["codebook_usage"], 4) if tm["codebook_usage"] is not None else "",
                        round(tm["road_token_fraction"], 4) if tm["road_token_fraction"] is not None else "",
                        r["best_src"], r["seconds"], r["peak_vram_gb"],
                    ]
                )

    print(f"\n=== V3 FULL SUITE COMPLETE ({len(records)} models, wall={(time.perf_counter()-t_all)/3600:.2f} h) ===", flush=True)


if __name__ == "__main__":
    main()
