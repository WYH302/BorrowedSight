"""BorrowedSight v3.0 Stop-Loss & Principled Hyperparameter Gate (Seed 0).

Evaluates every high-ROI v3.0 candidate strictly on VALIDATE (plus reports clean TEST for full transparency,
while locking all hyperparameter choices on VALIDATE ONLY!):
1. Free 6-bit codebook K=64 vs K=32 (0 extra wire bytes).
2. UFE loss weight Pareto sweep: ufe_w in {0.15, 0.25, 0.35}.
3. TDVF pure vs TDVF + Zero-Init FiLM height-prior gate (`use_sem_film=True`).
4. Mean fusion vs Confidence-Competitive Softmax multi-sender fusion (`fusion_mode="softmax"`),
   benchmarked on BOTH Clean and 0.5m/2deg Pose Error on VALIDATE!
5. B4-v3 vs B4-v3 w/o Env vs B3-v3 (32 KB and 16 KB) stop-loss verification.

All checkpoints saved to `checkpoints/real_opv2v_v3/` are automatically reused by `tools/run_v3_full.py`!
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
from tools.run_v2_gate import eval_serialized
from tools.train_receivers_real import compute_class_weights, load_cached_split
from tools.train_v3_common import build_v3_model, train_or_load_v3

CACHE_DIR = DATA_ROOT / "cache" / "opv2v_v3_clear"
CKPT_DIR = CHECKPOINTS / "real_opv2v_v3"
OUT = OUTPUTS / "real_opv2v" / "v3_gate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=8.0e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    with (CONFIGS / "train.yaml").open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    train_frames = load_cached_split(CACHE_DIR / "train")
    val_frames = load_cached_split(CACHE_DIR / "validate")
    test_frames = load_cached_split(CACHE_DIR / "test")
    val_lookup = {(fr["scene_name"], fr["frame"]): fr for fr in val_frames}
    class_weights = compute_class_weights(train_frames, device)
    print(f"V3 Gate loaded train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}", flush=True)

    base_b4 = dict(
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
    )
    base_b3 = dict(
        codebook_size=64,
        use_envelope=False,
        default_stage=2,
        compact_wire=True,
        use_norm_pe=True,
        use_time_decay=True,
    )

    # Gate candidates: (tag, method, kwargs, target_bytes, ufe_w)
    b4_sarc = {**base_b4, "use_sarc": True}
    specs = [
        ("B3_v3_32k", "B3", dict(base_b3), TARGET_BYTES, 0.25),
        ("B4_v3_32k_u25", "B4", dict(base_b4), TARGET_BYTES, 0.25),
        ("B4_v3_32k_sarc", "B4", dict(b4_sarc), TARGET_BYTES, 0.25),
        ("B4_v3_32k_sarc_u15", "B4", dict(b4_sarc), TARGET_BYTES, 0.15),
        ("B4_v3_32k_sarc_u35", "B4", dict(b4_sarc), TARGET_BYTES, 0.35),
        ("B4_v3_32k_sarc_K32", "B4", {**b4_sarc, "codebook_size": 32}, TARGET_BYTES, 0.25),
        ("B4_v3_32k_sarc_film", "B4", {**b4_sarc, "use_sem_film": True}, TARGET_BYTES, 0.25),
        ("B4_v3_32k_sarc_softmax", "B4", {**b4_sarc, "fusion_mode": "softmax", "softmax_temp": 4.0}, TARGET_BYTES, 0.25),
        ("B4_v3_32k_sarc_no_env", "B4", {**b4_sarc, "use_envelope": False, "use_zmax_env": False, "decouple_env": False}, TARGET_BYTES, 0.25),
        ("B3_v3_16k", "B3", dict(base_b3), FAIR_BUDGET_BYTES, 0.25),
        ("B4_v3_16k_sarc", "B4", dict(b4_sarc), FAIR_BUDGET_BYTES, 0.25),
    ]

    records = []
    t_all = time.perf_counter()
    for tag, method, mkw, tb, ufe_w in specs:
        rec = train_or_load_v3(
            tag=tag,
            method=method,
            model_kwargs=mkw,
            target_bytes=tb,
            seed=0,
            train_frames=train_frames,
            val_frames=val_frames,
            test_frames=test_frames,
            class_weights=class_weights,
            cfg=cfg,
            ckpt_dir=CKPT_DIR,
            epochs=args.epochs,
            lr=args.lr,
            warmup_epochs=2,
            ema_decay=0.995,
            device=device,
            batch_size=args.batch_size,
            lovasz_w=0.5,
            ufe_w=ufe_w,
        )
        # Also evaluate 0.5m/2deg pose noise on VALIDATE for mean vs softmax comparison
        if tag in {"B4_v3_32k_sarc", "B4_v3_32k_sarc_softmax"}:
            ckpt = torch.load(CKPT_DIR / f"{tag}_s0.pt", map_location=device, weights_only=False)
            m_eval, _ = build_v3_model(method, mkw, cfg, CHECKPOINTS / "sender_opv2v_v3.pt", device)
            m_eval.load_state_dict(ckpt["model"])
            m_eval.eval()
            val_pose = eval_serialized(
                m_eval, val_frames, val_lookup, tb, device, np.random.default_rng(7),
                xy_std=0.5, yaw_std=2.0,
            )
            rec["val_pose05"] = val_pose
            print(
                f"  [POSE 0.5m/2deg VAL] {tag}: mIoU={val_pose['mIoU']:.4f} veh={val_pose['vehicle_IoU']:.4f} UFE={val_pose['unobserved_as_free_err']:.4f}",
                flush=True,
            )
        records.append(rec)

        # Incremental flush after every model
        (OUT / "gate_summary.json").write_text(json.dumps({"records": records, "elapsed_s": time.perf_counter() - t_all}, indent=2), encoding="utf-8")
        with (OUT / "gate_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tag", "method", "bytes", "ufe_w", "val_mIoU", "val_veh", "val_UFE", "val_unkF1", "test_mIoU", "test_veh", "test_road", "test_UFE", "test_unkF1", "usage", "best_src", "seconds"])
            for r in records:
                vm, tm = r["val"], r["test"]
                w.writerow([
                    r["tag"], r["method"], r["target_bytes"], r["train_cfg"]["ufe_w"],
                    round(vm["mIoU"], 4), round(vm["vehicle_IoU"], 4), round(vm["unobserved_as_free_err"], 4), round(vm["unknown_F1"], 4),
                    round(tm["mIoU"], 4), round(tm["vehicle_IoU"], 4), round(tm["road_IoU"], 4), round(tm["unobserved_as_free_err"], 4), round(tm["unknown_F1"], 4),
                    round(tm["codebook_usage"], 4) if tm["codebook_usage"] is not None else "",
                    r["best_src"], r["seconds"],
                ])

    print(f"\n=== V3 GATE COMPLETE ({(time.perf_counter()-t_all)/60:.1f} min) ===", flush=True)
    for r in records:
        vm, tm = r["val"], r["test"]
        print(
            f"  {r['tag']:20s} | VAL mIoU={vm['mIoU']:.4f} UFE={vm['unobserved_as_free_err']:.4f} | "
            f"TEST mIoU={tm['mIoU']:.4f} veh={tm['vehicle_IoU']:.4f} road={tm['road_IoU']:.4f} UFE={tm['unobserved_as_free_err']:.4f} unkF1={tm['unknown_F1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
