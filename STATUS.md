# BorrowedSight Project Status

Updated: 2026-11-23 (+08:00).

## Status: FULL-SCALE PRODUCTION EXPERIMENTS, 5-FOLD CV, FIGURES & MANUSCRIPT COMPLETE

1. **Full Multi-Sender Real OPV2V + Semantic-OPV2V (`4LidarSurround`) Dataset (25 Scenes, 2,383 Frames, 0 Errors)**:
   - Raw OPV2V (`D:\bs_data\BorrowedSight\OPV2V`, 108.35 GiB, 69 scenes) verified 0 missing files.
   - Stream-voxelized 25 complete Semantic-OPV2V scenes (`D:\bs_data\BorrowedSight\semantic_voxels`):
     - `train`: 13 scenes, 1,084 frames, 2–4 CAVs (mean 2.62), 12/12 classes.
     - `validate`: 6 scenes, 573 frames, 2–5 CAVs (mean 2.67), 12/12 classes.
     - `test`: 6 scenes, 726 frames, 2–5 CAVs (mean 3.33, 5/6 multi-sender), 11/12 fg classes + Free.
   - Physically separated sparse caches (`opv2v_clear`, `opv2v_lisa_fog`, `opv2v_lisa_rain`) total `1.606 GB` (cap 80 GB). `D:` free space `~636 GiB` (floor 400 GiB).
2. **All High-ROI Upgrades Implemented & Audited**:
   - Category-Geometry factorization (`E_cls(cls) + VQ_g(code)`, +208 params, 0 extra wire bytes; `+4.07 pt test / +4.46 pt val`, Building `0.0061 -> 0.2339 / 0.2567`).
   - `25×25` Zstd visibility envelope inside fixed 512 B slot (`133–190 B`, mean `161.8 B`, 0 extra wire bytes, 0 extra params).
   - Multi-budget random training (`4–32 KB`: `0.2087 @ 2 KB`, `0.3472 @ 8 KB` beating 4-camera CoHFF `0.3416` with zero ego sensors).
   - Zero-sync pure-PyTorch vectorization (`7.26×` speedup: `9.20 ms / 108.8 FPS` @ 1 sender, `30.98 ms / 32.3 FPS` @ 4 senders, `189 MB` VRAM).
3. **Full Experimental Results Verified**:
   - Real Sender (`sender_opv2v.pt`, 30 ep): `val mIoU = 0.7150` (`veh 0.9133`, `road 0.9757`).
   - Real Receivers (`outputs/real_opv2v/phase4_gate3/`, 34 models, 0.0% byte error): Gate 3 PASS on 32 KB & 16 KB (`validate` & `test`).
     - 32 KB Test (3-seed): `B3 0.3742 ± 0.0121` vs **`B4 0.4122 ± 0.0053`** vs **`B4_env25 0.4133 ± 0.0014`** (`veh 0.4848 ± 0.0031`).
     - 16 KB Test: `B3 0.3371` vs **`B4 0.3700`** (`UFE 0.0647`) vs **`B4_env25 0.3863`** (`veh 0.4898`).
   - Scene-Grouped 5-Fold CV (`outputs/real_opv2v/cv5/`, 25 disjoint scenes): **B4 wins 10/10 folds** (`32 KB: 0.3625 vs 0.3383, p=0.0430`; `16 KB: 0.3380 vs 0.3064, p=0.0023`).
   - Refreshed Mini-200 (`outputs/phase4_gate3/`, `outputs/phase5/`): `B4 0.5814` (`3-seed 0.5383 ± 0.0378`, `veh 0.6078`) vs `B3 0.3067` (`3-seed 0.3327 ± 0.0265`).
4. **Publication Artifacts Ready**:
   - 7 visually verified 300-DPI vector PDF + PNG figures in `paper/figures_pub/`.
   - Compiled Elsevier manuscript `paper/main.pdf` (0 errors, 0 Overfull hbox, 0 undefined refs).
   - Ready for user Checklist B confirmation before GitHub open-source push.
