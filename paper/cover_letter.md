# Cover Letter — Submission to IEEE Transactions

**Date:** November 26, 2026

**To:** Editor-in-Chief, IEEE Transactions

**Subject:** Manuscript Submission — *"BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled **"BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy"** for consideration for publication in IEEE Transactions.

### Motivation and Core Contributions

Cooperative Vehicle-to-Everything (V2X) perception is rapidly advancing from sparse bounding-box sharing toward dense 3D semantic occupancy reconstruction. Existing cooperative occupancy frameworks break down whenever an ego receiver possesses zero functional onboard environmental sensors (due to legacy hardware, lens icing, mud occlusion, dense fog, or sensor outage) and must reconstruct its local 3D surround purely from bandwidth-constrained, lossy, and delayed V2X transmissions. Continuous feature sharing saturates vehicular channels, monolithic vector-quantized codebooks let planar road tokens crowd out vertical structures while remaining semantically opaque on the wire, and occupied-only anchor messaging conflates unobserved blind zones with verified free space.

Our manuscript introduces **BorrowedSight**, an exact-byte, pure-PyTorch zero-ego-observation cooperative 3D semantic occupancy framework (`2.42M` parameters, `16.71 GFLOPs`—using 43,939 fewer parameters and 0.90 fewer GFLOPs than equal-byte CodeFilling-Occ):

1. **Explicit 4-Bit Wire Semantics Unlocking Three Zero-Extra-Byte Receiver Mechanisms:** Packing every anchor into an exact 5-byte (40-bit) zero-waste record (`17b coord + 4b class ID + 6b code1 + 6b code2 + 7b conf`) simultaneously unlocks: (i) additive $3\times 64$ geometry-grouped ResidualSimVQ (`98.4%` Vertical Static Stage-2 utilization, zero extra trainable parameters for $K=64$), (ii) a **13-scalar Semantic Anchor Residual Connection (SARC)** that eliminates 2D BEV decoding blur on thin vertical structures (`+18.69` pts Building IoU, `+2.29` pts Test mIoU), and (iii) zero-parameter **Semantic-Selective Kinematic Forward Warping (SS-KFW)** that warps delayed vehicle anchors while locking static structures to exact world $\mathrm{SE}(3)$, boosting 500 ms delayed vehicle IoU by **`3.49×`** (`0.0571 → 0.1994`).
2. **Adaptive `50×50` Zstd Visibility Envelopes and Task-Decoupled Visibility Fusion (TDVF):** Self-contained 16-token Morton micro-blocks isolate channel drops alongside a fixed 512-byte Zstandard slot fitting `50×50` (`0.8 m`) visibility envelopes across **100.0%** of all 4,736 real OPV2V sweeps and **100.0%** of all 2,306 real-world 32-beam V2V4Real sweeps (`0.0%` overflow). Routing backward-warped envelopes strictly into TDVF's late-fusion tri-state branch eliminates early geometric shortcut overfitting (`+2.92` pts Test mIoU) while improving clean accuracy, LISA fog (`+2.19` pts), 50% block loss (`+1.55` pts), and epistemic risk separation (`1.83×`).
3. **Equal-Byte Benchmarking, 9/9 Positive Ablations, and Real-World V2V4Real Validation:** Across 25 disjoint multi-sender real OPV2V scenes at exact `0.0%` byte discrepancy, BorrowedSight attains **`0.4496 ± 0.0033`** 3-seed Test mIoU (`0.4252` Fg-mIoU, `0.4872` vehicle IoU, `0.4338` building IoU) at 32 KB—beating equal-byte re-implemented CodeFilling-Occ (`0.4292 ± 0.0095`, `+2.04` pts with `2.9×` lower variance), compact 5-byte B3-VQ (`0.4185 ± 0.0042`, `+3.11` pts), and Where2comm-Occ (`0.3937 ± 0.0178`, `+5.59` pts). At half bandwidth (16 KB), BorrowedSight achieves **`0.4214 ± 0.0125`** mIoU, **`0.4914`** vehicle IoU, and lowest UFE (`0.0474`), synchronized 25-scene 5-fold cross-validation confirms statistically significant half-bandwidth superiority over upgraded 5-byte B3-VQ (`p = 0.0216` mIoU, `p = 0.0196` Fg-mIoU, `5/5` vehicle folds), and multi-budget training reaches `0.3949` mIoU at 8 KB (`0.66 Mbps`) and `0.2886` at 2 KB (`+17.73` pts over CodeFilling-Occ). All 9/9 orthogonal ablations confirm strictly positive held-out gains. Zero-shot evaluation on six real Los Angeles V2V4Real scenarios (2,306 physical Velodyne VLP-32C sweeps) validates 100.0% lossless `50×50` envelope delivery and leading 8 KB free-space IoU (`0.5524`) and 3D vehicle recall (`0.3632`).
4. **Real-Time Embedded Efficiency & Reproducibility:** Pure-PyTorch serialized inference runs at **`12.68 ms` (`78.8 FPS`)** for one sender and **`41.88 ms` (`23.9 FPS`)** for four simultaneous senders in FP32 (`13.53–43.28 ms` BF16) within **`137.8 MB`** peak VRAM on one RTX 3090. Complete code, exact wire codecs, six automated zero-ego leakage tests, and plotting scripts are publicly available at `https://github.com/WYH302/BorrowedSight`.

### Declarations

- This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere.
- The author declares no competing financial or personal interests.

Thank you for your time and consideration. We look forward to the peer-review feedback from IEEE Transactions.

Sincerely,

**Yonghao Wu** (Corresponding Author)  
School of Electronic Science and Engineering (School of Microelectronics)  
South China Normal University, Foshan, Guangdong, China  
Email: `yonghaowu369@gmail.com`  
Code Repository: `https://github.com/WYH302/BorrowedSight`
