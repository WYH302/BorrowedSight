# Cover Letter — Submission to *Computer Communications* (Elsevier)

**Date:** November 25, 2026

**To:** Editor-in-Chief, *Computer Communications* (Elsevier)

**Subject:** Manuscript Submission — *"BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled **"BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy"** for consideration for publication in ***Computer Communications***.

### Relevance to *Computer Communications*

Cooperative Vehicle-to-Everything (V2X) perception is rapidly evolving from sparse bounding-box exchange toward dense 3D semantic occupancy reconstruction. Existing cooperative occupancy pipelines break down whenever an ego receiver lacks usable onboard perception sensors (due to legacy hardware, lens fouling, or adverse weather) and must rely purely on bandwidth-constrained, lossy V2X wireless links. Transmitting continuous feature maps saturates vehicular channels, monolithic vector-quantization codebooks suffer inter-class collapse under tight byte budgets, and occupied-only token streams conflate unobserved blind zones with verified free space.

Our manuscript addresses this vehicular communication and protocol-design bottleneck through **BorrowedSight**, an equalized-byte, loss-resilient V2X semantic occupancy wire protocol implemented in pure PyTorch (`2.34M–2.42M` parameters, `15.22–16.70 GFLOPs`):

1. **Zero-Waste 5-Byte Additive Category-Geometry VQ Coding:** Each 40-bit token packs a lossless 17-bit 3D voxel coordinate, 4-bit class anchor, two 6-bit geometry-group ResidualSimVQ codes (making Stage-2 residual refinement free in packet bytes), and 7-bit confidence with zero wasted bits (`+208` class-embedding parameters, `+1,232` total parameters / `+0.053%` over compact 4-byte shared VQ for `B4-Compact`).
2. **Loss-Resilient 16-Token Morton Micro-Blocks & Adaptive `50×50` Zstd Visibility Slot with TDVF:** Self-contained micro-blocks isolate wireless packet drops strictly to dropped blocks while a fixed 512-byte Zstandard slot adaptively compresses `50×50` (`0.8 m`) visibility bit-planes plus `25×25` max-height priors across **100%** of all 4,736 real OPV2V sweeps (`295–509 B`, mean `433.5 B`, zero extra wire bytes). On the receiver side, **Task-Decoupled Visibility Fusion (TDVF)** isolates backward-warped visibility envelopes inside a late-fusion tri-state refinement branch, eliminating geometric shortcut overfitting (`+1.39` mIoU points and `-20.6%` UFE relative to naive shared-stem envelope fusion).
3. **Rigorous Equal-Byte Evaluation & Statistical Verification:** Across 25 disjoint multi-sender real OPV2V + Semantic-OPV2V scenes (2,383 frames, 2–5 CAVs) at exact `0.0%` byte discrepancy, BorrowedSight attains **`0.4255 ± 0.0065`** held-out test mIoU (`B4-Compact`) and **`0.4168 ± 0.0043`** mIoU with top vehicle IoU (`0.4865 ± 0.0024`) and lowest blind-zone error (`UFE 0.0364 ± 0.0047`, `B4-TDVF`) at `32 KB` (`2.62 Mbps`)—surpassing compact 4-byte shared VQ (`0.4044 ± 0.0080`) and published 4-camera ego-equipped VQSOP (`0.4154`) with zero ego sensors. At half bandwidth (`16 KB`), `B4-TDVF` achieves **`0.4067 ± 0.0093`** mIoU and **`0.4861 ± 0.0053`** vehicle IoU—beating full-bandwidth `32 KB` shared VQ at half the wireless rate—while multi-budget training reaches **`0.3702`** mIoU at `8 KB` (`0.66 Mbps`) and `0.3250` at `4 KB` (`0.33 Mbps`). Scene-grouped 5-fold cross-validation across all 25 disjoint scenes confirms **10/10** statistically significant paired wins (`p = 0.0023` at `16 KB`, `p = 0.0430` at `32 KB`).
4. **Real-Time Embedded Efficiency & Reproducibility:** Zero-synchronization pure-PyTorch serialization and decoding achieve **`10.57–11.90 ms` (`84.0–94.6 FPS`)** for one sender and **`34.53–39.19 ms` (`25.5–29.0 FPS`)** for four simultaneous senders within **`170.5 MB`** peak VRAM on a single RTX 3090, with full code and six automated zero-ego leakage tests openly released at `https://github.com/WYH302/BorrowedSight`.

### Declarations & Publishing Model

- This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere.
- The author declares no competing financial or personal interests.
- We opt for the **traditional subscription publishing model (non-Open Access)**.

Thank you for your time and consideration. We look forward to the peer-review feedback from *Computer Communications*.

Sincerely,

**Yonghao Wu** (Corresponding Author)  
School of Electronic Science and Engineering (School of Microelectronics)  
South China Normal University, Foshan, Guangdong, China  
Email: `yonghaowu369@gmail.com`  
Code Repository: `https://github.com/WYH302/BorrowedSight`
