# BorrowedSight

**Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy**

[English](#english) | [中文说明](#中文说明)

<p align="center">
  <img src="paper/figures_pub/fig1_architecture.png" width="96%" alt="BorrowedSight Architecture and Wire-Packet Layout"/>
</p>

---

## English

### Overview

**BorrowedSight** is a lightweight, pure-PyTorch cooperative 3D semantic occupancy framework built for vehicles with **zero onboard environmental sensors** (no ego camera, LiDAR, or radar), relying strictly on 6-DoF localization and bandwidth-constrained V2X packets from LiDAR-equipped neighbors.

Rather than inflating neural backbones, BorrowedSight redesigns what information enters every wire bit under compact `2.42M`-parameter (`16.71 GFLOPs`) encoder/decoder capacity—**43,939 fewer parameters and 0.90 fewer GFLOPs than equal-byte CodeFilling-Occ**:

- **Strict Zero-Ego Receiver Boundary:** Receiver inference accepts strictly `(ego_pose, timestamp, motion, serialized_packets)` and is verified by a 6-check automated leakage audit. Complete V2X link outage deterministically falls back to `100% Unknown` (`UFE = 0.0`).
- **Explicit 4-Bit Wire Semantics Unlocking Three Zero-Extra-Byte Receiver Mechanisms:** Every anchor packs losslessly into an exact **5-byte (40-bit)** record—`17b coord + 4b class ID + 6b code1 + 6b code2 + 7b conf` (`0` wasted bits)—unlocking:
  1. **Additive $3\times 64$ Category-Geometry ResidualSimVQ (`+E_cls`):** Full 6-bit codebook capacity (`K=64`, zero extra trainable parameters) across *Things* (`Vehicles, TrafficSign`), *Ground* (`Terrain, Road, SideWalk`), and *Vertical Static* (`Building, Fence, Pole, Vegetation, Wall, GuardRail, Bridge`), reaching **98.4% (63/64)** Stage-2 Vertical Static utilization and `149/192 (77.6%)` total active prototypes.
  2. **Semantic Anchor Residual Connection (`SARC`, `+13` scalar parameters, `0` extra wire bytes):** Confidence- and freshness-weighted wire class voting directly reinforcing 3D semantic logits, lifting held-out Building IoU by **`+18.69 pt`** (`0.2130 → 0.3999`, 3-seed **`0.4338`**) and overall Test mIoU by **`+2.29 pt`**.
  3. **Semantic-Selective Kinematic Forward Warping (`SS-KFW`, `0` parameters, `0` extra wire bytes):** Receiver-side 8-connected BEV clustering over wire-tagged `Vehicles` anchors estimates multi-step world velocities to warp moving traffic forward while locking static structures to exact world $\mathrm{SE}(3)$, boosting 500 ms delayed Vehicle IoU by **`3.49×`** (`0.0571 → 0.1994`, **`3.67×`** CodeFilling-Occ).
- **Loss-Resilient Micro-Blocks, Adaptive `50×50` Zstd Envelopes, and TDVF:** Self-contained 16-token Morton micro-blocks isolate wireless drops alongside a fixed 512 B Zstandard slot fitting `50×50` (`0.8 m`) visibility envelopes across **100.0%** of all 4,736 real OPV2V sweeps (`295–509 B`) and **100.0%** of all **2,306 real-world 32-beam V2V4Real sweeps** (`0.0%` overflow, Tier-2 max `475 B ≤ 510 B`). **Task-Decoupled Visibility Fusion (TDVF)** isolates visibility exclusively inside a late-fusion tri-state refinement head (`+2.92 pt` Test mIoU vs. early stem concatenation).
- **Real-Time Pure-PyTorch Deployment (Zero `spconv`/`mmcv`):** Single-GEMM zero-sync quantization (`6.73×` speedup) and vectorized packing achieve **`12.68 ms` (`78.8 FPS`)** for 1 sender and **`41.88 ms` (`23.9 FPS`)** for 4 simultaneous senders in FP32 (`13.53–43.28 ms` BF16) within **`137.8 MB`** peak VRAM on one RTX 3090.

### Key Measured Results (25-Scene Real OPV2V, Real-World V2V4Real, and Mini-200, 0.0% Byte Discrepancy)

| Benchmark (Packet Budget) | Method | Bytes / Pkt | mIoU $\uparrow$ | Fg-mIoU $\uparrow$ | Vehicle IoU $\uparrow$ | Building IoU $\uparrow$ | Unknown F1 $\uparrow$ | UFE $\downarrow$ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real OPV2V Test (32 KB, 726 frames) | B0 (No Comm) | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7755 | 0.0000 |
| Real OPV2V Test (32 KB) | B1 (Sparse FP16, 36 B) | 32,768 | 0.2650 | 0.2293 | 0.3268 | 0.0140 | 0.8868 | 0.0599 |
| Real OPV2V Test (32 KB, 3-seed) | Where2comm-Occ (20 B) | 32,768 | 0.3937 ± 0.0178 | 0.3648 ± 0.0193 | 0.4789 ± 0.0025 | 0.3162 ± 0.0390 | 0.9068 ± 0.0010 | 0.0506 ± 0.0020 |
| Real OPV2V Test (32 KB) | B2 (Cont. 8-d, 20 B) | 32,768 | 0.3997 | 0.3717 | 0.4767 | 0.3163 | 0.9077 | 0.0479 |
| Real OPV2V Test (32 KB, 3-seed) | B3-VQ (Shared 5B) | 32,768 | 0.4185 ± 0.0042 | 0.3926 ± 0.0038 | 0.4773 ± 0.0017 | 0.2606 ± 0.0842 | 0.9079 ± 0.0019 | **0.0427 ± 0.0028** |
| Real OPV2V Test (32 KB, 3-seed) | CodeFilling-Occ (5B) | 32,768 | 0.4292 ± 0.0095 | 0.4027 ± 0.0104 | 0.4783 ± 0.0057 | 0.2619 ± 0.0496 | **0.9109 ± 0.0007** | 0.0443 ± 0.0007 |
| **Real OPV2V Test (32 KB, 3-seed)** | **BorrowedSight v3 (5B+50×50)** | **32,768** | **0.4496 ± 0.0033** | **0.4252 ± 0.0034** | **0.4872 ± 0.0069** | **0.4338 ± 0.0397** | 0.9095 ± 0.0024 | 0.0474 ± 0.0015 |
| Real OPV2V Test (16 KB, 3-seed) | CodeFilling-Occ (5B) | 16,384 | 0.4010 ± 0.0068 | 0.3735 ± 0.0075 | 0.4788 ± 0.0027 | 0.2593 ± 0.0981 | 0.9051 ± 0.0029 | 0.0508 ± 0.0028 |
| Real OPV2V Test (16 KB, 3-seed) | B3-VQ (Shared 5B) | 16,384 | 0.4163 ± 0.0016 | 0.3896 ± 0.0028 | 0.4757 ± 0.0062 | 0.2956 ± 0.0388 | **0.9073 ± 0.0041** | 0.0490 ± 0.0043 |
| **Real OPV2V Test (16 KB, 3-seed)** | **BorrowedSight v3 (5B+50×50)** | **16,384** | **0.4214 ± 0.0125** | **0.3958 ± 0.0142** | **0.4914 ± 0.0030** | **0.3098 ± 0.0737** | 0.9067 ± 0.0014 | **0.0474 ± 0.0041** |
| **Real OPV2V Test (8 KB / 0.66 Mbps)** | **B4-MultiBW v3** | **8,192** | **0.3949** | -- | **0.4810** | -- | 0.9032 | 0.0506 |
| **Real OPV2V Test (2 KB / 0.16 Mbps)** | **B4-MultiBW v3** | **2,048** | **0.2886** | -- | **0.2393** | -- | **0.8976** | 0.0549 |
| **Real OPV2V 500 ms Latency (0 Extra B)** | **BorrowedSight v3 + SS-KFW** | **32,768** | **0.3718** | -- | **0.1994** (**3.49×**) | -- | 0.9061 | 0.0552 |
| **V2V4Real Zero-Shot (8 KB, 316 frames)** | **B4-MultiBW v3 (Free IoU / Unk F1 / 3D Veh)** | **8,192** | **0.4347** (Tri) | **0.5524** (Free) | **0.3632** (3D Rec) | -- | **0.7480** | 0.1469 |
| Sync. v3.0 5-Fold CV (25 Scenes, 32 KB, 5B) | B3-VQ vs. **BorrowedSight v3** | 32,768 | 0.3818 vs. **0.3869** (4/5 folds) | 0.3561 vs. **0.3613** | 0.4588 vs. **0.4636** | -- | 0.8842 vs. 0.8841 | 0.0586 vs. 0.0614 |
| **Sync. v3.0 5-Fold CV (25 Scenes, 16 KB, 5B)** | B3-VQ vs. **BorrowedSight v3** | **16,384** | 0.3632 vs. **0.3729** ($p=0.0216$) | 0.3362 vs. **0.3460** ($p=0.0196$) | 0.4535 vs. **0.4635** (5/5 folds) | -- | 0.8809 vs. **0.8830** | 0.0650 vs. 0.0657 |
| Mini-200 3-Sender (16 KB, seed 0) | B3 vs. **B4** | 16,384 | 0.3067 vs. **0.5814** | -- | 0.1549 vs. **0.6078** | -- | 0.9657 vs. **0.9725** | 0.0184 vs. 0.0222 |

### Repository Layout

```text
├── borrowedsight/      # Core model (TDVF, SARC, grouped 3x64 VQ, W2C/CF baselines), kinematics (SS-KFW), nets, 5B codec, Morton packer
├── tools/              # v3 sender/receiver training, 37-model benchmark, 5-fold CV, Phase 5 stress/mechanism/compute eval, V2V4Real audit
├── tests/              # Unit tests: coordinates, ray visibility, 40-bit K=64 parity, SARC, SS-KFW, W2C/CF, TDVF & 6-check zero-ego audits
├── configs/            # YAML configurations for grid, weather (LISA / train jitter), and training
├── experiments/        # Experiment registry (registry.csv)
├── paper/              # IEEE Transactions LaTeX manuscript (main.tex, references.bib, 9 publication figures, cover_letter.md)
├── business/           # Deployment & V2X integration notes
└── requirements.txt    # Minimal reproducible dependencies
```

### Installation & Quickstart

```bash
# 1. Create a Python 3.12 CUDA environment and install dependencies
pip install -r requirements.txt

# 2. Point BorrowedSight to your external dataset/cache directory (kept outside the repo)
export BORROWEDSIGHT_DATA_ROOT=/path/to/borrowedsight_data   # PowerShell: $env:BORROWEDSIGHT_DATA_ROOT="D:\bs_data\BorrowedSight"

# 3. Run unit tests, 40-bit K=64 TX==RX parity, SARC/SS-KFW/TDVF invariants, and 6-check Zero-Ego audits
python -m pytest -q

# 4. Run the full BorrowedSight v3.0 25-scene real OPV2V pipeline + Phase 5 + V2V4Real zero-shot + 5-fold CV
python -m tools.train_sender_v3 --epochs 35
python -m tools.extract_cache_real --sender-ckpt sender_opv2v_v3.pt --cache-prefix opv2v_v3
python -m tools.run_v3_full --epochs 30 --multibw-epochs 35 --batch-size 4 --lr 8e-4 --ufe-w 0.25
python -m tools.eval_phase5_v3
python -m tools.eval_v2v4real_zeroshot
python -m tools.run_5fold_cv_v3 --epochs 20 --batch-size 4 --lr 8e-4 --ufe-w 0.25
```

### Engineering Insights & Lessons Learned

1. **All 9/9 Orthogonal Ablations Strictly Positive:** Every architectural component—TDVF (`+2.92 pt`), $3\times 64$ grouped codebooks (`+2.55 pt`), 13-scalar SARC (`+2.29 pt`), additive class embeddings (`+1.81 pt`), category quotas (`+1.60 pt`), Warmup+EMA (`+1.02 pt`), free 6-bit $K_{\mathrm{cb}}=64$ (`+0.47 pt`), free Stage-2 residuals (`+0.31 pt`), and the 512 B $50\times 50$ visibility envelope (`+0.28 pt` clean / `+2.19 pt` LISA Fog / `+1.55 pt` 50% Drop)—delivers verified held-out test gains.
2. **SARC Rescues Thin Vertical Semantics at +13 Parameters:** Even with an 82.6%-accurate Building sender, 2D BEV U-Net smoothing collapsed receiver Building IoU to `0.2130`. Adding 13 bounded classwise logit-skip scalars (`SARC`, 25% training prior dropout) immediately lifted Building IoU by `+18.69 pt` (`0.3999`, 3-seed `0.4338`) at `0` extra wire bytes.
3. **Zero-Byte SS-KFW Overcomes 500 ms V2X Delay:** Because `c_u == Vehicles` travels explicitly on the wire, receiver-only 8-connected clustering + lag-weighted multi-step velocity estimation boosts 500 ms Vehicle IoU by **`3.49×`** (`0.0571 → 0.1994`) while keeping static buildings/roads untouched.
4. **Zero-Sync Single-GEMM Grouped Quantization (`6.73×` Speedup):** Replacing per-group Python boolean indexing (`mask.any().item()`) with a single batched matrix multiply eliminated host-device sync stalls with `0.00e+00` numerical drift.
5. **Real-World V2V4Real 32-Beam Transfer Diagnosis:** Across 2,306 real Los Angeles Velodyne VLP-32C sweeps, adaptive Tier-2 (`50×50` tri-state) fits **100.0%** inside 512 B (`max 475 B ≤ 510 B`, `0.0%` overflow). Zero-shot evaluation also reveals that relaxing the hard `road_cap=0.45` whenever total occupied voxels fall below $N(B)$ (`2,456 < 5,840`) eliminates unnecessary ground-token pruning on sparse 32-beam highways.

### Citation

```bibtex
@article{wu2026borrowedsight,
  author  = {Wu, Yonghao},
  title   = {BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy},
  journal = {Submitted to IEEE Transactions},
  year    = {2026}
}
```

---

## 中文说明

### 项目简介

**BorrowedSight** 是一个轻量级纯 PyTorch 协同三维语义占据框架，面向**零本车环境传感器**（无本车相机、无激光雷达、无毫米波雷达）的接收车辆，仅依赖本车 6-DoF 位姿、运动状态与受限带宽下的邻车 V2X 报文，即可完成接收端中心三维语义占据与盲区风险估计。

核心亮点（v3.0 IEEE Transactions 旗舰版）：

- **显式 4-bit 线缆语义锚点驱动三大「零额外字节」接收端机制**：每个锚点无冗余打包为 **5 字节（40-bit）** 紧凑记录（`17b 无损坐标 + 4b 类别 ID + 6b Stage-1 + 6b Stage-2 + 7b 置信度`），在总参数量（`2.419M` / `16.71 GFLOPs`）**低于等字节 CodeFilling-Occ（`2.462M` / `17.61 GFLOPs`）**的前提下，同步解锁：
  1. **加法式 $3\times 64$ 几何分组 ResidualSimVQ**：吃满 6-bit 免费位宽（`K=64`，0 额外可训练参数），Vertical Static 二级残差利用率高达 **98.4%（63/64）**，总活跃原型达 **149/192（77.6%）**。
  2. **语义锚点残差直连（`SARC`，仅 `+13` 标量参数、0 额外字节）**：将置信度与时效加权的线缆类别投票直接残差注入 3D 语义输出 logits，一举将测试集 Building IoU 拉升 **`+18.69 pt`**（`0.2130 → 0.3999`，3 种子 **`0.4338`**），整体 mIoU 提升 **`+2.29 pt`**。
  3. **语义选择性运动学前向补偿（`SS-KFW`，0 参数、0 额外字节）**：仅对线缆标签为 `Vehicles` 的动态锚点执行接收端 8 邻域聚类与多步速度估计前向映射，静止背景严格锁定世界 $\mathrm{SE}(3)$ 坐标，**500 ms 高延迟下车辆 IoU 暴涨 `3.49×`**（`0.0571 → 0.1994`，为 CodeFilling-Occ 的 `3.67×`）。
- **任务解耦可见性融合（TDVF）与全量 9/9 正向正交消融**：固定 512 B 槽位自适应 Zstd `50×50` 包络在 4,736 条真实 OPV2V 扫描与 **2,306 条真实 V2V4Real 32 线实车扫描**中均实现 **100.0% 无溢出封装**；TDVF 彻底消除早融合几何捷径过拟合（`+2.92 pt`），**9 项正交消融全部严格正向增益**。
- **全面刷新等字节主对比与实车零样本泛化**：25 场景真实 OPV2V 测试集 32 KB 3 种子均值达 **`0.4496 ± 0.0033` mIoU**（前景 Fg-mIoU **`0.4252`**、车辆 **`0.4872`**、建筑 **`0.4338`**），显著超越等字节复现的 `CodeFilling-Occ`（`0.4292 ± 0.0095`，`+2.04 pt` 且方差收紧 `2.9×`）、`B3-VQ`（`0.4185`，`+3.11 pt`）与 `Where2comm-Occ`（`0.3937`，`+5.59 pt`）；半带宽 16 KB 达 **`0.4214`**（车辆 **`0.4914`**、全组最低 UFE **`0.0474`**）；`B4-MultiBW v3` 在 **8 KB（0.66 Mbps）达 `0.3949` mIoU**、**2 KB 达 `0.2886`**（领先 CodeFilling `+17.73 pt`）；实车 V2V4Real 8 KB 零样本全面领跑 Free IoU（`0.5524`）、Unknown F1（`0.7480`）与 3D 车辆框召回率（`0.3632`）。单张 RTX 3090 端到端纯 PyTorch 推理达 **1 车 `12.68 ms (78.8 FPS)` / 4 车 `41.88 ms (23.9 FPS)`**，峰值显存仅 **`137.8 MB`**。

### 联系作者

Yonghao Wu（`yonghaowu369@gmail.com`），华南师范大学电子科学与工程学院（微电子学院）。
