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

Rather than inflating neural backbones, BorrowedSight redesigns what information enters every wire byte under identical `2.34M`-parameter (`15.22 GFLOPs`) encoder/decoder capacity:

- **Strict Zero-Ego Receiver Boundary:** Receiver inference accepts strictly `(ego_pose, timestamp, motion, serialized_packets)` and is locked by a 6-check automated leakage audit. Total V2X link outage deterministically falls back to `100% Unknown` (`UFE = 0.0`).
- **Category-Geometry Factorized Residual SimVQ:** Decodes every 8-byte wire token as $\mathbf{E}_{\mathrm{cls}}(c_u) + \mathrm{VQ}_{g(c_u)}(q_u)$ (`+208` parameters, `0` extra wire bits) across three geometry-specialized `ResidualSimVQ(rotation_trick=True)` codebooks (*Things*, *Ground*, *Vertical Static*) with category-balanced token quotas—adding only `+1,232` total parameters (`+0.053%`) over shared VQ while boosting Building IoU by $38\times$--$42\times$.
- **Loss-Resilient Micro-Blocks & Fixed 512 B Zstd Visibility Envelopes:** Packs anchors into self-contained 16-token Morton micro-blocks paired with a backward-warped `$20\times 20$` or `$25\times 25$` Zstandard visibility envelope (`133–190 B` compressed inside the exact 512 B slot, `0` extra wire bytes) to separate *Observed Free* corridors from *Unknown* blind zones.
- **Rate-Adaptive Multi-Budget Training & Real-Time Vectorized Deployment:** Randomized `4–32 KB` training adapts one checkpoint across `2–32 KB` channels, while zero-sync pure-PyTorch vectorization achieves **`9.20 ms` (`108.8 FPS`)** for 1 sender and **`30.98 ms` (`32.3 FPS`)** for 4 simultaneous senders within `189 MB` peak VRAM on an RTX 3090 (no `spconv` or `mmcv`).

### Key Measured Results (25-Scene Multi-Sender Real OPV2V & Mini-200, 0.0% Byte Discrepancy)

| Benchmark (Packet Budget) | Method | Bytes / Pkt | mIoU $\uparrow$ | Vehicle IoU $\uparrow$ | Road IoU $\uparrow$ | Unknown F1 $\uparrow$ | UFE $\downarrow$ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real OPV2V Test (32 KB, 726 frames) | B0 (No Comm) | 0 | 0.0000 | 0.0000 | 0.0000 | 0.7755 | 0.0000 |
| Real OPV2V Test (32 KB) | B1 (Sparse FP16) | 32,768 | 0.2419 | 0.3133 | 0.6023 | 0.8864 | 0.0908 |
| Real OPV2V Test (32 KB) | B2 (Cont. 8-d) | 32,768 | 0.3611 | 0.4668 | 0.5943 | 0.9091 | 0.0713 |
| Real OPV2V Test (32 KB, 3-seed) | B3 (Shared VQ) | 32,768 | 0.3742 ± 0.0121 | 0.4655 ± 0.0011 | 0.6860 ± 0.0082 | 0.9029 ± 0.0035 | **0.0567 ± 0.0055** |
| **Real OPV2V Test (32 KB, 3-seed)** | **B4 ($20\times 20$ Env.)** | **32,768** | **0.4122 ± 0.0053** | **0.4812 ± 0.0030** | **0.6949 ± 0.0042** | **0.9119 ± 0.0043** | 0.0676 ± 0.0118 |
| **Real OPV2V Test (32 KB, 3-seed)** | **B4-Env25 ($25\times 25$ Env.)** | **32,768** | **0.4133 ± 0.0014** | **0.4848 ± 0.0031** | 0.6878 ± 0.0182 | **0.9128 ± 0.0034** | 0.0692 ± 0.0052 |
| Real OPV2V Test (16 KB) | B3 (Shared VQ) | 16,384 | 0.3371 | 0.4546 | 0.6158 | 0.9132 | 0.0794 |
| **Real OPV2V Test (16 KB)** | **B4 ($20\times 20$ Env.)** | **16,384** | **0.3700** | **0.4767** | **0.6269** | 0.9130 | **0.0647** |
| **Real OPV2V Test (16 KB)** | **B4-Env25 ($25\times 25$ Env.)** | **16,384** | **0.3863** | **0.4898** | **0.6455** | 0.9059 | 0.0897 |
| **Real OPV2V Test (8 KB / 0.66 Mbps)** | **B4-Env25-MultiBW** | **8,192** | **0.3472** | **0.4208** | **0.5864** | **0.9110** | **0.0657** |
| Scene-Grouped 5-Fold CV (25 Scenes, 32 KB) | B3 vs. **B4** | 32,768 | 0.3383 vs. **0.3625** ($p=0.0430$) | 0.4447 vs. **0.4538** | 0.6527 vs. **0.6653** | 0.8811 vs. 0.8691 | 0.0990 vs. **0.0969** |
| Scene-Grouped 5-Fold CV (25 Scenes, 16 KB) | B3 vs. **B4** | 16,384 | 0.3064 vs. **0.3380** ($p=0.0023$) | 0.4398 vs. **0.4580** | 0.5701 vs. **0.5985** | 0.8724 vs. **0.8795** | 0.1061 vs. **0.1042** |
| Mini-200 3-Sender (16 KB, seed 0) | B3 vs. **B4** | 16,384 | 0.3067 vs. **0.5814** | 0.1549 vs. **0.6078** | 0.5374 vs. **0.6407** | 0.9657 vs. **0.9725** | 0.0184 vs. 0.0222 |

### Repository Layout

```text
├── borrowedsight/      # Core model, pure-PyTorch nets, RVQ codec, vectorized Morton packer, metrics
├── tools/              # Training, sparse cache extraction, 5-fold CV, oracle, stress evaluation, audits
├── tests/              # Unit tests: coordinates, ray visibility, bit packing, and 6-check zero-ego audits
├── configs/            # YAML configurations for grid, weather (LISA / train jitter), and training
├── experiments/        # Experiment registry (registry.csv)
├── paper/              # Single-file Elsevier LaTeX manuscript (main.tex, references.bib, 300-DPI figures)
├── business/           # Deployment & V2X integration notes
└── requirements.txt    # Minimal reproducible dependencies
```

### Installation & Quickstart

```bash
# 1. Create a Python 3.12 CUDA environment and install dependencies
pip install -r requirements.txt

# 2. Point BorrowedSight to your external dataset/cache directory (kept outside the repo)
export BORROWEDSIGHT_DATA_ROOT=/path/to/borrowedsight_data   # PowerShell: $env:BORROWEDSIGHT_DATA_ROOT="D:\bs_data\BorrowedSight"

# 3. Run unit tests and automated Zero-Ego leakage verification
python -m pytest -q

# 4. Run the controlled Mini-200 benchmark
python -m tools.build_mini200
python -m tools.train_sender
python -m tools.extract_sparse_cache
python -m tools.train_receivers
python -m tools.phase5_stress

# 5. Run the full 25-scene real OPV2V + Semantic-OPV2V (4LidarSurround) pipeline + 5-fold CV
python -m tools.train_sender_real --epochs 30
python -m tools.extract_cache_real
python -m tools.train_receivers_real --epochs 15 --batch-size 4 --lr 8e-4
python -m tools.eval_phase5_real
python -m tools.run_5fold_cv_real --folds 5 --epochs 15 --batch-size 4 --lr 8e-4 --budgets 32768 16384
```

### Citation

```bibtex
@article{wu2026borrowedsight,
  author  = {Wu, Yonghao},
  title   = {BorrowedSight: Category-Geometry Token Communication and Visibility Envelopes for Zero-Ego-Observation Cooperative 3D Semantic Occupancy},
  journal = {Submitted to Vehicular Communications},
  year    = {2026}
}
```

---

## 中文说明

### 项目简介

**BorrowedSight** 是一个轻量级纯 PyTorch 协同三维语义占据框架，面向**零本车环境传感器**（无本车相机、无激光雷达、无毫米波雷达）的接收车辆，仅依赖本车 6-DoF 位姿、运动状态与受限带宽下的邻车 V2X 报文，即可完成接收端中心三维语义占据与盲区风险估计。

核心亮点：

- **严格零本车观测边界与安全回退**：接收端接口严格锁定为 `(ego_pose, timestamp, motion, serialized_packets)`，通过 6 项自动化零泄漏测试；全链路断联时确定性输出 `100% Unknown`（`UFE = 0.0`）。
- **类别-几何解耦残差量化（+208 参数、0 额外字节）**：基于 `E_cls(cls) + VQ_g(code)` 解码与 *Things / Ground / Vertical Static* 三组 `ResidualSimVQ(rotation_trick=True)` 码本，总参数量仅 **2.34M**（相比共享码本 B3 仅增 `+1,232` 参数 / `+0.053%`），彻底消除垂向结构类码本冲突，Building IoU 提升 `38×–42×`。
- **抗丢包 Morton 微块 + 固定 512 B 槽位 `25×25` Zstd 可见性包络**：16-token 独立微块保障 50% 丢包下平滑退化；`25×25` 粗包络经 Zstd 压缩仅占 `133–190 B`（0 额外字节），经精确 $\mathrm{SE}(3)$ 后向变换有效分离「观察为空」与「未观察盲区」。
- **多预算随机训练与纯 PyTorch 实时向量化**：在全量 25 场景多发送车真实 OPV2V 测试集上，32 KB 达 **`0.4133 ± 0.0014` mIoU**（车辆 `0.4848`），16 KB 达 **`0.3863` mIoU**（半带宽超越 32 KB B3），多预算模型在 **`8 KB (0.66 Mbps)` 达 `0.3472` mIoU**（零本车传感器超越 4 相机本车 CoHFF `0.3416`）；场景分组 5 折交叉验证 10/10 全胜（16 KB `p = 0.0023`）；单张 RTX 3090 端到端实测 **1 车 `9.20 ms (108.8 FPS)` / 4 车 `30.98 ms (32.3 FPS)`**，峰值显存仅 `189 MB`。

### 快速开始

1. **环境配置**：使用独立 Python 3.12 CUDA 环境安装 `requirements.txt`。
2. **数据路径设置**：通过环境变量 `BORROWEDSIGHT_DATA_ROOT` 指向外部数据与稀疏缓存目录（避免将大型缓存或下载脚本写入代码仓库）。
3. **测试与复现**：运行 `python -m pytest -q` 验证全部单元测试与 6 项零本车泄漏审计，再按英文 Quickstart 步骤复现 Mini-200 或 25 场景真实 OPV2V 全流程与 5 折交叉验证。

### 联系作者

Yonghao Wu（`yonghaowu369@gmail.com`），华南师范大学电子科学与工程学院（微电子学院）。
