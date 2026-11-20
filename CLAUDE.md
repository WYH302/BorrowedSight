# BorrowedSight 项目宪章

本文件锁死架构、输入和停止条件。改架构必须同时改本文件和 `.cursor/rules/borrowedsight-constitution.mdc`。论文包装在 `paper/STORY.md`。没有真实训练日志之前，任何文件都不得写入 mIoU、UFE 或其它实验结果。

故事只留一根轴：没有本车环境观测时，邻车消息是唯一证据；没传到的体素不等于安全空闲。

## 机器和路径

- 训练在 Windows 上原生进行。解释器只用 `D:\App1\bs_cuda\Scripts\python.exe`（Python 3.12，创建时不加 `--system-site-packages`）。
- `D:\App1\environment` 是 CPU 版 PyTorch。禁止往里面装包、改文件或拿它训练。
- 禁止导出、注销或写入 WSL（`OCCAM-Ubuntu`、`Astra-Ubuntu-22.04` 都不碰）。数据不要放进 WSL。
- 代码在本仓库。数据和缓存只放 `D:\bs_data\BorrowedSight`。仓库内禁止建立 `cache/`。
- 活跃数据硬顶 350GB。以后的稀疏特征缓存单独控制在 80GB 以内，并与 LISA 测试缓存分开。
- `refs/` 里的仓库只供阅读。禁止安装它们的环境，禁止编译自定义 CUDA。

## 纯 PyTorch 发送端

- 约 8 万体素用 `scatter` 体素化，再接高度折叠的 2D 残差 UNet。最粗层只用 PyTorch 自带注意力。
- 禁止安装或导入 spconv、MinkowskiEngine、mmcv，禁止编译自定义 CUDA 算子。
- `torch.compile` 能开就开。Windows 上失败就退回 eager，不为此停工。
- 网格和类名以 `refs/CoHFF` 的官方配置为准，禁止另写一套。`cohff_opv2v/config/train_config/task_config.yml` 写的是 `vox_range: [20, 20, 0.9, -20, -20, -2.5]`，网格 `h: 100`、`w: 100`、`z: 8`。这就是约 40m×40m、水平 0.4m、100×100×8 的那套范围；z 向用文件里的数，不要改成自拟的 3.2m。
- `cohff_opv2v/config/label_config/opv2v_label_name.yaml` 里 0 是 Unlabeled，1–12 才是十二类：Building、Fence、Terrain、Pole、Road、SideWalk、Vegetation、Vehicles、Wall、GuardRail、TrafficSign、Bridge。预测头是这十二类。Unlabeled 不进预测头，未知体素用 `ignore_index`。几何码本仍只有 Things、Ground、Vertical Static 三组，不按类各做一个码本。类到码本组的对应以后再定，不能加第四组。

## 量化和打包

- 量化只用 `vector-quantize-pytorch` 的 `ResidualSimVQ(rotation_trick=True)`。
- 预测头保持 Semantic-OPV2V 的 12 类。几何码本只按 Things、Ground、Vertical Static 分三组，不是 12 个码本。
- 低带宽发第 1 级，高带宽加第 2 级。
- 通信是 16 个 token 一块的 Morton 微块（丢一块不影响别的块），再加约 0.5KB 的粗可见性包络。只发占据锚点无法区分「看见的空」和「没看见」。
- 隐式射线雕刻可以保留，只作为辅助，不能代替这层包络。

## 三态标签

三态只由发送车的 OPV2V 激光射线生成，不由模型预测生成：

- 0 Unknown
- 1 Observed Free
- 2 Occupied

射线击中前是 Observed Free，首次击中是 Occupied，遮挡后、量程外、没有射线覆盖的地方是 Unknown。多辆发送车取直接观测的并集。Unknown 体素的语义损失用 `ignore_index`。Semantic-OPV2V 里因为没扫到而填 0 的格子，不当成空闲。没有点，不是 free。

语义类别来自 Semantic-OPV2V 的 4LidarSurround 标签。4LidarSurround 是标签，不是网络输入。发送端只读邻车的原始 OPV2V `.pcd`。

## Zero-Ego

接收端只能吃：`ego_pose`、时间、车辆状态、邻车消息。

禁止作为接收端输入：本车相机、本车激光雷达、本车雷达、本车深度、由本车环境传感器算出的特征或占据、带有本车环境信息的缓存、测试标签、未来帧、地图先验。EGO 不得出现在 sender 列表里。

六项泄漏测试（Phase 1 必须落地，现在就锁定）：

1. 把 EGO 原始点云换成随机噪声，输出必须完全不变。
2. 移除 EGO 图像目录，训练和前向仍然能跑。
3. `forward` 的参数里不允许出现本车环境特征。
4. 每帧的 sender id 不得包含 receiver id。
5. 关闭全部 sender 后，输出只能是先验或 Unknown，不能维持高性能。
6. 可视化出来的输入点云只有邻车点云变换到 EGO 坐标后的位置。

坐标变换是 `p_E = inv(T_E^W) · T_S^W · p_S`。坐标没对上之前不训练。

## 基线

B0 到 B4 实例化同一套骨干、解码器和打包器。换的是消息内容，不是模型族。禁止给 B4 单独换一个更大的骨干，也禁止把 CoHFF 或 VQSOP 的相机骨干接进来。

- B0：无通信。打包结果丢掉。
- B1：同一字节预算下的稀疏特征上界。
- B2：同预算连续特征。
- B3：共享码本。只借用 VQSOP 的 SAVQ 机制，不跑它的相机骨干。
- B4：本方法（三组几何码本、微块、粗可见性包络）。

Phase 4 先在 32KB/帧的同一真实字节上训练，seed 0。

## 天气

- 训练期只允许 LiDARWeather 那种选择性抖动和丢点。LiDARWeather 不是天气测试集。
- 测试天气只用 LISA 的物理模型（Beer-Lambert / 雨滴后向散射）。测试参数与训练增强分开。Clear 与 LISA 测试缓存分开存。
- 恶劣天气不是方法贡献的名字。方法贡献是证据边界和字节里的几何与可见性。

## 明确不做

- 3D 高斯溅射，扩散补全。
- 第二数据集。Gate 3 之前只有 OPV2V 激光雷达和 Semantic-OPV2V 4LidarSurround 标签。不做 V2V4Real、DAIR-V2X、TUMTraf、Co3SOP。
- 强化学习选点。
- 把全局 Delta 编码和丢包实验绑在一起。
- 在本仓库里下载数据集。OPV2V 下载不由本宪章触发。

## 停止规则

这些是判据，不是已经测到的结果。

Phase 4 里 B4 至少满足一条才允许扩实验：mIoU 比 B3 高约 1.5 个点；或车辆类明显更好且 Unobserved-as-Free 明显更低；或同样精度少用约 30% 带宽。出现下面任何一条就停下，并且只改一个因素：码本利用率低于 40%，Road token 超过 60%，字节差超过 2%，loss 变成 NaN。

Phase 3 先在 Mini-200 上确认 loss 下降且显存低于 18GB，再谈全量。车和体素对不齐就停，不训练。

## 每天只看三样

1. `experiments/registry.csv` 的最后一行。没有数据行就是还没有实验。
2. `outputs/` 里最新的 PNG。
3. `D:\App1\bs_cuda\Scripts\python.exe -m pytest` 是否全绿。环境检查是 `tests/test_env.py`。
