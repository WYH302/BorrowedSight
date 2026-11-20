# BorrowedSight 验收报告

日期：2026-11-21。对照《BorrowedSight 验收标准》的 A 级条目。合成 Mini-200 上的协议测试和 Gate 只记为证据。需要真实 OPV2V 或语义标签的条目保持「未达到」。

项目没有验收。真实 OPV2V 不完整，没有语义标签。A 级没有全部通过。

## 已经跑过的证据

协议测试：`D:\App1\bs_cuda\Scripts\python.exe -m pytest tests/test_env.py tests/test_coords.py tests/test_rays.py tests/test_bit_pack.py tests/test_zero_ego.py -q`，24 passed。其中六项 Zero-Ego 测试在 `tests/test_zero_ego.py`。`test_cuda_3090_and_bf16` 要求设备名是 NVIDIA GeForce RTX 3090，并且要求 bf16。`test_short_sender_pads_to_one_budget_and_does_not_shift` 要求格子少于预算时报文仍能往返，并且 B1–B4 落在同一个长度上。这是接口测试，不是真实数据审计。

合成 Mini-200 在 `D:\bs_data\BorrowedSight\mini200`：`scene_00.npz` 和 `scene_01.npz`，每个场景 100 帧。`configs/mini200.yaml` 写的也是 2 个场景、各 100 帧。五张检查图在 `outputs/phase1_check/`。`outputs/phase1_check/summary.json` 里 `vehicle_hit_on_vehicle_voxel` 是 1.0，`real_scene` 是 null，`real_redraw` 是 `skipped_incomplete`。

Phase 2 是 `outputs/phase2_oracle/coverage.csv`，`coverage.json` 里 `n_frames` 是 8，没有训练。0–10 m 的 `vehicle_recall`：1 辆晴天 0.19181946，3 辆晴天 0.39492243，1 辆雾 0.15744006，3 辆雾 0.35825106。同一档 `unknown_fraction`：1 辆晴天 0.59272595，1 辆雾 0.73108813，3 辆晴天 0.43633556，3 辆雾 0.49824921。20–40 m 的 `vehicle_recall` 单元格是空的。

Phase 3：`outputs/phase3_sender/loss.json` 的 `steps` 是 200，损失序列有 200 项，第一项 2.568633794784546，最后一项 0.10381318628787994，没有 NaN，状态 `loss_dropped`。`experiments/registry.csv` 的 `sender_mini_seed0`：mIoU 0.30996499，车辆 IoU 0.43907463，道路 IoU 0.47983207。`D:\bs_data\BorrowedSight\cache\mini200_clear` 有 12 个 `shard_*.npz`，文件长度合计 31849018 字节。没有全量真实训练划分。

`outputs/phase4_gate3/summary.md` 已经和下面这组注册表行一致，这次没有重写它。判定用的是 git commit `04ee1ba` 的最后一组 `no_communication` / `loss_dropped` 行。summary 记录 Pass 为 True，byte error 为 0.0。规则原文是 mIoU fraction +0.015，或者 vehicle IoU +0.05 且 UFE 不超过 B3 的 70%，或者 mIoU 差距在 0.005 以内且字节不超过对方的 70%。seed 0 上 B4 的 UFE 是 0.026472607，B3 是 0.030232978，70% 那一条没有单独成立。B4 codebook usage 是 0.6354166666666666，road token fraction 是 0.13773090806143393。这两项在 summary 里，不在 `measurements.csv` 或注册表的列里。summary 还写明：短发送车填不满的 token 槽是填充，不是锚点。

注册表里更早的 `d0c5f19` 行还在。那一组 B1 到 B4 的字节是 15248、15152、14286.2、14156.733。那一组不作为这次 Gate 的判定。

| method | bytes | mIoU | vehicle IoU | road IoU | unknown F1 | UFE | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0 | 0 | 0 | 0 | 0.84751653 | 0 | no_communication |
| B1 | 16384 | 0.36326521 | 0.43390366 | 0.51785785 | 0.94645161 | 0.025969892 | loss_dropped |
| B2 | 16384 | 0.43361293 | 0.58892433 | 0.32930311 | 0.94667013 | 0.053487873 | loss_dropped |
| B3 | 16384 | 0.29434875 | 0.092169858 | 0.37177833 | 0.9594036 | 0.030232978 | loss_dropped |
| B4 | 16384 | 0.35221265 | 0.29825046 | 0.43330968 | 0.96020233 | 0.026472607 | loss_dropped |

同一张表里，B4 的 mIoU 和车辆 IoU 高于 B3。B1 的 mIoU 是 0.36326521，B2 是 0.43361293，都高于 B4。这是合成 Mini-200 的证据，不是项目验收。

Phase 5 的行在 `outputs/phase5/measurements.csv`。`paper/figures/` 的表和图由 `paper/figures/gen_from_csv.py` 从这张 CSV 和注册表原样抄出。`outputs/phase5/ablations.csv` 四行都是 `not_run`：`envelope`、`grouped_codebooks`、`road_quota`、`second_code_level`。`paper/figures/ablations_not_run.tex` 只记录未跑，没有结果单元格。

种子行的字节都是 16384。B3 的 mIoU 是 0.29434875、0.25720516、0.30351106。B4 的 mIoU 是 0.35221265、0.30625656、0.37235945。B3 的车辆 IoU 是 0.092169858、0.22987702、0.078970984。B4 的车辆 IoU 是 0.29825046、0.3766055、0.25591896。B3 的 UFE 是 0.030232978、0.025816911、0.021625641。B4 的 UFE 是 0.026472607、0.03855305、0.030737817。这三颗种子上，B4 的 mIoU 和车辆 IoU 都高于同一种子的 B3。UFE 只在 seed 0 上 B4 更低；seed 1 和 seed 2 上 B4 更高。

天气行的 mIoU，字节 16384。B3：clear 0.29434875，fog 0.20397822，rain 0.28008357。B4：clear 0.35221265，fog 0.26239999，rain 0.28073157。

发送车数量的 mIoU，字节 16384。B3：1 辆 0.15463547，2 辆 0.25528612，3 辆 0.29434875。B4：1 辆 0.17669809，2 辆 0.2920871，3 辆 0.35221265。

B4 的位姿 mIoU：`0.0_0.0` 是 0.35221265，`0.5_2.0` 是 0.25991842，`1.0_5.0` 是 0.19719875。B4 的延迟 mIoU：level 0 是 0.35221265，1 是 0.32037436，2 是 0.32707454，5 是 0.30103485。B4 的丢包：level 0.0 的字节 16384、mIoU 0.35221265；0.1 的字节 14747.467、mIoU 0.32927958；0.25 的字节 12430.933、mIoU 0.2928118；0.5 的字节 8439.3333、mIoU 0.20946897。

`D:\bs_data\BorrowedSight\checkpoints` 里能看到 `sender_mini.pt`、`B0_mini.pt`、`B1_mini.pt`、`B2_mini.pt`、`B3_mini.pt`、`B4_mini.pt`，以及 `B3_mini_seed1.pt`、`B4_mini_seed1.pt`、`B3_mini_seed2.pt`、`B4_mini_seed2.pt`。

## A 级

| 条目 | 结论 | 依据 |
| --- | --- | --- |
| A1 问题设定 | 未达到 | Mini-200 上有天气行和发送车数量行。真实 OPV2V 没有完整场景，也没有语义标签。 |
| A2 无泄漏 | 未达到 | 六项 Zero-Ego 测试包含在上述 24 passed 里。合成验证上 B0 的 mIoU 是 0，UFE 是 0。没有真实数据审计。 |
| A3 借用感知 | 未达到 | 合成 Gate 上 B0 的 mIoU 是 0，B4 是 0.35221265。同一预算下 B1、B2 的 mIoU 更高。真实语义标签缺失。 |
| A4 接收端三维输出 | 未达到 | 接收端按 `borrowedsight/grid.py` 的 100×100×8 网格输出语义和三态。没有在真实 OPV2V 上核对车和体素。 |
| A5 类别几何进入报文 | 未达到 | 四项消融都是 `not_run`。分组码本和可见性包络没有被单独拆开。 |
| A6 真实字节 | 未达到 | 合成 Gate 的 B1–B4 字节都是 16384，summary 的 byte error 是 0.0。真实 OPV2V 通信没有测。 |
| A7 强基线 | 未达到 | `04ee1ba` 这组 B0–B4 使用同一字节预算，只存在于合成 Mini-200。B4 的 mIoU 高于 B3，低于 B1 和 B2。 |
| A8 Unknown | 未达到 | 三态来自发送车射线。B0 的 UFE 是 0。seed 0 上 B4 的 UFE 是 0.026472607，B3 是 0.030232978，没有降到规则里的 70%。seed 1 和 seed 2 上 B4 的 UFE 高于 B3。可见性消融没有跑。 |
| A9 天气 | 未达到 | `measurements.csv` 有 clear、fog、rain。这是合成条件，不是带语义标签的真实 OPV2V。 |
| A10 拥堵与遮挡 | 未达到 | 合成集上 1、2、3 辆发送车有测量行。这不是真实拥堵数据。 |
| A11 位姿、延迟、丢包 | 未达到 | 三类扰动在合成集上有测量行。这不是真实车载网络。 |
| A12 消融 | 未达到 | `envelope`、`grouped_codebooks`、`road_quota`、`second_code_level` 都是 `not_run`。没有结果表。 |
| A13 多种子 | 未达到 | B3 和 B4 有 seed 0、1、2。mIoU 和车辆 IoU 的高低方向在这三行上保持。UFE 的高低方向在 seed 1 和 seed 2 上相反。关键消融没有重复。数据仍是合成 Mini-200。 |
| A14 故事闭环 | 未达到 | 动机、方法和实验没有在真实 OPV2V 语义日志里闭合。消融没有跑。 |

十项最终清单全部为否。

## 明确没有做的事

- 这次只补了记账，没有重新训练，没有下载数据，没有改模型、打包器或损失。
- 没有把 LiDARWeather 当测试集。
- 没有下载第二套数据集。
- 没有把合成 Gate 的 Pass 写成项目验收。
- 没有给没跑的消融填数字。
- 没有把未观测区域记成空闲。
- 没有把填充字节当成占据锚点。
