# CloudSEN12_High_L2A 实验记录

更新日期：2026-10-10。指标单位为%。论文数据保留原表两位小数，本地实际实验单独记录。

论文名称：`CloudSEN12_High_L2A`；本地数据/实验目录简称：`CloudSEN12_L2A`。

## Cloud-Adapter论文结果（Table XI）

来源：[论文HTML Table XI](https://arxiv.org/html/2411.13127v2#S4.T11)、[论文PDF arXiv:2411.13127v2](https://arxiv.org/pdf/2411.13127v2)。已核对模型名称、列顺序与原表数值，保留论文顺序；此表共有9个模型。论文表未报告逐模型best step，统一标为“—”。这些是论文报告结果，不是本地复现实验。2026-10-10 重新从论文原始 HTML 提取并逐项核对了9个模型、144个数值（4项总体 + 12项逐类别）。

| ID | 模型（论文） | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | Best step |
|---|---|---:|---:|---:|---:|---:|
| P01 | SCNN | 67.14 | 28.76 | 40.50 | 36.48 | — |
| P02 | KappaMask | 79.60 | 45.28 | 55.47 | 53.56 | — |
| P03 | MCDNet | 75.67 | 46.52 | 58.54 | 59.59 | — |
| P04 | CDNetv1 | 84.74 | 62.39 | 73.50 | 75.15 | — |
| P05 | DBNet | 86.82 | 65.65 | 75.52 | 77.81 | — |
| P06 | CDNetv2 | 86.88 | 66.05 | 75.91 | 78.19 | — |
| P07 | HRCloudNet | 88.35 | 68.35 | 77.35 | 79.85 | — |
| P08 | UNetMobv2 | 88.96 | 70.36 | 79.57 | 81.45 | — |
| P09 | Cloud-Adapter | 89.90 | 73.38 | 83.93 | 83.87 | — |

### 逐类别结果核对

论文列顺序为 **CRS（晴空）、TNC（薄云）、TKC（厚云）、CDS（云影）**，与本地配置的“晴空、厚云、薄云、云影”顺序不同，使用类别名称对齐。以下三表保留论文的全部逐类别数据；原表00.00统一显示为0.00，数值不变。

#### IoU（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 62.20 | 0.00 | 52.74 | 0.08 |
| KappaMask | 79.58 | 23.35 | 78.20 | 0.00 |
| MCDNet | 72.46 | 14.96 | 67.04 | 31.62 |
| CDNetv1 | 82.58 | 38.61 | 78.72 | 49.64 |
| DBNet | 85.42 | 42.76 | 80.80 | 53.62 |
| CDNetv2 | 85.35 | 44.01 | 80.85 | 54.00 |
| HRCloudNet | 87.24 | 44.18 | 82.78 | 59.20 |
| UNetMobv2 | 87.76 | 47.38 | 84.16 | 62.13 |
| Cloud-Adapter | 89.04 | 54.97 | 84.91 | 64.60 |

#### Acc（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 85.73 | 0.00 | 76.19 | 0.08 |
| KappaMask | 97.24 | 38.05 | 86.58 | 0.00 |
| MCDNet | 85.76 | 21.09 | 85.29 | 42.01 |
| CDNetv1 | 92.73 | 52.65 | 88.12 | 60.50 |
| DBNet | 94.83 | 53.53 | 90.27 | 63.45 |
| CDNetv2 | 95.65 | 56.37 | 88.06 | 63.56 |
| HRCloudNet | 95.41 | 51.00 | 93.17 | 69.83 |
| UNetMobv2 | 95.98 | 59.60 | 90.94 | 71.75 |
| Cloud-Adapter | 94.65 | 73.64 | 90.61 | 76.81 |

#### Dice（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 76.70 | 0.00 | 69.06 | 0.16 |
| KappaMask | 88.63 | 37.85 | 87.77 | 0.00 |
| MCDNet | 84.03 | 26.02 | 80.27 | 48.04 |
| CDNetv1 | 90.46 | 55.71 | 88.09 | 66.35 |
| DBNet | 92.14 | 59.90 | 89.38 | 69.81 |
| CDNetv2 | 92.09 | 61.12 | 89.41 | 70.13 |
| HRCloudNet | 93.19 | 61.28 | 90.58 | 74.37 |
| UNetMobv2 | 93.48 | 64.29 | 91.40 | 76.64 |
| Cloud-Adapter | 94.20 | 70.95 | 91.84 | 78.49 |

已检查四类别指标的算术均值与原表mIoU、mAcc、mDice，差异均在两位小数的舍入容差内。aAcc使用原表总体像素准确率，不按类别平均。IoU、Acc、Dice逐项引用论文，不互相推算。

### 对照说明

L2A 的本地正式双卡运行已经开始，尚无完整 best 结果。当前本地使用 RGB 512×512、原生 FMamba_CAFBR，并在 test split 上选优；论文数据仅作报告值参考。训练批次 loss 不属于分割测试指标，smoke 和计时测试不作为正式成绩。

## 本地完整实验结果

当前正式目录尚无 `best_metrics.json`，因此 best step 和全部正式测试指标保留为“—”。论文的全部9个模型及逐类别数据已列在上文，待本地训练结束再追加实际 best 行。

| ID | 模型 | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | Best step | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| L2A-01 | FMamba_CAFBR（原生，DDP） | — | — | — | — | — | 正式运行尚未完成 |

## 当前运行快照（L2A-01）

读取的最后训练记录：**2026-10-10 13:12:28（北京时间）**，step **900 / 40,000**。此步数是当时已写入日志的窗口终点，并非保证实时进度。当前验证记录 0 条、最终测试记录 0 条；首次验证在第 2,000 步。CAFBR 将从第 20,001 步启用。

来源：[results.json](../experiments/CloudSEN12_L2A/FMamba_CAFBR/results.json) · [训练日志](../experiments/CloudSEN12_L2A/FMamba_CAFBR/train.log)。

W&B：[本次 L2A 正式运行](https://wandb.ai/jiangzixun12134-zhejiang-university-of-technology/Cloud-Adapter/runs/4c2d4915)。

## 本次正式运行配置（L2A-01）

下表引用该运行保存的 `config.json` 和 `environment.json`，不根据仓库当前默认配置反推。

| 项目 | 实际设置 |
|---|---|
| 模型 / 输出头 | FMamba_CAFBR / 原生 FMamba decoder + classifier，无 Mask2Former |
| 输入 / 类别 | RGB，512×512；clear、thick cloud、thin cloud、cloud shadow |
| 数据量 | train：8,490 张；选优/最终 test：975 张；原 val 不参与 |
| GPU / 并行 | 2 × NVIDIA RTX A6000；DDP / NCCL |
| Batch | 每卡 1 × 累积 2 × 2 卡 = 全局有效 batch 4 |
| 优化器 | AdamW，peak lr=0.0001，weight decay=0.05 |
| LR schedule | 前 4,000 步线性 warmup（1e-6 → 1e-4）；之后 cosine 降至 0 |
| Loss | 5 × CE + 5 × multiclass soft Dice，eps=1e-5，ignore_index=255 |
| 总步数 / 验证间隔 | 40,000 / 2,000；单位均为 optimizer update |
| CAFBR | 第 1–20,000 步关闭并冻结；第 20,001 步起启用；Fourier 分支为 stage_4 / stage_3 / stage_2 |
| 增强 / 归一化 | 训练随机裁剪、水平翻转、亮度/对比度/饱和度扰动；RGB mean=[123.675,116.28,103.53]，std=[58.395,57.12,57.375] |
| Seed / 精度 / workers | 42 / FP32（AMP 关闭）/ 每 rank 4 workers |
| Checkpoint 选优 | 在全部 975 张 test 图像上按 mIoU 降序保留 top-3 / best / last |
| 环境 | Python 3.11.7；PyTorch 2.1.2+cu118；CUDA runtime 11.8 |

`smoke_val_limit` / `smoke_test_limit` 是配置中的备用 smoke 参数；本次正式运行没有 `train_limit` / `val_limit` / `test_limit`，实际使用全部 train/test 数据。训练仅计算优化 loss，未计算全训练集分割指标。BatchNorm 仍按每卡物理 batch 1 统计。

来源：[config.json](../experiments/CloudSEN12_L2A/FMamba_CAFBR/config.json) · [environment.json](../experiments/CloudSEN12_L2A/FMamba_CAFBR/environment.json) · [data_manifest.json](../experiments/CloudSEN12_L2A/FMamba_CAFBR/data_manifest.json)。

## 后续追加规则

完成后从正式目录的 best_metrics.json 追加实际 best step、全部总体与逐类别指标，并核对 top_k.json 和 results.json 的最终 best 测试记录。每次记录注明验证/测试分组；保留不同运行的实际配置，不使用训练批次指标或 smoke/benchmark 替代正式 best。
