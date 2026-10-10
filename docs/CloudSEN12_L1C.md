# CloudSEN12_High_L1C 实验记录

更新日期：2026-10-10。指标单位为%。论文数据保留原表两位小数，本地实际实验单独记录。

论文名称：`CloudSEN12_High_L1C`；本地数据/实验目录简称：`CloudSEN12_L1C`。

## Cloud-Adapter论文结果与本地 best 对照（Table X）

来源：[论文HTML Table X](https://arxiv.org/html/2411.13127v2#S4.T10)、[论文PDF arXiv:2411.13127v2](https://arxiv.org/pdf/2411.13127v2)。已核对模型名称、列顺序与原表数值，保留论文顺序；论文原表共有9个模型。论文表未报告逐模型best step，统一标为“—”。P01–P09 是论文报告结果；L1C-02 是本次本地正式训练的 best 结果。2026-10-10 重新从论文原始 HTML 提取并逐项核对了9个模型、144个数值（4项总体 + 12项逐类别）。

| ID | 模型 / 来源 | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | Best step |
|---|---|---:|---:|---:|---:|---:|
| P01 | SCNN | 60.19 | 22.75 | 33.69 | 30.88 | — |
| P02 | KappaMask | 76.27 | 41.27 | 50.76 | 49.33 | — |
| P03 | MCDNet | 72.68 | 44.80 | 59.99 | 58.08 | — |
| P04 | CDNetv1 | 83.48 | 60.35 | 72.46 | 73.30 | — |
| P05 | DBNet | 86.83 | 65.52 | 75.15 | 77.68 | — |
| P06 | CDNetv2 | 86.68 | 65.60 | 75.65 | 77.83 | — |
| P07 | HRCloudNet | 87.86 | 68.26 | 78.13 | 79.88 | — |
| P08 | UNetMobv2 | 89.52 | 71.65 | 81.05 | 82.47 | — |
| P09 | Cloud-Adapter | 90.19 | 74.18 | 84.79 | 84.46 | — |
| L1C-02 | FMamba_CAFBR（本地双卡 DDP） | 57.51 | 33.61 | 54.14 | 47.74 | 26,000 |

### 逐类别结果核对

论文列顺序为 **CRS（晴空）、TNC（薄云）、TKC（厚云）、CDS（云影）**，与本地配置的“晴空、厚云、薄云、云影”顺序不同，使用类别名称对齐。以下三表保留论文的全部逐类别数据；原表00.00统一显示为0.00，数值不变。

#### IoU（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 56.39 | 0.00 | 34.60 | 0.00 |
| KappaMask | 76.84 | 14.75 | 73.51 | 0.00 |
| MCDNet | 68.39 | 13.02 | 65.26 | 32.52 |
| CDNetv1 | 80.48 | 36.50 | 79.60 | 44.83 |
| DBNet | 85.20 | 43.02 | 81.40 | 52.44 |
| CDNetv2 | 84.98 | 43.37 | 80.67 | 53.40 |
| HRCloudNet | 86.01 | 45.88 | 83.54 | 57.61 |
| UNetMobv2 | 88.51 | 50.12 | 84.79 | 63.17 |
| Cloud-Adapter | 89.19 | 56.15 | 85.46 | 65.93 |

#### Acc（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 89.69 | 0.00 | 45.08 | 0.00 |
| KappaMask | 97.54 | 27.78 | 77.73 | 0.00 |
| MCDNet | 77.85 | 18.39 | 84.49 | 59.20 |
| CDNetv1 | 90.60 | 50.19 | 88.34 | 60.72 |
| DBNet | 95.56 | 53.15 | 89.40 | 62.47 |
| CDNetv2 | 94.80 | 53.13 | 89.32 | 65.34 |
| HRCloudNet | 94.49 | 59.47 | 91.39 | 67.19 |
| UNetMobv2 | 95.77 | 61.87 | 91.44 | 75.13 |
| Cloud-Adapter | 94.08 | 75.23 | 91.72 | 78.15 |

#### Dice（%）

| 模型 | CRS 晴空 | TNC 薄云 | TKC 厚云 | CDS 云影 |
|---|---:|---:|---:|---:|
| SCNN | 72.11 | 0.00 | 51.41 | 0.00 |
| KappaMask | 86.91 | 25.70 | 84.73 | 0.00 |
| MCDNet | 81.23 | 23.04 | 78.98 | 49.08 |
| CDNetv1 | 89.19 | 53.48 | 88.64 | 61.91 |
| DBNet | 92.01 | 60.16 | 89.75 | 68.80 |
| CDNetv2 | 91.88 | 60.50 | 89.30 | 69.62 |
| HRCloudNet | 92.48 | 62.90 | 91.03 | 73.10 |
| UNetMobv2 | 93.90 | 66.78 | 91.77 | 77.43 |
| Cloud-Adapter | 94.29 | 71.91 | 92.16 | 79.47 |

已检查四类别指标的算术均值与原表mIoU、mAcc、mDice，差异均在两位小数的舍入容差内。aAcc使用原表总体像素准确率，不按类别平均。IoU、Acc、Dice逐项引用论文，不互相推算。

### 对照说明

本地 L1C-02 已完成 40,000 步；best 为第 26,000 步，最终使用该 best 在全部 975 张 test 图像上重新评估。与论文 Cloud-Adapter 报告值相比，本地 mIoU 为 33.61%，低 40.57 个百分点。此差值仅对照报告数值，不能解释为控制变量复现结论：本地模型为从头训练的原生 FMamba_CAFBR，且将 test 同时用于 checkpoint 选优与最终评估，未获得独立留出的最终泛化成绩。

## 本地完整实验结果（L1C-02）

正式运行完成：**2026-10-10 12:46:23（北京时间）**。训练完成步数为 **40,000**，best step 为 **26,000**，CAFBR 已启用。下表来自 `best_metrics.json`，并已与 `results.json` 的最终 best 测试及 `top_k.json` 排名交叉核验；不是末步 last checkpoint，也不是 smoke/benchmark 的结果。

| 模型 | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | mFscore ↑ | mPrecision ↑ | mRecall ↑ | Best step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FMamba_CAFBR（原生，DDP） | 57.51 | 33.61 | 54.14 | 47.74 | 47.74 | 48.73 | 54.14 | 26,000 |

完整优化目标 loss（5CE+5Dice）：**13.958998**；按有效像素汇总的 CE loss：**2.113480**。评估样本：975 张；验证选优 split=`test`，最终评估 split=`test`。

### Best 逐类别指标（%）

按论文类别顺序对齐，避免将本地厚云/薄云的索引顺序混淆。

| 类别 | IoU ↑ | Acc / Recall ↑ | Dice ↑ | Fscore ↑ | Precision ↑ |
|---|---:|---:|---:|---:|---:|
| CRS 晴空 | 45.07 | 48.34 | 62.14 | 62.14 | 86.97 |
| TNC 薄云 | 10.30 | 18.73 | 18.67 | 18.67 | 18.61 |
| TKC 厚云 | 54.14 | 83.22 | 70.25 | 70.25 | 60.77 |
| CDS 云影 | 24.94 | 66.29 | 39.92 | 39.92 | 28.56 |

### Best 混淆矩阵（像素数）

行是真实类别，列是预测类别；使用本地配置顺序：晴空、厚云、薄云、云影。

| 真实 \ 预测 | 晴空 | 厚云 | 薄云 | 云影 |
|---|---:|---:|---:|---:|
| 晴空 | 64,902,674 | 22,909,454 | 16,640,537 | 29,821,243 |
| 厚云 | 2,793,237 | 61,673,202 | 942,833 | 8,703,033 |
| 薄云 | 3,657,890 | 13,138,712 | 4,286,682 | 1,804,170 |
| 云影 | 3,269,191 | 3,760,433 | 1,166,918 | 16,120,191 |

### Checkpoint 排名

| 排名 | Step | 选优 mIoU（%） | 保存文件 |
|---|---:|---:|---|
| 1 | 26,000 | 33.61 | `iter_0026000.pth` |
| 2 | 32,000 | 32.47 | `iter_0032000.pth` |
| 3 | 18,000 | 32.47 | `iter_0018000.pth` |

最终第 40,000 步的 last checkpoint mIoU 为 **30.30%**，未替代 best。

来源：[best_metrics.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/best_metrics.json) · [best.pth](../experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/best.pth) · [top_k.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/top_k.json) · [results.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/results.json) · [训练日志](../experiments/CloudSEN12_L1C/FMamba_CAFBR/train.log)。

Best checkpoint SHA256（最终评估历史保存值）：`4e4e993d1dbc0cc17eb7a060afe99888f3c554dc0b60794c4ff9e879b0dc512c`。

W&B：[本次 L1C 正式运行](https://wandb.ai/jiangzixun12134-zhejiang-university-of-technology/Cloud-Adapter/runs/c6ee31b0)。

## 本次正式运行配置（L1C-02）

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

来源：[config.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/config.json) · [environment.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/environment.json) · [data_manifest.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/data_manifest.json)。

## 历史未完成记录（保留旧文档快照）

| ID | 模型 | 最后已记录训练step | Best step | aAcc / mIoU / mAcc / mDice | 状态 |
|---|---|---:|---|---|---|
| L1C-01 | FMamba_CAFBR | 1,050 | 无 | 无验证/测试结果 | 未完成；日志未到首次验证 |

| ID | 目标lr | LR schedule | Loss构成 | Batch / 输入 | 总step / 验证间隔 | 验证/测试分组 | CAFBR启用step |
|---|---:|---|---|---|---|---|---:|
| L1C-01 | 0.0001 | 无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 1 / 512×512 | 40,000 / 4,000 | val / test | 20,001 |

优化器AdamW，weight decay=0.05，seed=42，AMP关闭。此历史运行包含训练集指标评估；1050是最后记录的训练窗口step，不代表确定的停止step。未生成best checkpoint评估，不把训练batch的mIoU作为测试结果。

以上为 2026-10-09 旧文档中的 L1C-01 记录。当前主机的同名正式目录已保存 L1C-02 的双卡运行，不能再将其 `config.json` / `results.json` 用作上述历史配置的证据；当前主机未找到该历史运行的独立原始目录。

## 后续追加规则

新运行逐条追加，使用运行目录保存的 config.json 和 best_metrics.json；标明验证/测试分组与 checkpoint 选优规则。未完成运行单列，不用默认配置、smoke 或 benchmark 推断最终成绩。
