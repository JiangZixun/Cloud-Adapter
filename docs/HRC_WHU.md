# HRC_WHU 实验记录

更新日期：2026-10-09。指标单位为%。论文结果保留原表两位小数，本地结果显示四位小数，完整精度见原始JSON。P01–P10为论文报告结果，H01–H08为本地实验。

## 论文与本地实验结果

论文部分完整收录Cloud-Adapter论文的HRC_WHU Table VII，共10个模型，顺序与原表一致。来源：[论文Table VII（arXiv v2）](https://arxiv.org/html/2411.13127v2#S4.T7)。论文表没有提供各模型最佳step、末轮指标和CAFBR状态，使用“—”表示。

本地八组均训练至40,000 step。本地行来自各运行`best_metrics.json`：按验证mIoU选择best，然后加载best在全部30张测试图上评估。当前验证与最终测试均使用原test分组，不能视作独立测试集上的无偏泛化评估。

| ID | 模型/实验 | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | Best step | 40k末轮mIoU | best时CAFBR |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P01 | MCDNet（论文） | 75.14 | 53.50 | 68.91 | 67.96 | — | — | — |
| P02 | SCNN（论文） | 74.51 | 57.22 | 81.27 | 72.31 | — | — | — |
| P03 | KappaMask（论文） | 84.73 | 67.48 | 80.30 | 79.74 | — | — | — |
| P04 | CDNetv2（论文） | 89.71 | 76.75 | 87.46 | 86.46 | — | — | — |
| P05 | DBNet（论文） | 90.11 | 77.78 | 88.80 | 87.17 | — | — | — |
| P06 | CDNetv1（论文） | 89.88 | 77.79 | 89.93 | 87.20 | — | — | — |
| P07 | RSAM-Seg（论文） | 92.07 | 80.90 | 88.72 | 89.16 | — | — | — |
| P08 | UNetMobv2（论文） | 92.13 | 79.91 | 85.61 | 88.45 | — | — | — |
| P09 | HRCloudNet（论文） | 92.93 | 83.44 | 92.39 | 90.79 | — | — | — |
| P10 | Cloud-Adapter（论文） | 94.50 | 89.05 | 93.74 | 94.19 | — | — | — |
| H01 | UNet | 92.9469 | 86.1040 | 91.8389 | 92.5022 | 20,000 | 79.7674 | — |
| H02 | UNet（5CE+5Dice） | 92.0679 | 84.3671 | 90.4807 | 91.4741 | 20,000 | 78.7905 | — |
| H03 | FMamba_CAFBR | 91.8912 | 84.0064 | 90.1792 | 91.2571 | 24,000 | 77.7597 | 开启 |
| H04 | FMamba_CAFBR | 93.6275 | 87.3218 | 92.4239 | 93.2050 | 24,000 | 79.3365 | 开启 |
| H05 | FMamba_CAFBR_Mask2Former | 93.1499 | 86.5910 | 92.4821 | 92.7895 | 10,000 | 81.7124 | 关闭 |
| H06 | LSMamba | 93.1351 | 86.4235 | 91.9484 | 92.6868 | 28,000 | 83.8553 | — |
| H07 | FMamba_CAFBR（lr=5e-5） | 93.7501 | 87.5453 | 92.5363 | 93.3331 | 24,000 | 76.9354 | 开启 |
| H08 | FMamba_CAFBR（lr=3e-5） | 93.8926 | 87.8875 | 92.9830 | 93.5314 | 24,000 | 72.2446 | 开启 |

论文与本地实验的具体训练、预处理和选优设置可能不同；表格用于对照报告值，不代表严格控制变量的公平排名。论文UNetMobv2也不是本地UNet实现。

## 每次运行的精简配置

下表读取每次运行保存的`config.json`，不使用当前默认配置倒推历史。共同设置：AdamW、weight decay=0.05、seed=42、batch size=4、输入256×256、AMP关闭、总40,000 step、best按mIoU选取。训练增强为随机裁切、50%水平翻转及独立触发的亮度/对比度/饱和度变化。

| ID | 目标lr | LR schedule | Loss构成 | 验证间隔 | CAFBR启用step | 训练集指标评估 | 原始配置与结果 |
|---|---:|---|---|---:|---:|---|---|
| H01 | 0.0001 | 无warmup；PolyLR，power=0.9，至0 | CE | 4,000 | — | 开启 | [config.json](../experiments/HRC_WHU/UNet/config.json) · [best_metrics.json](../experiments/HRC_WHU/UNet/best_metrics.json) · [results.json](../experiments/HRC_WHU/UNet/results.json) |
| H02 | 0.0001 | 无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 4,000 | — | 开启 | [config.json](../experiments/HRC_WHU/UNet_CE5_Dice5/config.json) · [best_metrics.json](../experiments/HRC_WHU/UNet_CE5_Dice5/best_metrics.json) · [results.json](../experiments/HRC_WHU/UNet_CE5_Dice5/results.json) |
| H03 | 0.0001 | 无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 4,000 | 20001 | 开启 | [config.json](../experiments/HRC_WHU/FMamba_CAFBR/config.json) · [best_metrics.json](../experiments/HRC_WHU/FMamba_CAFBR/best_metrics.json) · [results.json](../experiments/HRC_WHU/FMamba_CAFBR/results.json) |
| H04 | 0.0001 | 线性warmup 4,000 step（目标lr的1%起步）→ cosine至0 | 5CE + 5Dice | 2,000 | 20001 | 关闭 | [config.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/config.json) · [best_metrics.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/best_metrics.json) · [results.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine/results.json) |
| H05 | 0.0001 | 线性warmup 4,000 step（目标lr的1%起步）→ cosine至0 | 2×分类CE + 5×mask BCE + 5×mask Dice；含辅助层loss | 2,000 | 20001 | 关闭 | [config.json](../experiments/HRC_WHU/FMamba_CAFBR_Mask2Former_warmup_cosine/config.json) · [best_metrics.json](../experiments/HRC_WHU/FMamba_CAFBR_Mask2Former_warmup_cosine/best_metrics.json) · [results.json](../experiments/HRC_WHU/FMamba_CAFBR_Mask2Former_warmup_cosine/results.json) |
| H06 | 0.0001 | 线性warmup 4,000 step（目标lr的1%起步）→ cosine至0 | 5CE + 5Dice（平方概率Dice） | 2,000 | — | 关闭 | [config.json](../experiments/HRC_WHU/LSMamba_warmup_cosine/config.json) · [best_metrics.json](../experiments/HRC_WHU/LSMamba_warmup_cosine/best_metrics.json) · [results.json](../experiments/HRC_WHU/LSMamba_warmup_cosine/results.json) |
| H07 | 0.00005 | 线性warmup 4,000 step（目标lr的1%起步）→ cosine至0 | 5CE + 5Dice | 2,000 | 20001 | 关闭 | [config.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e-5/config.json) · [best_metrics.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e-5/best_metrics.json) · [results.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e-5/results.json) |
| H08 | 0.00003 | 线性warmup 4,000 step（目标lr的1%起步）→ cosine至0 | 5CE + 5Dice | 2,000 | 20001 | 关闭 | [config.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5/config.json) · [best_metrics.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5/best_metrics.json) · [results.json](../experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5/results.json) |

Mask2Former分类项含no-object类别权重0.1，辅助decoder/encoder输出参与loss；不能将其总loss数值与原生分割头loss直接比较。LSMamba的Dice使用平方概率分母，即便权重相同也不同于UNet/FMamba的Dice实现。

## 当前观察

- 新CAFBR FMamba的best mIoU为87.3218%，比原Poly实验84.0064%高3.3154个百分点。warmup和衰减曲线同时改变，验证间隔也从4000改为2000，无法单独归因于warmup。
- Mask2Former的best在10000 step，CAFBR尚未启用；最终best结果不是CAFBR开启后的效果。
- 三组新实验的末轮均低于best；LSMamba末轮与best差距较小。比较部署效果使用best，不能混用last。
- UNet目前只有旧Poly完整结果，尚无本地warmup cosine完整运行结果。
- lr调整实验H07、H08的best mIoU分别为87.5453%、87.8875%，相对H04提高0.2235、0.5657个百分点；H08比H07高0.3422个百分点。三组best均在24000 step，CAFBR已开启。当前仅seed=42，不能据此宣称差异具有统计显著性。
- 降低lr没有改善本次末轮表现：H07、H08的40000-step mIoU分别为76.9354%、72.2446%，比各自best低10.6099、15.6429个百分点。

## 未完成/归档运行

| 运行目录 | 状态 | 目标lr / schedule | Loss | 最佳step与指标 |
|---|---|---|---|---|
| `FMamba_CAFBR_incomplete_20261008T090417Z_005f76` | 已归档；没有训练窗口、验证或最终测试记录 | 0.0001；无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 无；不纳入完整结果表 |

## 后续追加规则

每次新运行增加独立ID，不覆盖旧行。完整结果以保存的运行config、best_metrics和results为依据；记录实际best step及验证协议。未完成运行单列，训练batch指标和计时测试不填入最终结果表。


## lr调整实验（H07、H08已完成）

已核对H04、H07、H08保存的`config.json`，差异仅为目标lr与输出目录，三组`data_manifest.json`相同。共同设置：AdamW、weight decay=0.05、batch=4、seed=42、5CE+5Dice、40000 step、每2000 step验证、10%线性warmup后cosine至0、CAFBR从20001 step启用、训练不计算分割指标。warmup起始lr仍为目标lr的1%。两组均有20次验证及重新加载best后的30张测试图最终评估。

| ID | 目标lr | warmup起始lr | 启动脚本 | 输出目录 | Best step / mIoU |
|---|---:|---:|---|---|---|
| H07 | 5e-5 | 5e-7 | [train_fmamba_cafbr_lr5e-5.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_lr5e-5.sh) | `experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e-5` | 24,000 / 87.5453% |
| H08 | 3e-5 | 3e-7 | [train_fmamba_cafbr_lr3e-5.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_lr3e-5.sh) | `experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5` | 24,000 / 87.8875% |

以下best指标来自各运行的`best_metrics.json`，末轮来自`results.json`中40000-step验证。分割指标单位为%，CE与加权目标loss为原始数值，显示六位小数；加权目标为5CE+5Dice。

| ID | 云 IoU | 云 Precision | 云 Recall | best纯CE | best加权目标loss | 40k末轮mIoU |
|---|---:|---:|---:|---:|---:|---:|
| H04 | 84.3496 | 97.0046 | 86.6053 | 0.171541 | 1.303025 | 79.3365 |
| H07 | 84.6137 | 97.2742 | 86.6687 | 0.166490 | 1.293460 | 76.9354 |
| H08 | 85.1899 | 95.6939 | 88.5857 | 0.160733 | 1.246207 | 72.2446 |

## 下一轮调参实验（H09–H12，尚未完整运行）

以H08的目标lr=3e-5为基线。共同设置：AdamW、weight decay=0.05、batch=4、seed=42、输入256×256、AMP关闭、5CE+5Dice、每2000 step验证、训练仅记录loss。四组均从头训练，使用独立输出目录；`keep_top_k=1`，仅保留一个排名checkpoint及同一权重的`best.pth`副本，另保留`last.pth`供断点续训。既有运行的权重不改动。

| 待记录ID | 总step | CAFBR启用step | LR schedule / warmup | 启动脚本 | 输出目录 |
|---|---:|---:|---|---|---|
| H09 | 30,000 | 15,001（50%之后） | 单次warmup 3,000 step → cosine至0 | [train_fmamba_cafbr_30k.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_30k.sh) | `experiments/HRC_WHU/FMamba_CAFBR_lr3e-5_30k` |
| H10 | 40,000 | 20,001（50%之后） | 前后20k各自warmup 2,000 step → cosine至0 | [train_fmamba_cafbr_40k_two_phase.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_40k_two_phase.sh) | `experiments/HRC_WHU/FMamba_CAFBR_lr3e-5_40k_two_phase` |
| H11 | 30,000 | 7,501（25%之后） | 单次warmup 3,000 step → cosine至0 | [train_fmamba_cafbr_30k_cafbr25.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_30k_cafbr25.sh) | `experiments/HRC_WHU/FMamba_CAFBR_lr3e-5_30k_cafbr25` |
| H12 | 30,000 | 22,501（75%之后） | 单次warmup 3,000 step → cosine至0 | [train_fmamba_cafbr_30k_cafbr75.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_30k_cafbr75.sh) | `experiments/HRC_WHU/FMamba_CAFBR_lr3e-5_30k_cafbr75` |

对应配置位于[`configs/fmamba/tuning/`](../configs/fmamba/tuning/)，四个JSON分别为`hrc_whu_30k.json`、`hrc_whu_40k_two_phase.json`、`hrc_whu_30k_cafbr25.json`、`hrc_whu_30k_cafbr75.json`。`start_train.sh`按H09→H10→H11→H12顺序运行，已完成的lr实验改为注释。

H10使用`lr_schedule="cosine_cafbr_phase"`，每段以本段长度的10%作为warmup，从目标lr的1%（3e-7）升至3e-5，再cosine衰减至0。完成第20000次optimizer更新后，scheduler重置到3e-7，供CAFBR首次启用的第20001步使用；不重置模型、AdamW状态或随机种子。前后阶段连续训练，后半程更新整个模型及CAFBR。此调度的resume禁止改变总step、CAFBR比例或学习率配方。

H09/H11/H12把总长改为30k，因此warmup也缩至3k，cosine跨度随之改变；H09的CAFBR也从H08的20001步提前至15001步。它与H08的差异包含这些因素，不能作为只改变停止时间的早停对照。H11/H12与H09的配置差异仅为CAFBR启用比例及输出目录。

### 完整运行耗时估算

依据H07、H08在RTX 5090 D上的实际日志：每组40k约3.34小时；CAFBR关闭阶段约0.281秒/step，开启阶段约0.320秒/step（按相邻50-step日志窗口的实际墙钟时间统计，含周期验证/保存开销）。同一GPU、batch=4、AMP关闭且无额外负载时估算：

| ID | 预计耗时 |
|---|---:|
| H09 | 约2.5小时 |
| H10 | 约3.3小时 |
| H11 | 约2.6小时 |
| H12 | 约2.4小时 |
| 四组串行合计 | 约10.9小时；建议预留11～12小时 |

top-1减少部分权重写入，但耗时估算不预先扣除未经实测的节省量；完整实验结果完成后再追加到主结果表。

### 脚本与调度验证

- 20项测试通过，覆盖新调度前后阶段的warmup/cosine边界、实际optimizer更新所用lr、跨CAFBR边界的scheduler恢复、配置变更拒绝，以及已有CAFBR、loss和checkpoint回归。
- 四个启动脚本均完成RTX 5090 D上的20-step CUDA smoke：8张训练图，batch=4，每5步在全部30张测试图上验证，最后重载best再次评估30张图。CAFBR实际启用步分别为11、11、6、16；逐步核对学习率和开启状态，检查仅一个排名权重及best/last存在。每组约18秒，smoke结果不作为正式实验成绩。
- Smoke产物：`/tmp/cloud_adapter_cafbr_tuning_smoke_20261009_1gbqfua3/`，四个子目录分别为`30k`、`40k_two_phase`、`30k_cafbr25`、`30k_cafbr75`，同名`.log`位于根目录。
- `bash -n`和`git diff --check`通过；正式配置与H08的共同训练条件、H11/H12相对H09的控制变量及独立输出目录已核对。尚未启动四组完整训练。
