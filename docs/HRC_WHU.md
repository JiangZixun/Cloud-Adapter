# HRC_WHU 实验记录

更新日期：2026-10-09。指标单位为%。论文结果保留原表两位小数，本地结果显示四位小数，完整精度见原始JSON。P01–P10为论文报告结果，H01–H06为本地实验。

## 论文与本地实验结果

论文部分完整收录Cloud-Adapter论文的HRC_WHU Table VII，共10个模型，顺序与原表一致。来源：[论文Table VII（arXiv v2）](https://arxiv.org/html/2411.13127v2#S4.T7)。论文表没有提供各模型最佳step、末轮指标和CAFBR状态，使用“—”表示。

本地六组均训练至40,000 step。本地行来自各运行`best_metrics.json`：按验证mIoU选择best，然后加载best在全部30张测试图上评估。当前验证与最终测试均使用原test分组，不能视作独立测试集上的无偏泛化评估。

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

Mask2Former分类项含no-object类别权重0.1，辅助decoder/encoder输出参与loss；不能将其总loss数值与原生分割头loss直接比较。LSMamba的Dice使用平方概率分母，即便权重相同也不同于UNet/FMamba的Dice实现。

## 当前观察

- 新CAFBR FMamba的best mIoU为87.3218%，比原Poly实验84.0064%高3.3154个百分点。warmup和衰减曲线同时改变，验证间隔也从4000改为2000，无法单独归因于warmup。
- Mask2Former的best在10000 step，CAFBR尚未启用；最终best结果不是CAFBR开启后的效果。
- 三组新实验的末轮均低于best；LSMamba末轮与best差距较小。比较部署效果使用best，不能混用last。
- UNet目前只有旧Poly完整结果，尚无本地warmup cosine完整运行结果。

## 未完成/归档运行

| 运行目录 | 状态 | 目标lr / schedule | Loss | 最佳step与指标 |
|---|---|---|---|---|
| `FMamba_CAFBR_incomplete_20261008T090417Z_005f76` | 已归档；没有训练窗口、验证或最终测试记录 | 0.0001；无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 无；不纳入完整结果表 |

## 后续追加规则

每次新运行增加独立ID，不覆盖旧行。完整结果以保存的运行config、best_metrics和results为依据；记录实际best step及验证协议。未完成运行单列，训练batch指标和计时测试不填入最终结果表。


## 下一轮已配置实验（尚未运行）

以H04为基线，仅修改目标lr与输出目录。共同设置：AdamW、weight decay=0.05、batch=4、seed=42、5CE+5Dice、40000 step、每2000 step验证、10%线性warmup后cosine至0、CAFBR从20001 step启用、训练不计算分割指标。warmup起始lr仍为目标lr的1%。

| 待记录ID | 目标lr | warmup起始lr | 启动脚本 | 输出目录 | Best step / 指标 |
|---|---:|---:|---|---|---|
| H07 | 5e-5 | 5e-7 | [train_fmamba_cafbr_lr5e-5.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_lr5e-5.sh) | `experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e-5` | 待运行 |
| H08 | 3e-5 | 3e-7 | [train_fmamba_cafbr_lr3e-5.sh](../scripts/train/HRC_WHU/train_fmamba_cafbr_lr3e-5.sh) | `experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e-5` | 待运行 |

`start_train.sh`已按H07→H08顺序加入两组实验，已有完成实验保持注释。脚本通过`--lr`和`--work-dir`覆盖基线JSON；实际配置会保存到各运行目录的`config.json`。
