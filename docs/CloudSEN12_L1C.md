# CloudSEN12_L1C 实验记录

更新日期：2026-10-09。只记录实际运行；指标单位为%。

## 完整实验结果

当前未发现该数据集的完整训练`best_metrics.json`，暂无可填写的最终结果。计时测试和数据预览不作为正式实验成绩。

| ID | 模型 | aAcc ↑ | mIoU ↑ | mAcc ↑ | mDice ↑ | Best step | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | 暂无完整结果 |

## 已有未完成运行

| ID | 模型 | 最后已记录训练step | Best step | aAcc / mIoU / mAcc / mDice | 状态 |
|---|---|---:|---|---|---|
| L1C-01 | FMamba_CAFBR | 1,050 | 无 | 无验证/测试结果 | 未完成；日志未到首次验证 |

| ID | 目标lr | LR schedule | Loss构成 | Batch / 输入 | 总step / 验证间隔 | 验证/测试分组 | CAFBR启用step |
|---|---:|---|---|---|---|---|---:|
| L1C-01 | 0.0001 | 无warmup；PolyLR，power=0.9，至0 | 5CE + 5Dice | 1 / 512×512 | 40,000 / 4,000 | val / test | 20,001 |

优化器AdamW，weight decay=0.05，seed=42，AMP关闭。此历史运行包含训练集指标评估；1050是最后记录的训练窗口step，不代表确定的停止step。未生成best checkpoint评估，不把训练batch的mIoU作为测试结果。

来源：[config.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/config.json) · [results.json](../experiments/CloudSEN12_L1C/FMamba_CAFBR/results.json)。当前默认配置已经改动，以上只描述这次实际运行。

## 后续追加规则

新运行逐条追加，使用运行目录保存的config.json及best_metrics.json；标明验证/测试分组。未完成运行单列，不用默认配置或benchmark推断最终结果。
