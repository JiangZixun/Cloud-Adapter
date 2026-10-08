# HRC-WHU：CAFBR FMamba 训练分析与调参计划

记录日期：2026-10-08。已更新 UNet 5CE+5Dice 的完整40000-step运行结果。当前已完成三组实验：UNet CE、UNet 5CE+5Dice、CAFBR FMamba 5CE+5Dice；均为历史PolyLR运行。最新要求仅保留UNet、LS-Mamba和CAFBR FMamba的10%线性warmup＋cosine配置，其他模型已恢复原配置，以下结果不代表新调度效果；新调参实验尚未运行。

## 1. 已完成FMamba的实际运行配置（历史PolyLR）

配置来源：`experiments/HRC_WHU/FMamba_CAFBR/config.json`，优化实现见 `tools/train_unet.py`。

| 项目 | 实际设置 |
|---|---|
| 模型 | 原生 CAFBR FMamba，RGB 输入，二分类，原生解码器，无 Mask2Former |
| 参数量 | 63,767,255（environment.json 记录） |
| 数据 | 120 张训练图，30 张测试图；测试集同时用于 checkpoint 选择和最终评估 |
| 输入 | 256×256；训练从大图随机裁切，图像/标签坐标一致；测试整图 256×256 |
| 类别 | clear sky=0，cloud=1；ignore_index=255 |
| 标准化 | mean=[123.675,116.28,103.53]，std=[58.395,57.12,57.375] |
| 增强 | 水平翻转概率 0.5；亮度/对比度/饱和度各以 0.5 概率调整，系数 0.5～1.5 |
| Optimizer | PyTorch AdamW，lr=1e-4，weight_decay=0.05 |
| AdamW 默认参数 | betas=(0.9,0.999)，eps=1e-8，amsgrad=False；代码未覆盖这些默认值 |
| 参数分组 | 单一参数组，无独立 backbone/CAFBR 学习率；未设置 norm/bias 的 decay 例外 |
| 学习率 | PolyLR：lr(t)=1e-4×max(0,1−t/40000)^0.9；无 warmup |
| 训练长度 | 40,000 step；batch=4；num_workers=4；无梯度累积，无梯度裁剪 |
| 随机种子/精度 | seed=42，AMP=false，cudnn.benchmark=false |
| 损失 | 5×CE+5×soft Dice loss；dice_eps=1e-5；CE 无类别权重 |
| Dice 定义 | softmax 概率与 one-hot 标签；按 batch 和空间维求和，再对两类平均；包含 clear sky，忽略 255 |
| 验证/保存 | 每 4,000 step；mIoU 选优；保留 top 3 和 best/last；日志每 50 step |
| CAFBR 时序 | cafbr_start_ratio=0.5：step 1～20000 关闭并冻结 refiners，20001 起启用并训练 |
| SCGM | num_groups=3 |
| CAFBR 模块 | reduction=8，spatial_kernel=7，detail_kernel=3，gate_bias=0 |
| 初始缩放 | gate=1.0，detail=0.1，context=0.05，spectral=0.05 |
| Fourier | enabled=true；stage_4、stage_3、stage_2；spectral_reduction=4 |
| 硬件 | RTX 5090 D，PyTorch 2.10.0+cu128，CUDA 12.8 |

FMamba 配置虽含 base_channels=64，但 native_model 并未将该字段传给 Fmamba；不要把修改此字段当成有效的 FMamba 宽度实验。宽度调整需要先改模型构造接口。

`loss` 是加权训练目标；`ce_loss` 和 `val_loss` 是纯 CE。不同 loss 配置的原始 loss 数字不应直接比较。5CE+5Dice 同时改变了损失成分和整体梯度尺度；1CE+1Dice 与其比例一致，但固定 optimizer 下也不应假定训练轨迹完全相同。

## 2. 训练记录与结论

以下历史对比数值来自原UNet CE与FMamba两个实验的 `results.json`。新增UNet损失对照见下方第2.1节。FMamba 训练集评估将整张大图缩放到 256×256；优化阶段使用随机局部裁切，两者输入分布不同。

| step | UNet 测试 mIoU | FMamba 测试 mIoU | FMamba 训练集评估 mIoU | FMamba 测试纯 CE | FMamba 云 Recall |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 70.78 | 76.15 | 73.63 | 0.3347 | 70.44 |
| 8,000 | 83.84 | 77.60 | 75.47 | 0.2983 | 73.31 |
| 12,000 | 83.56 | 70.93 | 73.55 | 0.4226 | 62.97 |
| 16,000 | 79.23 | 76.29 | 74.89 | 0.3469 | 70.26 |
| 20,000 | 86.10 | 77.17 | 77.13 | 0.3097 | 71.84 |
| 24,000 | 81.68 | 84.01 | 81.82 | 0.2150 | 81.90 |
| 28,000 | 75.56 | 83.67 | 81.73 | 0.2208 | 81.08 |
| 32,000 | 78.40 | 73.21 | 76.96 | 0.4090 | 65.74 |
| 36,000 | 83.82 | 80.08 | 80.87 | 0.2873 | 75.77 |
| 40,000 | 79.77 | 77.76 | 78.75 | 0.3260 | 72.20 |

### 最佳结果

- UNet（纯 CE）：best=20000，mIoU=86.1040%，mDice=92.5022%，aAcc=92.9469%。
- CAFBR FMamba（5CE+5Dice）：best=24000，mIoU=84.0064%，mDice=91.2571%，aAcc=91.8912%；CAFBR 已启用。
- FMamba 比 UNet 低 2.0976 个 mIoU 百分点。FMamba 云 Precision=97.2109%、Recall=81.9027%；UNet 对应为 95.2970%、86.4827%。当前 FMamba 误报更少、漏检更多。
- best_metrics.json 是重新加载最佳权重后的最终评估。FMamba 与 24000 step 的验证值有约 0.00027 个百分点的小差异，属于这次重新执行的实测差异；汇总使用 best_metrics.json。

### “后半程训练无益”的准确范围

UNet 在 20000 达到本次采样到的最高分，随后五次验证均未刷新；最终 40000 的 mIoU=79.7674%，比最佳低 6.3366 个百分点。

FMamba 后半程一开始有明显提升：20000 为 77.1678%，24000 为 84.0061%，提升约 6.84 个百分点；28000 仍有 83.6698%。24000 之后未刷新最佳，40000 降至 77.7597%，比最终 best 低 6.2467 个百分点。不能把后半程整体称为无用，更不能把这 6.84 个百分点单独归因于 CAFBR：缺少同轨迹关闭 CAFBR 的对照。

每 4000 step 才验证一次，真实峰值可能位于这些检查点之间。“最佳 step”是已观测检查点的最佳，而非每一步都测得的结论。

### 问题假设及证据边界

1. 优化 batch 的平均 loss 持续降低：FMamba 从首个窗口 2.8844 降至最后窗口 1.0783，UNet 从 0.3769 降至 0.1351；但泛化指标没有持续改善。
2. 不能仅据此断言典型过拟合。FMamba 训练集整图评估 mIoU 在 24000/28000 约 81.8%，32000 降至 76.96%，40000 为 78.75%；训练集评估也恶化。可能涉及输入尺度差异、类别偏移、优化或归一化统计波动，需要专项实验确认。
3. 32000 step 的 FMamba 云 Recall 从 28000 的 81.08% 降至 65.74%，Precision 却升至 98.81%，说明性能下降主要表现为云漏检。40000 云 Recall 只有 72.20%。
4. 当前大图随机裁切训练、整图缩放评估的尺度差异值得优先排查。后续应增加固定训练裁块评估，避免把两个不同输入协议的 train/test loss 误当成常规泛化差距。
5. batch=4 的归一化统计、CAFBR 延迟启用、类别/场景采样都有可能影响波动；现有记录不足以区分原因。

## 2.1. 新增 UNet 5CE+5Dice 对照：已完成

数据来源：三组 `best_metrics.json`、`results.json`、`metrics.jsonl`、`config.json`、`environment.json`、`data_manifest.json`。新运行已验证到40000 step，并在最后重新加载best执行全部30张测试图评估。

### 对照条件核查

两个UNet配置仅有 `loss` 和 `work_dir` 不同；均为64基础通道、31,037,698参数、seed=42、batch=4、AdamW lr=1e-4/decay=0.05、40000-step PolyLR、4000-step验证。三组manifest的train/test图像及标签配对完全相同；环境均为RTX 5090 D、PyTorch 2.10.0+cu128。新UNet与FMamba复用同一个CE/Dice函数，训练/评估都使用对应的目标。

清单核查验证了路径配对一致，未对图像内容逐文件做哈希核验；也没有每次运行的完整代码快照。同seed不保证GPU执行逐位确定性。两版数据加载器在代码演进中新增了显式validation字段，新旧HRC流程均从test路径读取，并绕过随机训练增强；没有记录表明改用了不同测试集。

### 最佳结果与最后一步

指标单位为%，末次列是40000 step的验证mIoU；best各列来自最终重新加载评估。

| 实验 | best step | best mIoU | best mDice | best aAcc | 云 IoU | 云 Precision | 云 Recall | 末次 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UNet CE | 20,000 | 86.1040 | 92.5022 | 92.9469 | 82.9427 | 95.2970 | 86.4827 | 79.7674 |
| UNet 5CE+5Dice | 20,000 | 84.3671 | 91.4741 | 92.0679 | 80.5449 | 96.7182 | 82.8081 | 78.7905 |
| CAFBR FMamba 5CE+5Dice | 24,000 | 84.0064 | 91.2571 | 91.8912 | 80.0223 | 97.2109 | 81.9027 | 77.7597 |

- 新UNet相对纯CE：mIoU下降1.7369、mDice下降1.0280、aAcc下降0.8790个百分点；两者best均为20000。
- 相同5CE+5Dice下，UNet比CAFBR FMamba高0.3607个mIoU、0.2170个mDice百分点。补齐损失因素后，两者最佳分数接近；单seed无法确认这个小差距有统计意义。
- 原来的UNet CE与FMamba差距为2.0976个百分点；现在相同损失的差距为0.3607。差距缩小是观测事实，不能据此把差值全部做成可加的“损失贡献/架构贡献”：不同网络对损失的响应可能不同，FMamba CE对照仍未运行。

### 两版UNet完整验证曲线

| step | CE mIoU | 5CE+5Dice mIoU | 新−旧（百分点） | 新纯CE loss | 新云 Recall |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 70.78 | 69.23 | -1.56 | 0.4919 | 59.80 |
| 8,000 | 83.84 | 81.19 | -2.65 | 0.2303 | 78.51 |
| 12,000 | 83.56 | 80.43 | -3.13 | 0.2543 | 76.70 |
| 16,000 | 79.23 | 79.40 | +0.17 | 0.2860 | 74.62 |
| 20,000 | 86.10 | 84.37 | -1.74 | 0.1898 | 82.81 |
| 24,000 | 81.68 | 83.18 | +1.50 | 0.2102 | 80.95 |
| 28,000 | 75.56 | 75.26 | -0.30 | 0.3659 | 69.58 |
| 32,000 | 78.40 | 79.31 | +0.91 | 0.2893 | 74.65 |
| 36,000 | 83.82 | 82.31 | -1.51 | 0.2214 | 79.48 |
| 40,000 | 79.77 | 78.79 | -0.98 | 0.3019 | 73.83 |

新UNet的优化batch窗口平均目标从3.1233持续降低至1.1054，但测试mIoU在20000达峰，40000降到78.7905%，比best低5.5766个百分点。24000和32000时新损失比纯CE稍好，但10个验证点中8个更低；不能说Dice在每一步都降低效果。

20000之后五个验证点的平均mIoU为：UNet CE 79.8457%、UNet 5CE+5Dice 79.7727%、FMamba 79.7446%。这个描述性平均不是独立重复实验的统计结论，但说明三组都有明显后期波动，也提示仅比较best可能放大单个检查点的优势。

### 可确认的下降原因：误报减少，漏检增加更多

对照两个UNet在20000 step的最佳模型，混淆矩阵行是真实类别、列是预测类别，0=非云、1=云：

| 错误类型 | UNet CE | UNet 5CE+5Dice | 新−旧 |
|---|---:|---:|---:|
| 非云被误判为云（FP） | 33,277 | 21,908 | −11,369 |
| 云被漏判为非云（FN） | 105,393 | 134,044 | +28,651 |
| 错误像素总数 | 138,670 | 155,952 | +17,282 |

新损失的云Precision增加1.4211个百分点，Recall下降3.6747个百分点，云IoU下降2.3979个百分点。新模型预测为云的像素从707,575降至667,555，减少40,020。因此本次分数下降的直接表现是模型更保守，漏掉的云增加量超过减少的误报量。尚未逐图确认是否集中在薄云、边界或特定场景，不把这些具体区域当成已证实原因。

### 机制假设与排查优先级

1. **损失配方确实影响本次结果，但不能断言Dice本身一定有害。** 新目标同时把CE从1倍改成5倍并加入5倍Dice；现有对照无法分开整体尺度与Dice成分。Dice是batch内汇总的两类soft Dice，而mIoU是全测试集混淆矩阵的硬分类指标；优化其中一个不保证另一个单调改善。未来可用5CE+0Dice、1CE+1Dice等对照拆分；按最新优先级，此类实验暂缓，先做warmup与lr。
2. **整体loss乘5不是学习率简单乘5。** AdamW的自适应归一化会改变整体梯度缩放的作用，不应因此直接把lr除5视为等价修正。epsilon、decay与新增Dice方向也影响结果；lr需要单独测试。
3. **共同流程波动应优先排查。** 两版UNet在28000同时显著下跌（75.56%/75.26%），随后恢复；两者训练集整图评估也在28000跌至77.40%/77.10%。FMamba的主要下跌在32000，因此没有证据表明三种模型都在同一步失败。共同训练裁切/随机采样/BN统计是候选原因，尚未定位到具体环节。
4. **尺度与BN问题比单一过拟合解释更值得先查。** 训练用大图局部裁块，训练集整图评估却把1280×720等大图缩成256×256；UNet含BatchNorm，batch=4，随机裁块的云占比/纹理分布可能造成running statistics波动。这是代码与曲线支持的假设，需要固定训练裁块评估和训练数据BN重校准实验验证；不能宣称已经发现BN故障。
5. **纯CE几乎相同不代表硬分类效果相同。** best纯CE为0.188634/0.189762，差约0.001128，但argmax预测已明显偏向非云。CE衡量概率目标，mIoU衡量离散分类，二者不会一一对应。
6. **暂未发现明显loss路由或train/eval切换错误。** 代码每次优化前调用model.train()，评估用model.eval()+inference_mode；两个模型共享的loss含正确ignore处理。既有单测验证过损失与梯度。这排除了当前代码中几个简单错误，并不替代运行代码快照、逐图预测和BN的进一步诊断。

## 3. 当前调度与后续实验（UNet、LS-Mamba、CAFBR FMamba）

最新要求：仅UNet两个损失版本、CAFBR FMamba原生/Mask2Former、LS-Mamba原生/Mask2Former默认采用 **10%线性warmup＋cosine衰减**，loss先保持原配方，每2000 step验证。Cloud-Adapter、DINOv2、SAM、CLIP的7份HRC专用MMSeg配置已恢复原设置。

### 当前学习率曲线

上述三个模型的40000-step配置使用4000-step warmup：从目标lr的1%线性升至目标lr，剩余36000 step做单次cosine衰减，最低lr=0，无周期重启。以目标lr=1e-4为例：

| 已完成step | lr |
|---:|---:|
| 0 | 1e-6 |
| 2000 | 5.05e-5 |
| 4000 | 1e-4 |
| 22000 | 5e-5 |
| 40000 | 0 |

共享trainer以已完成optimizer更新数t计算：W=ceil(T×0.1)，f=0.01；t<W时lr=lr_target×[f+(1−f)t/W]；t≥W时lr=lr_min+(lr_target−lr_min)×[1+cos(π(t−W)/(T−W))]/2，每次optimizer更新后推进一次scheduler。

其他MMSeg模型恢复原PolyLR（power=0.9），无warmup。Cloud-Adapter、DINOv2、SAM保持40000 step训练，每4000 step验证、记录和保存；CLIP保持1500 step训练，每15 step验证、记录和保存。此次新增的MMSeg输出目录覆盖也已撤销。

JSON设置：`lr_schedule="cosine"`、`warmup_ratio=0.1`、`warmup_start_factor=0.01`、`min_lr_ratio=0.0`。新配置移除不生效的poly_power。共享调度器为旧checkpoint及非HRC配置保留默认poly兼容路径；并不把其他数据集默认为cosine。

### 第一阶段：warmup有无对照（若继续做该消融）

UNet、LS-Mamba和CAFBR FMamba的默认HRC配置已经有warmup；原no_warmup配置文件已移除，以免误用旧PolyLR。需要无warmup对照时，在同一cosine配置上显式覆盖 `--warmup-ratio 0`，不恢复PolyLR。两组均固定lr=1e-4、40000总长、2000验证、5CE+5Dice、CAFBR在20001启用，使用不同目录。

```bash
# 当前默认：warmup＋cosine
python tools/train_fmamba.py --config configs/fmamba/hrc_whu_native.json --wandb

# 显式消融：cosine，无warmup；总长40000不变
python tools/train_fmamba.py --config configs/fmamba/hrc_whu_native.json \
  --warmup-ratio 0 --work-dir experiments/HRC_WHU/FMamba_CAFBR_cosine_no_warmup --wandb
```

warmup可能改善初期优化，但不预先保证提升。该对照同时改变cosine阶段长度（36000或40000），应按完整调度配方解释。历史PolyLR结果仅作参考，不将其与新warmup cosine的差异全部归因于warmup。比较2000间隔新结果时，也记录4000倍数检查点的best，控制选优密度带来的差异。

### 第二阶段：lr大小

默认先在统一warmup cosine下比较目标lr **3e-5、5e-5、1e-4**；其他设置和warmup比例不变。如果先完成有无warmup消融，则使用选定的warmup设置。每组必须新目录、从头训练，loss暂不修改。

```bash
python tools/train_fmamba.py --config configs/fmamba/hrc_whu_native.json \
  --lr 5e-5 --work-dir experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr5e5 --wandb

python tools/train_fmamba.py --config configs/fmamba/hrc_whu_native.json \
  --lr 3e-5 --work-dir experiments/HRC_WHU/FMamba_CAFBR_warmup_cosine_lr3e5 --wandb
```

记录best/last mIoU、mDice、best step、云Precision/Recall、纯CE及后期波动；若最佳lr位于网格边界，再依据趋势扩展。优胜设置用seed=42、43、44复现。不要以单seed的0.36个百分点差距宣称架构显著优劣。

### 后续暂缓项与训练长度

loss倍率/比例、FMamba纯CE、CAFBR启用时机/Fourier开关、weight_decay、batch、BN/尺度诊断均作为后续候选；本轮只推进调度和lr。已有UNet曲线在28000共同下跌的线索继续保留，不能据此断言BN故障。

先保持原总训练长度。修改max_iters会同时影响10%warmup步数、cosine时长与CAFBR启用时机。若复现支持早停，再实现与调度总长独立的停止规则；当前不自动早停。

### 运行与历史实验隔离

所有JSON默认输出到`experiments/HRC_WHU/<Model>_warmup_cosine/`；`hrc_whu_native_warmup.json`作为显式预热别名使用额外的`_lr1e4_val2k`目录。已有训练启动脚本继续使用原入口，即可读取新默认配置。

旧PolyLR实验使用其保存的config.json进行评估/恢复；不能拿新默认配置直接resume旧last。共享trainer会校验调度类型、warmup、lr、最低lr比例和decay，不会静默切换；采用新曲线需从头开始。旧数据与结果文件未改写。

完整新GPU实验尚未启动。

## 4. 比较口径与验收

当前三组实验采用相同本地train/test配对和评估函数。新增UNet对照已经补齐5CE+5Dice的损失一致性；UNet与FMamba同损失差0.3607个百分点。架构、参数量和CAFBR训练时序仍不同，且只有一个seed，尚不能隔离CAFBR贡献或宣称架构显著优劣。FMamba CE对照仍未完成，按当前优先级暂缓。

原测试集同时用于选优；调参结果应标注为此协议下的开发结果。需要独立泛化结论时，再从训练数据划分固定验证集，按场景平衡、按原图隔离后裁块；30张测试图仅作最后评估。该新协议要为所有参与比较的模型统一重跑，不能与当前分数混成同一组公平排名。

每次实验记录：完整config、seed、数据manifest、CAFBR启用状态、best step、best/last mIoU、mDice、aAcc、云Precision/Recall、纯CE、训练目标。至少保留top3与last，检查后期波动是否复现。

## 5. 已有代码及运行方式

- `configs/unet/hrc_whu_ce5_dice5.json`：UNet 5CE+5Dice配置。
- `scripts/train/HRC_WHU/train_unet_ce5_dice5.sh`：独立启动器。
- `baselines/model_factory.py`：UNet含loss配置时使用与native FMamba相同的CE/Dice函数；原纯CE配置继续使用CE。
- `tests/test_native_cafbr_loss.py`：验证两种模型相同目标的损失值/梯度一致及ignore处理。

```bash
# 完整训练入口（已有seed=42实验已完成；重新运行需新目录）
bash scripts/train/HRC_WHU/train_unet_ce5_dice5.sh

# 断点续训
bash scripts/train/HRC_WHU/train_unet_ce5_dice5.sh \
  --resume experiments/HRC_WHU/UNet_CE5_Dice5/checkpoints/last.pth

# 最佳权重评估
bash scripts/train/HRC_WHU/train_unet_ce5_dice5.sh \
  --evaluate experiments/HRC_WHU/UNet_CE5_Dice5/checkpoints/best.pth
```

UNet损失对照需从头训练，不能接着旧UNet纯CE训练。本次更新已读取用户完成的40000步结果，没有启动新的完整实验。

### 初次代码交付的验证记录

- 7项相关单元测试通过，包括损失手算、UNet/FMamba损失及梯度一致性、纯CE回归、UNet前向/反向和checkpoint管理。
- 新启动脚本通过 `bash -n`；`git diff --check` 通过。
- CPU 5-step smoke通过：16基础通道、8张训练图，每次验证/最终评估均使用30张测试图，最佳模型保存与重新加载完成。该检查验证流程，不代表完整64通道GPU训练的效果。
- 新UNet配置与原配置的差异仅为 `loss` 和 `work_dir`。
- 临时结果位于 `/tmp/cloud_adapter_unet_ce5_dice5_smoke_20261008/`，未启动完整GPU实验。

## 6. 历史结果分析的核查记录

已检查三组运行均有40000-step验证及最终best评估；三组train/test路径配对一致；两个UNet配置差异仅为loss和work_dir。对比表与原始JSON逐项核对，FP/FN增减与aAcc变化一致。本次只更新调参文档，未更改损失、模型或现有实验数据，也未启动新训练。训练代码在工作区另有已有修改，本文对当前代码的检查不能冒充历史运行的完整版本证明。

## 7. 前一次PolyLR warmup支持的历史修改记录

新增 `baselines/lr_schedule.py`；共享trainer增加线性warmup、lr覆盖参数和resume调度一致性保护。新增W0/W1两份原生CAFBR配置；两者仅warmup比例与输出目录不同，均2000-step验证、40000-step总长、5CE+5Dice。`configs/fmamba/hrc_whu_native.json`的默认验证间隔改为2000；其他数据集和已保存实验不修改。

10项相关单测通过：原无warmup曲线不变、10%warmup边界、scheduler恢复后lr一致、参数校验及CAFBR时序/损失回归。完整GPU调参尚未启动。

共享trainer的CPU 5-step warmup冒烟流程通过，包含优化、验证、checkpoint保存/加载及全部30张测试图最终评估；短smoke按向上取整使用1步warmup，不代表4000步GPU实验的性能。CLI入口、两份配置控制变量和git diff格式检查通过。

## 8. 本次统一warmup cosine的修改与验证

保留7份JSON训练配置的warmup cosine修改；7份MMSeg HRC专用配置已恢复原版本；删除已被替代的no_warmup JSON，显式消融可用CLI。共享trainer和HRC基准计时工具使用同一个调度实现，实验追踪增加lr_schedule/warmup/min_lr字段，resume保护调度变化。说明与运行目录同步更新。

17项相关测试通过，覆盖cosine的起点/峰值/中点/终点、单调性、无重启、最低lr比例、断点lr恢复、旧poly兼容、调度变更拒绝、7份JSON的warmup cosine与7份MMSeg的原PolyLR配置、损失/CAFBR/UNet回归。恢复后的MMSeg配置通过Python解析与原调度/验证间隔检查；当前qwen3环境没有MMEngine，未执行MMSeg模型训练。

共享UNet训练器的CPU 5-step warmup cosine冒烟流程通过：训练/验证、保存及重新加载best、全部30张最终测试图评估成功。完整GPU训练尚未启动。MMSeg新增输出目录已撤销，恢复后的7份配置与Git原版本逐字节一致。


## 9. 移除训练集指标评估

HRC的UNet、LS-Mamba和CAFBR FMamba共享训练器已删除额外的训练集整图评估，以及训练batch的预测转换、混淆矩阵和分割指标计算。训练仅计算优化loss并反向传播，记录loss、lr和CAFBR状态；每2000 step的验证、best选择与最终测试保留。所有7份HRC JSON标记`train_loss_only: true`，计时工具也不再统计训练集评估。Cloud-Adapter、DINOv2、SAM、CLIP原MMSeg配置未发现额外训练集指标评估，无需修改。

历史结果与上述历史训练集指标保留，不再为新运行生成训练集指标。已有checkpoint允许恢复；新的日志窗口只记录loss。加速幅度需实际计时，当前不作数值承诺。
