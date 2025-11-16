# Task 1: 61类农作物病害分类训练文档

## 目录

- [概述](#概述)
- [核心功能](#核心功能)
- [关键组件详解](#关键组件详解)
- [训练策略](#训练策略)
- [命令行参数](#命令行参数)
- [使用示例](#使用示例)
- [训练流程](#训练流程)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 概述

Task 1 是一个**61类农作物病害图像分类任务**，旨在从农作物叶片图像中识别病害类型。该训练脚本 `task1train.py` 实现了一套完整的深度学习训练流程，包含多种先进的训练技术和策略，用于处理长尾分布、类别不平衡等真实场景问题。

### 主要特性

- **多模型架构支持**：baseline、multitask、fewshot 三种模式
- **高级数据增强**：Mixup、CutMix、渐进式分辨率调整
- **长尾分布优化**：加权采样、Focal Loss、类别权重
- **训练稳定性增强**：EMA（指数移动平均）、标签平滑、梯度裁剪
- **多阶段训练策略**：分阶段解冻、自适应学习率调整
- **特征学习增强**：Cosine Classifier、Center Loss
- **跨平台支持**：CUDA、MPS（Apple Silicon）、CPU

---

## 核心功能

### 1. 模型架构

#### **CosineClassifier（余弦分类器）**

```python
class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0)
```

**功能**：使用余弦相似度代替传统的点积进行分类，提高特征向量的判别性。

**原理**：
- 将特征向量和权重向量进行 L2 归一化
- 计算归一化后的余弦相似度
- 通过缩放因子（scale）调整输出范围

**优势**：
- 特征和权重在单位超球面上，消除范数影响
- 更好的类间分离和类内聚合
- 适用于长尾分布和小样本场景

**应用时机**：在训练后期（默认第35轮）激活，用于边界优化

---

#### **CenterLoss（中心损失）**

```python
class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int)
```

**功能**：约束同类样本的特征向量向类中心聚拢，增强类内紧凑性。

**原理**：
- 为每个类别维护一个可学习的特征中心
- 计算样本特征与其类别中心的欧氏距离
- 最小化该距离作为辅助损失

**损失函数**：
```
L_center = (1/N) * Σ ||f_i - c_{y_i}||²
```
其中 `f_i` 是特征，`c_{y_i}` 是对应类别的中心

**优势**：
- 提高特征的判别性
- 配合 Softmax Loss 使用效果更好
- 有助于减少类内方差

**权重系数**：默认 0.01，可通过 `--center-loss-weight` 调整

---

### 2. 数据采样策略

#### **build_weighted_sampler（加权采样器）**

```python
def build_weighted_sampler(metadata_df, label_col: str = "label_61", power: float = 0.5)
```

**功能**：构建加权随机采样器，缓解类别不平衡问题。

**采样权重计算**：
```
weight_i = 1 / (count_i)^power
```

**参数说明**：
- `power=0.5`：使用平方根倒数频率（inverse sqrt frequency）
- `power=1.0`：完全倒数频率（可能过度补偿）
- `power=0.0`：均匀采样

**策略特点**：
- 使用 `power=0.5` 在平衡性和稳定性之间取得折中
- 避免极端长尾类别被过度采样
- 配合 `--balance-sampler` 参数启用

**禁用时机**：在第15轮（默认）自动切换回自然分布，避免过拟合

---

### 3. 模型正则化

#### **ModelEMA（指数移动平均）**

```python
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999)
```

**功能**：维护模型参数的指数移动平均副本，提供更稳定的验证性能。

**更新公式**：
```
shadow_param = decay * shadow_param + (1 - decay) * current_param
```

**工作流程**：
1. 训练时：每个 batch 后更新 shadow 参数
2. 验证时：临时将 shadow 参数复制到模型
3. 恢复时：将原始参数复制回模型

**优势**：
- 减少模型权重的随机波动
- 提供更稳定的验证指标
- 类似于 Polyak 平均的效果

**典型 decay 值**：
- 0.999：标准配置（推荐）
- 0.9999：更平滑但更新慢
- 0.995：更激进的更新

---

### 4. 数据增强技术

#### **apply_mixup_cutmix（混合增强）**

```python
def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
```

**功能**：随机应用 Mixup 或 CutMix 数据增强，提高模型泛化能力。

##### **Mixup 策略**

**原理**：线性混合两张图像及其标签
```
x_mixed = λ * x_i + (1 - λ) * x_j
y_mixed = λ * y_i + (1 - λ) * y_j
```

**λ 采样**：从 Beta(α, α) 分布采样，α 由 `mixup_alpha` 指定

**效果**：
- 提供虚拟训练样本
- 正则化决策边界
- 减少对标签噪声的敏感度

##### **CutMix 策略**

**原理**：将一张图像的矩形区域替换为另一张图像的对应区域

**裁剪比例**：
```
cut_ratio = sqrt(1 - λ)
λ = 1 - (cut_area / image_area)
```

**效果**：
- 强制模型关注目标的局部特征
- 更好地学习目标的空间分布
- 保持图像的真实性

##### **动态衰减策略**

训练过程中逐步降低增强强度：

| 阶段 | Epoch | Mixup Alpha | CutMix Alpha |
|------|-------|-------------|--------------|
| 初始 | 0-9   | 0.4         | 0.6          |
| 衰减1 | 10-11 | 0.2         | 0.6          |
| 衰减2 | 12-14 | 0.1         | 0.6          |
| 禁用 | 15+   | 0.0         | 0.0          |

**设计理念**：早期增强泛化，后期精细调优

---

## 关键组件详解

### 1. 自定义训练循环

#### **custom_train_loop 函数**

这是一个高度定制化的训练循环，实现了多种高级训练策略。

##### **核心参数**

```python
def custom_train_loop(
    model: nn.Module,
    optimizer,
    scheduler,
    criterion,
    device,
    train_loader,
    val_loader,
    epochs: int,
    use_amp: bool,
    multi_task: bool,
    # 数据增强
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
    # 模型优化
    ema: Optional[ModelEMA] = None,
    # 阶段调度
    stage2_epoch: int = 10,
    stage3_epoch: int = 20,
    # ... 更多参数
)
```

##### **adaptive_loss（自适应损失）**

**功能**：根据训练阶段动态切换损失函数。

**策略**：
1. **前期（< tail_focal_epoch）**：使用标准交叉熵损失
2. **后期（≥ tail_focal_epoch）**：
   - 长尾类别（底部25%频率）使用 Focal Loss
   - 其他类别使用交叉熵损失

**Focal Loss 公式**：
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- γ=1.5：聚焦困难样本
- 自动识别长尾类别

**优势**：
- 针对性优化不同类别
- 避免头部类别主导训练
- 提升长尾类别准确率

##### **run_val（验证函数）**

**功能**：执行验证并计算多维度指标。

**评估指标**：

1. **Overall Accuracy（总体准确率）**
   ```
   Acc = (正确预测数 / 总样本数) × 100%
   ```

2. **Tail Mean Accuracy（长尾类别平均准确率）**
   ```
   TailAcc = (Σ 长尾类别准确率 / 长尾类别数) × 100%
   ```
   - 长尾类别：频率最低的25%类别
   - 专门评估模型对罕见类别的性能

3. **Macro-F1（宏平均F1分数）**
   ```
   Precision_c = TP_c / (TP_c + FP_c)
   Recall_c = TP_c / (TP_c + FN_c)
   F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
   Macro-F1 = (Σ F1_c) / C
   ```
   - 对每个类别给予相同权重
   - 更公平地评估不平衡数据集

**输出示例**：
```
[Val] Epoch 25 | Loss 0.3247 | Acc 91.23% | TailAcc 78.45% | MacroF1 0.887
```

---

### 2. 多阶段训练策略

训练过程分为多个阶段，每个阶段有不同的优化目标和策略。

#### **阶段划分**

```
Epoch:  0───────10──────20──────30──────40──────50
Stage:  [Stage1][Stage2][Stage3──────────────────>
Freeze: [Head  ][Part  ][Backbone Unfrozen       ]
        Only    Unfreeze Full Fine-tune
```

##### **Stage 1: Head-Only Training（0-9轮）**

**目标**：训练分类头，保持 backbone 冻结

**配置**：
- Backbone：冻结（`requires_grad=False`）
- Classifier：可训练
- 学习率：标准 LR（默认 5e-4）

**原理**：
- 利用预训练特征，快速收敛
- 避免破坏预训练知识
- 让分类头适应新任务

##### **Stage 2: Partial Fine-tuning（10-19轮）**

**触发条件**：`epoch == stage2_epoch`（默认10）

**操作**：
```python
# LR Bump (学习率微调)
for pg in optimizer.param_groups:
    pg["lr"] = pg["lr"] * 1.2  # 提升20%
```

**目标**：
- 准备进入backbone微调
- 轻微提升学习率以适应更深层参数
- 平滑过渡阶段

##### **Stage 3: Full Fine-tuning（20+轮）**

**触发条件**：`epoch == stage3_epoch`（默认20）

**操作**：
```python
# 解冻全部参数
for p in model.parameters():
    p.requires_grad = True
```

**目标**：
- 微调整个网络
- 学习任务特定特征
- 最大化性能

---

#### **动态超参数调整**

##### **标签平滑衰减**

```python
Epoch 0-11:   label_smoothing = 0.05
Epoch 12-17:  label_smoothing = 0.02  # smoothing_decay_epoch1
Epoch 18+:    label_smoothing = 0.00  # smoothing_decay_epoch2
```

**理由**：
- 前期：平滑标签防止过拟合
- 中期：减少平滑提高置信度
- 后期：无平滑，锐化决策边界

##### **采样器切换**

```python
Epoch 0-14:   WeightedRandomSampler (平衡采样)
Epoch 15+:    Shuffle (自然分布)
```

**理由**：
- 前期：平衡类别学习
- 后期：适应真实分布，避免过拟合采样分布

##### **学习率重启**

```python
Epoch 25:  lr = lr × 0.4
```

**理由**：
- 跳出局部最优
- 精细调优参数
- 类似 warm restart 策略

##### **渐进式分辨率提升**

```python
Epoch 0-29:   224×224
Epoch 30+:    256×256  # progressive_resize_epoch
```

**理由**：
- 低分辨率：快速训练，学习粗粒度特征
- 高分辨率：精细调优，捕捉细节特征
- 渐进式学习：稳定且高效

##### **最终清洁增强**

```python
Epoch 0-39:   完整数据增强
Epoch 40+:    轻量增强  # final_clean_epoch
```

**理由**：
- 减少增强噪声
- 让模型适应干净数据
- 提升测试性能

---

### 3. 损失函数策略

#### **Focal Loss for Tail Classes**

**触发时机**：`epoch >= tail_focal_epoch`（默认18）

**实现逻辑**：
```python
# 1. 识别长尾类别（底部25%）
tail_set = {低频类别ID集合}

# 2. 分离样本
tail_samples = samples where label in tail_set
non_tail_samples = samples where label not in tail_set

# 3. 分别计算损失
tail_loss = FocalLoss(tail_samples)  # γ=1.5
non_tail_loss = CrossEntropyLoss(non_tail_samples)

# 4. 合并损失
total_loss = tail_loss + non_tail_loss
```

**Focal Loss 参数**：
- α (alpha): None（不使用类别权重，因为已经有采样器）
- γ (gamma): 1.5（聚焦困难样本）

**效果**：
- 头部类别：保持稳定学习
- 尾部类别：加强困难样本学习
- 整体：平衡各类别性能

---

### 4. 特征增强技术

#### **Cosine Classifier 激活**

**触发条件**：
```python
if epoch == center_update_epoch and use_cosine_classifier:
    # 动态添加余弦分类器
    model.cosine_head = CosineClassifier(feat_dim, num_classes)
```

**应用场景**：
- 训练后期（默认第35轮）
- 仅在 baseline 模型使用
- 需要通过 `--use-cosine-classifier` 启用

**实现细节**：
1. 自动检测特征维度
2. 初始化归一化权重
3. 替换原有线性层

#### **Center Loss 激活**

**触发条件**：
```python
if epoch == center_update_epoch and use_center_loss:
    # 动态添加中心损失
    model.center_loss_mod = CenterLoss(num_classes, feat_dim)
```

**损失计算**：
```python
center_loss = model.center_loss_mod(features, targets)
total_loss = ce_loss + center_loss_weight * center_loss
```

**权重设置**：
- 默认：0.01（较小权重，辅助作用）
- 可调：`--center-loss-weight`

---

## 训练策略

### 1. 设备优化

#### **设备选择优先级**

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU 后备
```

#### **DataLoader 优化**

根据设备类型自动调整：

| 设备类型 | Workers | Pin Memory | Persistent Workers |
|---------|---------|------------|-------------------|
| CUDA    | 用户指定 | True       | True              |
| MPS     | min(4)  | False      | True              |
| CPU     | min(8)  | False      | True              |

**原理**：
- **CUDA**: 支持 pin_memory，加速 CPU→GPU 传输
- **MPS**: 内存传输开销大，减少 workers
- **CPU**: 无内存传输，增加 workers 提高并行度

#### **Mixed Precision Training**

```python
scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**限制**：
- 仅在 CUDA 设备启用
- MPS 和 CPU 使用 FP32

**优势**：
- 降低显存占用（约50%）
- 加速训练（1.5-2x）
- 保持数值稳定性

#### **Memory Format 优化**

```python
if device.type == "cuda":
    images = images.to(memory_format=torch.channels_last)
```

**Channels Last 格式**：
- 内存布局：NHWC（而非默认的 NCHW）
- 适配 TensorCore 加速
- 提升卷积操作性能

---

### 2. 优化器配置

#### **Few-shot 模式特殊处理**

```python
if args.model_type == "fewshot":
    base_lr = args.lr
    head_lr = base_lr * args.fewshot_head_lr_scale  # 默认5.0x
    
    param_groups = [
        {"params": head_params, "lr": head_lr}
    ]
```

**设计理念**：
- Backbone 冻结：保持预训练知识
- Head 高学习率：快速适应新任务
- 适用于小样本场景

#### **学习率调度**

##### **Warmup + Cosine Annealing**

```python
# Warmup (前5轮)
start_lr = 0.02 × target_lr  # 1e-5 for target_lr=5e-4
end_lr = target_lr            # 5e-4

# Cosine Decay (5-50轮)
lr_t = eta_min + 0.5 × (lr_max - eta_min) × (1 + cos(π × t / T))
eta_min = 1e-6
```

**Warmup 目的**：
- 避免大学习率破坏预训练权重
- 让 Batch Normalization 统计量稳定
- 平滑训练起始阶段

**Cosine 优势**：
- 平滑衰减，避免阶跃抖动
- 后期学习率足够小，精细调优
- 无需手动调整衰减节点

---

### 3. 梯度管理

#### **梯度裁剪**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**目的**：
- 防止梯度爆炸
- 稳定训练过程
- 特别重要于大 batch 和混合精度训练

**max_norm=1.0**：
- 适中的裁剪阈值
- 平衡稳定性和收敛速度

#### **梯度累积（隐式）**

虽然未显式实现，但可通过以下方式模拟：

```python
# 每 N 个 batch 更新一次
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**适用场景**：
- 显存不足时
- 需要大 batch size 效果时

---

## 命令行参数

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-dir` | str | `data/cleaned/train` | 训练数据目录 |
| `--val-dir` | str | `data/cleaned/val` | 验证数据目录 |
| `--train-meta` | str | `data/cleaned/metadata/train_metadata.csv` | 训练元数据 CSV |
| `--val-meta` | str | `data/cleaned/metadata/val_metadata.csv` | 验证元数据 CSV |
| `--class-weights` | str | `data/cleaned/metadata/class_weights.csv` | 类别权重文件 |

### 模型参数

| 参数 | 类型 | 默认值 | 选项 | 说明 |
|------|------|--------|------|------|
| `--model-type` | str | `baseline` | baseline, multitask, fewshot | 模型类型 |
| `--backbone` | str | `resnet50` | timm支持的模型 | 骨干网络 |
| `--pretrained` | bool | True | - | 使用预训练权重 |
| `--dropout` | float | 0.3 | 0.0-1.0 | Dropout 比率 |
| `--image-size` | int | 224 | - | 输入图像尺寸 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 50 | 训练轮数 |
| `--batch-size` | int | 32 | 批次大小 |
| `--lr` | float | 5e-4 | 学习率 |
| `--weight-decay` | float | 1e-4 | 权重衰减（L2正则化） |
| `--label-smoothing` | float | 0.05 | 标签平滑因子 |
| `--optimizer` | str | `adamw` | adam, adamw, sgd |
| `--scheduler` | str | `cosine` | cosine, step, none |
| `--use-amp` | bool | True | 使用自动混合精度 |

### 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--loss-type` | str | `weighted_ce` | weighted_ce, multitask, focal |
| `--use-class-weights` | bool | True | 使用类别权重 |

### 高级采样与增强参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--balance-sampler` | bool | False | 启用加权平衡采样 |
| `--mixup-alpha` | float | 0.4 | Mixup 强度（0禁用） |
| `--mixup-prob` | float | 0.7 | Mixup 应用概率 |
| `--cutmix-alpha` | float | 0.6 | CutMix 强度（0禁用） |
| `--cutmix-prob` | float | 0.5 | CutMix 应用概率 |
| `--use-ema` | bool | False | 启用模型EMA |
| `--ema-decay` | float | 0.999 | EMA 衰减率 |

### 阶段调度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--stage2-epoch` | int | 10 | Stage 2 开始轮数 |
| `--stage3-epoch` | int | 20 | Stage 3（全解冻）开始轮数 |
| `--smoothing-decay-epoch1` | int | 12 | 标签平滑第一次衰减 |
| `--smoothing-decay-epoch2` | int | 18 | 标签平滑第二次衰减（归零） |
| `--disable-sampler-epoch` | int | 15 | 禁用加权采样器 |
| `--lr-restart-epoch` | int | 25 | 学习率重启 |
| `--tail-focal-epoch` | int | 18 | 启用长尾Focal Loss |
| `--progressive-resize-epoch` | int | 30 | 提升分辨率 |
| `--progressive-image-size` | int | 256 | 提升后的图像尺寸 |
| `--final-clean-epoch` | int | 40 | 切换到轻量增强 |

### 正则化衰减参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mixup-decay-epoch1` | int | 10 | Mixup第一次衰减 |
| `--mixup-decay-epoch2` | int | 12 | Mixup第二次衰减 |
| `--mixup-disable-epoch` | int | 15 | 禁用Mixup |
| `--cutmix-disable-epoch` | int | 10 | 禁用CutMix |

### 特征增强参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-cosine-classifier` | bool | False | 启用余弦分类器 |
| `--use-center-loss` | bool | False | 启用中心损失 |
| `--center-loss-weight` | float | 0.01 | 中心损失权重 |
| `--center-update-epoch` | int | 35 | 特征增强激活轮数 |

### Few-shot 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fewshot-freeze-backbone` | bool | True | 冻结骨干网络 |
| `--fewshot-hidden` | int | 512 | 隐藏层维度 |
| `--fewshot-dropout` | float | 0.5 | Dropout比率 |
| `--fewshot-head-lr-scale` | float | 5.0 | 分类头学习率倍数 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-workers` | int | 4 | DataLoader工作进程数 |
| `--seed` | int | 42 | 随机种子 |
| `--save-dir` | str | `checkpoints/task1_baseline` | 检查点保存目录 |
| `--save-freq` | int | 5 | 保存检查点频率（轮） |
| `--log-interval` | int | 10 | 日志打印间隔（批次） |
| `--resume` | str | None | 恢复训练的检查点路径 |

---

## 使用示例

### 基础训练

```bash
python task1train.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4
```

### 完整配置（推荐）

```bash
python task1train.py \
    --backbone resnet50 \
    --model-type baseline \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --image-size 224 \
    --dropout 0.3 \
    --label-smoothing 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --use-amp \
    --balance-sampler \
    --mixup-alpha 0.4 \
    --mixup-prob 0.7 \
    --cutmix-alpha 0.6 \
    --cutmix-prob 0.5 \
    --use-ema \
    --ema-decay 0.999 \
    --stage2-epoch 10 \
    --stage3-epoch 20 \
    --progressive-resize-epoch 30 \
    --progressive-image-size 256 \
    --save-dir checkpoints/task1_full
```

### 高级特征增强训练

```bash
python task1train.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --use-ema \
    --balance-sampler \
    --mixup-alpha 0.4 \
    --cutmix-alpha 0.6 \
    --use-cosine-classifier \
    --use-center-loss \
    --center-loss-weight 0.01 \
    --center-update-epoch 35 \
    --save-dir checkpoints/task1_advanced
```

### Few-shot 训练

```bash
python task1train.py \
    --model-type fewshot \
    --backbone resnet50 \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-4 \
    --fewshot-freeze-backbone \
    --fewshot-head-lr-scale 5.0 \
    --fewshot-dropout 0.5 \
    --save-dir checkpoints/task1_fewshot
```

### 从检查点恢复

```bash
python task1train.py \
    --resume checkpoints/task1_baseline/best_custom.pth \
    --epochs 80 \
    --save-dir checkpoints/task1_resume
```

---

## 训练流程

### 完整训练时间线

```
Epoch:    0─────5─────10────15────20────25────30────35────40────45────50
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Warmup:  │█████│     │     │     │     │     │     │     │     │     │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Stage:   │ S1  │  S1 │  S2 │  S2 │  S3─────────────────────────────>│
Sampler: │Weighted───────┤Shuffle────────────────────────────────────>│
Mixup:   │ 0.4 │ 0.4 │ 0.2 │ 0.1 │  0  ────────────────────────────>│
CutMix:  │ 0.6 │ 0.6 │  0  ──────────────────────────────────────────>│
LS:      │0.05 │0.05 │0.05 │0.02 │0.02 │ 0.0 ────────────────────────>│
FocalTail│ No  │ No  │ No  │ Yes─────────────────────────────────────>│
LR-Scale:│ 1.0 │ 1.0 │ 1.2 │ 1.2 │ 1.2 │0.48 ────────────────────────>│
ImgSize: │ 224────────────────────────┤ 256 ────────────────────────>│
Augment: │ Full──────────────────────────────────────┤ Light ────────>│
Feature: │ Linear Classifier ───────────────────────┤Cosine+Center──>│
```

**图例说明**：
- S1/S2/S3: 训练阶段
- LS: Label Smoothing
- LR-Scale: 学习率相对缩放
- ImgSize: 图像分辨率
- Augment: 数据增强强度

### 各阶段详细说明

#### **阶段1：特征适配（Epoch 0-9）**

**目标**：让分类头适应新任务

**配置**：
- 骨干网络：冻结
- 分类头：可训练
- 学习率：标准（5e-4）+ Warmup
- 数据增强：完整（Mixup + CutMix）
- 采样策略：加权平衡采样

**预期效果**：
- 快速收敛到60-70%准确率
- 分类头学习基本决策边界

#### **阶段2：过渡准备（Epoch 10-19）**

**目标**：准备深层微调

**变化**：
- 学习率提升：×1.2
- Mixup衰减：0.4 → 0.2 → 0.1
- CutMix禁用：Epoch 10
- 标签平滑衰减：0.05 → 0.02
- 采样器禁用：Epoch 15
- Mixup禁用：Epoch 15

**预期效果**：
- 准确率提升到75-80%
- 决策边界更清晰

#### **阶段3：全网络微调（Epoch 20-29）**

**目标**：学习任务特定特征

**变化**：
- 骨干网络解冻
- 长尾Focal Loss启用：Epoch 18
- 标签平滑归零：Epoch 18
- 学习率重启：Epoch 25（×0.4）

**预期效果**：
- 准确率提升到82-87%
- 尾部类别性能改善

#### **阶段4：高分辨率精调（Epoch 30-39）**

**目标**：捕捉细粒度特征

**变化**：
- 图像尺寸：224 → 256
- 保持所有优化策略

**预期效果**：
- 准确率提升到88-92%
- 细节特征识别增强

#### **阶段5：特征增强收敛（Epoch 40-50）**

**目标**：最终性能优化

**变化**：
- 数据增强：轻量化
- 余弦分类器激活：Epoch 35
- 中心损失激活：Epoch 35

**预期效果**：
- 准确率达到92-95%
- 特征判别性最大化
- 类内紧凑，类间分离

---

## 性能优化

### 训练速度优化

#### **1. 数据加载优化**

```python
# CUDA 设备
DataLoader(
    dataset,
    num_workers=4,        # 多进程加载
    pin_memory=True,      # 固定内存
    persistent_workers=True  # 保持工作进程
)
```

**效果**：
- 减少CPU→GPU传输时间
- 避免进程重复创建开销
- 提速约20-30%

#### **2. 混合精度训练**

```python
# FP16 计算 + FP32 累积
with torch.cuda.amp.autocast():
    outputs = model(images)
```

**效果**：
- 显存占用减少50%
- 训练速度提升1.5-2x
- Batch size 可增大

#### **3. Channels Last 内存格式**

```python
images = images.to(memory_format=torch.channels_last)
```

**效果**：
- TensorCore 加速
- 卷积操作提速10-15%

#### **4. 梯度累积**

```python
# 模拟大 batch size
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

**适用场景**：
- 显存不足
- 需要大 batch 稳定训练

### 内存优化

#### **1. 梯度检查点（Gradient Checkpointing）**

虽然未在代码中实现，但可添加：

```python
from torch.utils.checkpoint import checkpoint

# 在模型forward中
def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    # ...
```

**效果**：
- 显存占用减少60-80%
- 训练速度降低20-30%

#### **2. 梯度清零优化**

```python
optimizer.zero_grad(set_to_none=True)
```

**优势**：
- 比 `zero_grad()` 更快
- 减少内存写入操作

#### **3. EMA Shadow 参数管理**

```python
# 仅存储可训练参数的shadow
for name, param in model.named_parameters():
    if param.requires_grad:
        self.shadow[name] = param.data.clone()
```

**优势**：
- 节省内存
- 加快 EMA 更新

---

## 常见问题

### Q1: 训练过程中显存不足（OOM）怎么办？

**解决方案**：

1. **减小 batch size**
   ```bash
   --batch-size 16  # 从32减到16
   ```

2. **降低图像分辨率**
   ```bash
   --image-size 192  # 从224减到192
   ```

3. **禁用 EMA**
   ```bash
   # 不添加 --use-ema 参数
   ```

4. **减少 workers**
   ```bash
   --num-workers 2
   ```

5. **使用梯度累积**
   ```python
   # 需要修改代码实现
   accumulation_steps = 4
   ```

### Q2: 训练速度太慢怎么办？

**解决方案**：

1. **启用 AMP**
   ```bash
   --use-amp
   ```

2. **增加 workers**
   ```bash
   --num-workers 8  # CUDA设备
   ```

3. **使用更小的模型**
   ```bash
   --backbone resnet34  # 替代 resnet50
   ```

4. **减少数据增强**
   ```bash
   --mixup-alpha 0
   --cutmix-alpha 0
   ```

5. **禁用 persistent_workers**（如果内存充足）
   ```python
   # 代码中修改
   persistent_workers=False
   ```

### Q3: 验证准确率不提升或震荡怎么办？

**解决方案**：

1. **启用 EMA**
   ```bash
   --use-ema --ema-decay 0.999
   ```

2. **降低学习率**
   ```bash
   --lr 3e-4  # 从5e-4降低
   ```

3. **增加 warmup 轮数**
   ```python
   # 代码中修改
   warmup_epochs = 10  # 从5增到10
   ```

4. **减少标签平滑**
   ```bash
   --label-smoothing 0.02
   ```

5. **检查数据质量**
   - 查看训练样本是否有标注错误
   - 使用 `visualize_samples.py` 检查

### Q4: 长尾类别准确率低怎么办？

**解决方案**：

1. **启用加权采样**
   ```bash
   --balance-sampler
   ```

2. **延长采样器使用时间**
   ```bash
   --disable-sampler-epoch 20  # 从15延长到20
   ```

3. **提前启用 Focal Loss**
   ```bash
   --tail-focal-epoch 15  # 从18提前到15
   ```

4. **增加 Focal Loss gamma**
   ```bash
   --focal-gamma 2.0  # 从1.5增到2.0
   ```

5. **使用类别权重**
   ```bash
   --use-class-weights
   ```

### Q5: 如何选择合适的 backbone？

**推荐配置**：

| Backbone | 参数量 | 训练时间 | 准确率 | 适用场景 |
|----------|--------|---------|--------|---------|
| resnet34 | 21M | 快 | 中 | 快速实验，显存受限 |
| resnet50 | 25M | 中 | 高 | **推荐默认选择** |
| resnet101 | 44M | 慢 | 高 | 追求最高精度 |
| efficientnet_b0 | 5M | 快 | 中高 | 边缘设备部署 |
| efficientnet_b3 | 12M | 中 | 高 | 平衡性能和速度 |
| convnext_tiny | 28M | 中快 | 高 | 现代架构，推荐尝试 |

**选择建议**：
- 初次训练：`resnet50`
- 显存受限：`efficientnet_b0` 或 `resnet34`
- 追求精度：`resnet101` 或 `convnext_base`
- 生产部署：`efficientnet_b3`

### Q6: 如何判断模型是否过拟合？

**检查指标**：

1. **训练 vs 验证准确率差距**
   ```
   过拟合标志：Train Acc - Val Acc > 10%
   ```

2. **验证损失趋势**
   ```
   过拟合标志：Val Loss 持续上升
   ```

3. **长尾类别表现**
   ```
   过拟合标志：TailAcc 显著低于 Overall Acc
   ```

**解决方案**：

1. **增强正则化**
   ```bash
   --dropout 0.5  # 从0.3增加
   --weight-decay 5e-4  # 从1e-4增加
   ```

2. **增强数据增强**
   ```bash
   --mixup-alpha 0.6
   --cutmix-alpha 0.8
   --label-smoothing 0.1
   ```

3. **提早停止**
   - 监控 `best_val_acc`
   - 连续5轮无提升则停止

4. **减少模型容量**
   ```bash
   --backbone resnet34  # 使用更小模型
   ```

### Q7: MPS 设备训练异常怎么办？

**常见问题**：
- AMP 报错
- Memory format 报错
- 训练速度慢

**解决方案**：

1. **禁用 AMP**（代码已自动处理）
   ```python
   # MPS 设备自动禁用 AMP
   use_amp = args.use_amp and device.type == "cuda"
   ```

2. **减少 workers**（代码已自动处理）
   ```python
   # MPS 自动设置 workers=min(4)
   ```

3. **避免 channels_last**（代码已处理）
   ```python
   # 仅 CUDA 使用 channels_last
   if device.type == "cuda":
       images = images.to(memory_format=torch.channels_last)
   ```

4. **使用较小 batch size**
   ```bash
   --batch-size 16  # MPS 显存较小
   ```

### Q8: 如何复现训练结果？

**保证可复现性**：

1. **固定随机种子**
   ```bash
   --seed 42
   ```

2. **禁用 cudnn benchmark**（代码已实现）
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

3. **使用相同配置**
   - 保存完整命令行参数
   - 或使用配置文件

4. **固定软件版本**
   ```bash
   # 记录环境
   pip freeze > requirements.txt
   ```

**注意**：
- 不同设备（CUDA/MPS/CPU）结果可能略有差异
- 多卡训练需要额外同步设置

### Q9: 如何监控训练过程？

**方法1：命令行输出**
```
Epoch 25/50 | Train Loss 0.3247 | Train Acc 89.23% | LR 0.000234
  [Val] Epoch 25 | Loss 0.4156 | Acc 87.45% | TailAcc 78.32% | MacroF1 0.856
  ✅ New best val acc: 87.45% | TailAcc 78.32% | MacroF1 0.856
```

**方法2：TensorBoard**
```bash
tensorboard --logdir checkpoints/task1_baseline/logs
```

查看指标：
- Train/Val Loss 曲线
- Train/Val Accuracy 曲线
- Learning Rate 曲线

**方法3：检查点文件**
```python
checkpoint = torch.load("checkpoints/task1_baseline/best_custom.pth")
print(f"Best Epoch: {checkpoint['epoch']}")
print(f"Val Acc: {checkpoint['val_acc']:.2f}%")
print(f"Tail Acc: {checkpoint['tail_mean_acc']:.2f}%")
print(f"Macro F1: {checkpoint['macro_f1']:.3f}")
```

### Q10: 如何为新数据集调整超参数？

**调优顺序**：

1. **先确定基础配置**
   ```bash
   --backbone resnet50
   --epochs 50
   --batch-size 32
   --lr 5e-4
   ```

2. **观察初期训练（0-10轮）**
   - 如果损失下降慢 → 提高学习率
   - 如果损失震荡 → 降低学习率
   - 如果过拟合 → 增加正则化

3. **调整数据增强**
   - 小数据集 → 增强 Mixup/CutMix
   - 大数据集 → 减少增强
   - 高质量标注 → 减少标签平滑

4. **调整采样策略**
   - 严重不平衡 → 启用加权采样
   - 轻微不平衡 → 仅使用类别权重
   - 平衡数据集 → 禁用采样器

5. **精细调整**
   - 根据验证集表现调整各阶段节点
   - 平衡 Overall Acc 和 Tail Acc

---

## 输出文件

### 检查点文件

保存在 `--save-dir` 目录下：

- `best_custom.pth`：最佳验证准确率的模型
- `epoch_N.pth`：每隔 `--save-freq` 轮保存的检查点
- `interrupted.pth`：中断训练时的紧急保存

### 检查点内容

```python
{
    'epoch': int,                    # 训练轮数
    'model_state_dict': OrderedDict, # 模型参数
    'optimizer_state_dict': dict,    # 优化器状态
    'best_val_acc': float,           # 最佳验证准确率
    'train_loss': float,             # 训练损失
    'train_acc': float,              # 训练准确率
    'val_loss': float,               # 验证损失
    'val_acc': float,                # 验证准确率
    'tail_mean_acc': float,          # 长尾类别平均准确率
    'macro_f1': float,               # 宏平均F1分数
}
```

### TensorBoard 日志

位置：`{save_dir}/logs/`

包含：
- Scalar：Loss、Accuracy、Learning Rate
- 可通过 `tensorboard --logdir {save_dir}/logs` 查看

---

## 总结

Task 1 训练脚本实现了一套**生产级别的深度学习训练流程**，包含：

**核心优势**：
1. **多阶段训练策略**：逐步解冻，稳定收敛
2. **长尾分布优化**：加权采样 + Focal Loss + 类别权重
3. **高级数据增强**：Mixup + CutMix + 渐进式分辨率
4. **训练稳定性**：EMA + 标签平滑 + 梯度裁剪 + Warmup
5. **特征增强**：Cosine Classifier + Center Loss
6. **跨平台支持**：CUDA/MPS/CPU 自动优化
7. **灵活配置**：丰富的命令行参数和阶段调度

**适用场景**：
- 多类别农作物病害识别
- 长尾分布图像分类
- 小样本学习任务
- 类别不平衡问题

**推荐工作流程**：
1. 基础训练（50轮）
2. 分析错误案例
3. 调整超参数
4. 启用高级特性（EMA、Cosine Classifier）
5. 最终模型评估

**下一步**：
- 在验证集上评估模型
- 使用混淆矩阵分析错误
- 针对性优化困难类别
- 测试集最终评估