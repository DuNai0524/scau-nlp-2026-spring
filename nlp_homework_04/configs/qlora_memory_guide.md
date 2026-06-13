# QLoRA 显存优化配置说明

## 修改对比

### 1. 量化配置（关键）
```yaml
# 新增 - 4-bit 量化
quantization_bit: 4
quantization_method: bitsandbytes
```

### 2. LoRA 参数优化
| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| lora_rank | 64 | 16 | 降低 rank 减少可训练参数 |
| lora_alpha | 128 | 32 | 与 rank 保持 2:1 比例 |
| use_rslora | false | true | 使用 Rank-Stabilized LoRA，训练更稳定 |

### 3. 训练参数优化
| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| per_device_train_batch_size | 4 | 1 | 减小单设备 batch size |
| gradient_accumulation_steps | 4 | 16 | 增大梯度累积，保持等效 batch size = 16 |
| cutoff_len | 1024 | 512 | 减小序列长度 |
| bf16 | true | false | QLoRA 使用 fp16 更兼容 |
| output_dir | lora | qlora | 区分输出目录 |

## 显存占用对比

| 配置 | 预估显存 | 适用显卡 |
|------|----------|----------|
| 原始 LoRA (bf16) | ~60-65 GB | A100 80GB |
| **QLoRA (4-bit)** | **~8-12 GB** | **RTX 3090/4090, A10** |

## 启动训练

```bash
bash train.sh
```

## 额外优化技巧

如果仍然显存不足，可以进一步调整：

### 1. 减小序列长度
```yaml
cutoff_len: 256  # 甚至 128
```

### 2. 启用梯度检查点
```yaml
# 在 config 中添加
flash_attn: fa2  # 如果使用 Flash Attention 2
```

### 3. 使用分页优化器
```yaml
# LLaMA-Factory 默认使用，无需配置
```

### 4. 监控显存使用
训练时观察输出日志，或使用 `nvidia-smi` 监控：
```bash
watch -n 1 nvidia-smi
```

## 常见问题

### Q: 量化后模型质量会下降吗？
A: QLoRA 的 4-bit Normal Float (NF4) 量化经过特殊设计，配合双量化和分页优化器，通常能保持 99%+ 的原始性能。

### Q: 训练速度会变慢吗？
A: 会略慢（约 20-30%），因为需要实时反量化，但大幅降低显存占用的收益远大于速度损失。

### Q: 如何恢复之前的 LoRA 配置？
A: 备份文件为 `configs/qwen2.5_lora_sft.yaml.bak`（如有），或使用 git 恢复：
```bash
git checkout configs/qwen2.5_lora_sft.yaml
```
