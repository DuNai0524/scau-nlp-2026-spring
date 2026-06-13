# 口语对话理解 - 意图检测任务

基于 LLaMA-Factory 和 Qwen2.5-7B-Instruct 的中文客服对话意图分类系统。

## 🎯 任务描述

- **任务类型**: 中文客服对话意图分类（34分类）
- **训练数据**: 162 条（5-shot）
- **验证数据**: 3200 条
- **测试数据**: 4000 条
- **评价指标**: Accuracy

## 🛠️ 环境配置

### 硬件要求
- **GPU**: RTX 3090 (24GB VRAM)
- **内存**: 24GB+
- **存储**: 30GB+（模型缓存）

### 软件环境
```bash
# 1. 安装环境（只需执行一次）
bash setup.sh

# 2. 激活环境
source .venv/bin/activate
```

## 🚀 快速开始

### 方式1：完整端到端流程
```bash
# 数据准备 + 训练 + 预测
./run.sh
```

### 方式2：分步执行
```bash
# 1. 数据预处理
./run.sh prepare

# 2. 模型训练
./run.sh train

# 3. 验证集评估
./run.sh eval

# 4. 测试集预测
./run.sh predict
```

## 📁 项目结构

```
nlp_homework_04/
├── configs/
│   └── qwen2.5_lora_sft.yaml    # LoRA微调配置
├── data/
│   └── intent_classification/    # 转换后的训练数据
├── nlp-text-classification-experiments/
│   ├── train_new_5shot.csv      # 训练集
│   ├── dev_new.csv              # 验证集
│   ├── kaggle_test.csv          # 测试集
│   └── sample_submission.csv    # 提交样例
├── prepare_data.py              # 数据预处理脚本
├── predict.py                   # 推理预测脚本
├── train.sh                     # 训练启动脚本
├── run.sh                       # 完整流程脚本
├── setup.sh                     # 环境安装脚本
└── submission.csv               # 预测结果（生成）
```

## ⚙️ 模型配置

### 基座模型
- **模型**: Qwen/Qwen2.5-7B-Instruct
- **规模**: 7B参数
- **优势**: 中文理解能力强，指令遵循优秀

### LoRA 配置
```yaml
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: all
```

### 训练参数
```yaml
learning_rate: 2.0e-4
batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 10
warmup_ratio: 0.1
max_length: 1024
```

## 📊 训练流程

### 1. 数据转换
将 CSV 转换为 LLaMA-Factory 的 Alpaca 格式：

```json
{
  "instruction": "请判断以下客服对话的意图类别...",
  "input": "用户对话：您好请讲...",
  "output": "30"
}
```

### 2. 启动训练
```bash
cd /root/LlamaFactory
llamafactory-cli train \
    /root/nlp_homework_04/configs/qwen2.5_lora_sft.yaml
```

### 3. 监控训练
- 日志: `saves/qwen2.5-7b/lora/sft/trainer_log.jsonl`
- 可视化: `saves/qwen2.5-7b/lora/sft/training_loss.png`

## 🔮 推理预测

### 使用微调模型预测
```bash
python predict.py \
    --test_file nlp-text-classification-experiments/kaggle_test.csv \
    --output submission.csv \
    --batch_size 8
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model` | 基础模型路径 | Qwen/Qwen2.5-7B-Instruct |
| `--lora_path` | LoRA适配器路径 | saves/qwen2.5-7b/lora/sft |
| `--test_file` | 测试集CSV | kaggle_test.csv |
| `--batch_size` | 批处理大小 | 8 |
| `--output` | 输出文件 | submission.csv |

## 📈 结果分析

### 验证集评估
```bash
./run.sh eval
```

输出示例：
```
验证集准确率: 0.8234 (82.34%)
```

### 提交格式
```csv
Id,Category
1,30
2,1
3,15
...
```

## 🎛️ 调优建议

### 1. 调整 LoRA 参数
```yaml
# 如果过拟合
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.1

# 如果欠拟合
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.05
```

### 2. 调整学习率
```yaml
# 尝试范围: 1e-4 ~ 5e-4
learning_rate: 1.0e-4  # 更保守
learning_rate: 3.0e-4  # 更激进
```

### 3. 调整训练轮数
```yaml
# 数据量小，适当增加轮数
num_train_epochs: 15.0
```

## 🐛 常见问题

### Q1: CUDA Out of Memory
**解决方案**: 减小 batch_size 或启用梯度检查点
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### Q2: 模型下载慢
**解决方案**: 配置 HuggingFace 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: 预测结果不准确
**解决方案**: 
- 检查训练是否收敛（查看 loss 曲线）
- 增加训练轮数
- 调整 prompt 模板

## 📚 参考资源

- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

## 📝 许可证

本项目仅供学习和研究使用。
