# homework2-c 实施计划

## 目标

在 5-shot (162 样本 / 34 类) 场景下，使用 TextCNN 超越 DADGNN 的 14.3% dev accuracy。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型 | TextCNN (Kim 2014) | 最轻量深度模型，强归纳偏置，5-shot 友好 |
| 训练数据 | train_new_5shot.csv (162 条) | 课程要求 5-shot |
| 验证数据 | dev_new.csv (3200 条) | 仅用于验证和早停 |
| Embedding | Word2Vec 200d + freeze | 减少可训练参数，避免过拟合 |
| 卷积配置 | filter sizes [2,3,4], 64 filters/size | 总 192 维特征，~108K 可训练参数 |
| 分类头 | Dropout(0.5) → Linear(192, 34) | 轻量分类器 |
| 正则化 | Dropout 0.5 + weight_decay 1e-3 + label_smoothing 0.1 | 三种机制互补，防过拟合 |
| Batch size | 16 | 162/16 ≈ 10 步/epoch，细粒度更新 |
| 学习率 | 5e-4 | embedding freeze 后可稍高 |
| Epochs | 200 + patience 30 | 给足收敛机会 |
| Optimizer | AdamW | 配合 weight decay |
| 分词 | jieba, 输入 sentence_sep 列 | 与 homework2/homework2-b 一致 |
| max_length | 128 | 中文对话通常不长，减少无效 padding |
| 数据增强 | 不加，先跑纯 baseline | 避免无法归因，不够再加 |

## 模型架构

```
Input (token IDs, max_length=128)
  → Embedding(vocab_size, 200, freeze=True)  # Word2Vec 初始化
  → Conv1d(200, 64, kernel_size=2) + ReLU + MaxPool1d
  → Conv1d(200, 64, kernel_size=3) + ReLU + MaxPool1d
  → Conv1d(200, 64, kernel_size=4) + ReLU + MaxPool1d
  → Concat → [192]
  → Dropout(0.5)
  → Linear(192, 34)
```

## 实施步骤

### Step 1: 从 homework2-b 复制基础代码

复制以下文件，保持目录结构：
- `main.py` — CLI 入口 (train/predict)
- `src/data_loader.py` — CSV 加载、jieba 分词、Dataset
- `src/vocab.py` — 词汇表构建、Word2Vec 加载
- `src/train.py` — 训练循环 (需微调)
- `src/predict.py` — 推理和提交生成
- `pyproject.toml` — 项目依赖

### Step 2: 替换 model.py

删除 TinyDeBERTa，实现 TextCNN：
- `TextCNN` 类：Embedding(freeze) + 多尺度 Conv1d + MaxPool + Dropout + Linear
- `__init__` 参数：vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout
- `forward`：embedding lookup → unsqueeze → conv+relu+maxpool per filter → cat → dropout → linear

### Step 3: 修改 train.py

- 替换模型实例化为 TextCNN
- 更新超参数：lr=5e-4, batch_size=16, epochs=200, patience=30
- 添加 label_smoothing=0.1 到 CrossEntropyLoss
- 添加 weight_decay=1e-3 到 AdamW
- freeze embedding：遍历 model.embedding.parameters() 设 requires_grad=False
- max_length 改为 128
- 保存/加载逻辑适配 TextCNN

### Step 4: 修改 data_loader.py

- max_length 默认值改为 128

### Step 5: 修改 main.py

- 数据路径指向 `../nlp-text-classification-experiments/`（与 homework2-b 一致）
- 确认 train/predict 子命令正常工作

### Step 6: 验证

- `uv sync` 安装依赖
- `python main.py train` 训练
- `python main.py predict` 生成 submission.csv
- 对比 dev accuracy 是否超过 14.3%

## 预期参数量

| 层 | 参数 |
|----|------|
| Embedding | ~2M (freeze, 不计入可训练) |
| Conv1d(200, 64, k=2) | 200×64×2 + 64 = 25,664 |
| Conv1d(200, 64, k=3) | 200×64×3 + 64 = 38,464 |
| Conv1d(200, 64, k=4) | 200×64×4 + 64 = 51,264 |
| Linear(192, 34) | 192×34 + 34 = 6,562 |
| **可训练总计** | **~122K** |

## Baseline 对比

| 模型 | 可训练参数 | Dev Acc |
|------|-----------|---------|
| DADGNN (homework2) | ~260K (fine-tune) | 14.3% |
| TinyDeBERTa (homework2-b) | ~2.6M (fine-tune) | 11.78% |
| TextCNN (homework2-c) | ~122K (freeze) | ? |

## 回退方案

如果纯 TextCNN baseline 未超过 14.3%，按顺序尝试：
1. 加随机 dropout 词增强（训练时随机丢弃 10-20% token）
2. 调整 filter_sizes 为 [2,3,4,5]，增加覆盖范围
3. 尝试 unfreeze embedding + 更低学习率
