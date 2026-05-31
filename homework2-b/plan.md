# Homework2-b: Tiny DeBERTa for 5-Shot SLU Intent Detection

## Overview

在 homework2-b 中实现基于 DeBERTa 架构（简化版）的中文 SLU 意图检测模型，与 homework2 的 DADGNN 进行对比。严格遵循课程约束：不使用预训练模型，仅使用 162 条 5-shot 训练数据。

## 架构决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型规模 | 2 层、128 维、2 头、FFN 512（~1M 参数） | 5-shot 极端场景，模型容量必须极度压缩 |
| 注意力 | 标准自注意力 + 可学习相对位置编码 | 解耦注意力参数过多，162 条数据无法学到 |
| 输入 | jieba + Word2Vec 200d → 线性投影 128d | 与 homework2 一致，Word2Vec 提供关键语义先验 |
| 分类头 | 平均池化 → dropout → linear | 零额外参数，比 [CLS] 更稳定 |
| 正则 | dropout=0.5, wd=1e-3, label_smoothing=0.1, patience=10 | 强正则抑制过拟合 |
| 评估 | dev accuracy | 简单直接对比 |

## 目录结构

```
homework2-b/
├── main.py                  # 入口：train / predict 命令
├── src/
│   ├── data_loader.py       # CSV 加载、jieba 分词、标签编码
│   ├── vocab.py             # 词汇表构建、Word2Vec 加载
│   ├── model.py             # Tiny DeBERTa 模型定义
│   ├── train.py             # 训练循环（含 early stopping、label smoothing）
│   └── predict.py           # 加载模型 → 批量推理 → 生成 submission.csv
├── results/                 # 保存模型、配置、指标（gitignore）
├── data/                    # 数据文件（gitignore，复用 homework2/data/ 的 Word2Vec）
└── README.md
```

## 实现步骤

### Step 1: 项目初始化

- 创建 `homework2-b/` 目录及子目录
- 创建 `homework2-b/pyproject.toml`，依赖：torch, gensim, jieba, numpy, pandas, scikit-learn
- 数据 CSV 复用 `../nlp-text-classification-experiments/` 路径，Word2Vec 复用 `../homework2/data/`

### Step 2: data_loader.py

- 从 CSV 加载 train/dev/test 数据
- jieba 分词处理中文文本
- 构建 `LabelEncoder`：c_numerical ↔ index 映射（34 类）
- 输出：`word_mf2` 列作为原始文本，`c_numerical` 作为标签

### Step 3: vocab.py

- 从 train+dev 文本构建词汇表（min_freq=1），加 PAD/UNK 特殊 token
- 加载 Word2Vec embedding（优先 Tencent 200d，fallback sgns.merge.word）
- 构建初始 embedding 矩阵，未登录词随机初始化
- 提供 `encode(text)` 方法：分词 → token id 序列，截断到 max_length

### Step 4: model.py — Tiny DeBERTa

核心组件：

1. **RelativePositionBias**
   - 可学习参数：`relative_bias` shape=(2*max_seq_len-1, num_heads)
   - 输入 query/key 位置差，查表得到位置 bias 加到 attention score

2. **MultiHeadSelfAttention**
   - 标准 Q/K/V 线性投影 + 相对位置 bias
   - 2 头，dim=128（每头 64）
   - 输出线性投影 + dropout

3. **TransformerBlock**
   - MultiHeadSelfAttention → LayerNorm → 残差连接
   - FFN(128→512→128) + GELU → LayerNorm → 残差连接
   - Dropout 贯穿

4. **TinyDeBERTa**
   - Input: token_ids (batch, seq_len)
   - Embedding lookup (vocab_size, 200) → Linear(200, 128) → dropout
   - 2 层 TransformerBlock
   - Mean pooling（mask 掉 PAD token）
   - Dropout(0.5) → Linear(128, 34)
   - 输出 logits

### Step 5: train.py

- 超参数：
  - learning_rate: 2e-4
  - weight_decay: 1e-3
  - dropout: 0.5
  - label_smoothing: 0.1
  - batch_size: 16
  - max_epochs: 100
  - early_stop_patience: 10
  - max_seq_length: 350
  - seed: 42
- AdamW 优化器
- CrossEntropyLoss(label_smoothing=0.1)
- 每 epoch 评估 dev accuracy，保存最佳模型
- Early stopping：连续 10 epoch 无提升则停止
- 保存：模型权重、训练配置、标签映射、词汇表、metrics

### Step 6: predict.py

- 加载保存的模型、配置、词汇表、标签映射
- 对 test 数据做同样的 jieba 分词 + encode
- 批量推理，argmax 得到预测类别
- 映射回 c_numerical，生成 `submission.csv`（ID, c_numerical）

### Step 7: main.py

- `python main.py train` — 执行训练流程
- `python main.py predict` — 执行预测流程

## 关键风险

1. **162 条数据训练 transformer 仍然很难** — 即使 Tiny 版本，自注意力的参数量比 DADGNN 的 GAT 层更多。dev accuracy 可能不如 DADGNN。
2. **相对位置编码的 max_seq_len** — 设置过大会浪费参数，过小会截断长文本。建议 512，覆盖 99% 样本。
3. **Word2Vec 覆盖率** — jieba 分词后部分词可能在 Word2Vec 中缺失，随机初始化的词在 5-shot 下几乎学不到。
