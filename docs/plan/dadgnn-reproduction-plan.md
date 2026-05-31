# Plan: 复现 DADGNN 到 homework2

## Context

homework2 当前是 `bert-base-chinese` 微调做 34 类中文 SLU 意图检测。用 DADGNN（EMNLP 2021, 135 引用）论文方法完全替换现有 BERT 微调代码。DADGNN 将文档转为词图（n-gram 滑窗建边），用 Attention Diffusion GNN 在图上做分类。原汁原味复现（中文 Word2Vec 词向量 + GNN），引入 DGL 依赖，保持 homework2 代码风格。

## 架构

```
中文对话文本 → jieba 分词 → 词表查找 → 去重词作为图节点
                                      → n-gram 滑窗构建有向边
                                      → 中文 Word2Vec 300d Embedding
                                      → Linear(300→64) + ReLU
                                      → Linear(64→34) + Dropout
                                      → Attention Diffusion GNN (5层, 5步扩散, 2头)
                                      → WeightAndSum 图级读出
                                      → 34 类 logits → CrossEntropyLoss
```

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `pyproject.toml` | 修改 | 添加 `dgl`, `gensim` 依赖 |
| `homework2/src/__init__.py` | 修改 | 更新包描述 |
| `homework2/src/data_loader.py` | 重写 | jieba 分词 + 词表构建 + 图数据集 |
| `homework2/src/model.py` | 新建 | SingleHeadGATLayer + GATLayer + GATNet + DADGNNModel |
| `homework2/src/graph_utils.py` | 新建 | 文档→DGL图转换（n-gram 边构建等） |
| `homework2/src/train.py` | 重写 | DADGNN 训练循环（不用 HuggingFace Trainer） |
| `homework2/src/predict.py` | 重写 | DADGNN 推理 + submission.csv 生成 |
| `homework2/main.py` | 微调 | 更新打印信息和注释 |

## 各文件详细设计

### 1. `pyproject.toml`

添加依赖：`dgl>=2.0`, `gensim>=4.0`

### 2. `homework2/src/data_loader.py`

保留：`load_csv`, `save_json`, `load_json`, `build_label_metadata`

新增：
- `tokenize_chinese(text) -> list[str]`：jieba 分词
- `build_vocab(documents, min_freq=1) -> (word2id, id2word)`：从语料构建词表，`<PAD>=0`, `<UNK>=1`
- `encode_document(tokens, word2id, max_length) -> list[int]`：词→id，UNK+截断
- `load_chinese_embeddings(path, word2id, embed_dim=300) -> ndarray`：gensim 加载，未覆盖词随机初始化
- `GraphTextDataset(Dataset)`：`__getitem__` 返回 `doc_ids` (tensor) + `label` (tensor)

### 3. `homework2/src/graph_utils.py`

- `doc_to_graph(doc_ids, node_hidden, ngram, max_length, device) -> dgl.DGLGraph`：
  - 截断→去重建节点→n-gram 滑窗建边→查 Embedding→返回 DGLGraph
  - 用 `dgl.graph()` 替代废弃的 `dgl.DGLGraph()`
- `batch_graphs(doc_ids_list, ...) -> dgl.DGLGraph`：批量建图 + `dgl.batch()`

### 4. `homework2/src/model.py`

移植 DADGNN 原仓库，适配修改：
- `SingleHeadGATLayer`：k 步 Attention Diffusion，PPR 风格可学习扩散系数，移除硬编码 cuda
- `GATLayer`：多头包装器，merge='mean'
- `GATNet`：堆叠 num_layers 个 GATLayer，移除未使用的 self.s
- `DADGNNModel`：
  - Embedding + Linear(300→64) + ReLU + Linear(64→34) + GATNet + WeightAndSum
  - 支持加载预训练中文词向量
  - `forward(doc_ids_list) -> logits (batch_size, class_num)`
  - 图构建在 forward 中动态完成（与原论文一致）

### 5. `homework2/src/train.py`

原生 PyTorch 训练循环，超参默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| embed_dim | 300 | 词向量维度 |
| num_hidden | 64 | 隐藏层维度 |
| num_layers | 5 | GNN 层数 |
| num_heads | 2 | 注意力头数 |
| k | 5 | 扩散步数 |
| alpha | 0.5 | 扩散率 |
| ngram | 4 | 滑窗大小 |
| max_length | 350 | 最大文档长度 |
| dropout | 0.5 | Dropout 率 |
| learning_rate | 1e-3 | 学习率 |
| weight_decay | 1e-6 | 权重衰减 |
| batch_size | 64 | 批大小 |
| num_train_epochs | 100 | 最大训练轮数 |
| early_stop_patience | 10 | 早停耐心 |
| seed | 42 | 随机种子 |

训练流程：分词→建词表→加载词向量→建模型→Adam+CrossEntropy→每 epoch 评估 dev→早停→保存最佳模型

### 6. `homework2/src/predict.py`

加载模型+词表→分词→编码→批量推理→argmax→映射 c_numerical→写 submission.csv

### 7. `homework2/main.py`

微调打印信息和描述，接口不变。

## 中文词向量

- 来源：[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 的 Word2Vec skip-gram 300d
- 存放：`homework2/data/sgns.merge.word`（gitignored）
- Fallback：文件不存在时随机初始化 Embedding

## 关键适配点（中文 vs 英文）

1. 分词：空格→jieba
2. 词向量：GloVe 6B 300d→中文 Word2Vec 300d
3. 词表：预定义→动态构建
4. 标签：2/4/5 类→34 类意图
5. 设备：硬编码 cuda→自动检测
6. DGL API：`dgl.DGLGraph()`→`dgl.graph()`

## 验证

1. `cd homework2 && python main.py train` — 训练完成，打印 dev accuracy
2. `cd homework2 && python main.py predict` — 生成 submission.csv
3. submission.csv 格式：两列 `ID` + `c_numerical`，行数 = 4000
4. `results/` 下有 `model/dadgnn_model.pt`, `training_config.json`, `label_metadata.json`, `vocab.json`
