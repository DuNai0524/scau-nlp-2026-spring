# Homework 1: SLU Intent Detection

基于 NumPy 实现的中文口语理解意图识别，当前版本采用标准 softmax regression，并重点搜索更有效的文本特征与优化参数。

## Environment

```bash
uv sync
```

## Data

代码默认读取项目根目录下 `nlp-text-classification-experiments/` 中的以下文件：

- `train_new_5shot.csv`
- `dev_new.csv`
- `kaggle_test.csv`

## Run

所有命令在 `homework1/` 目录下执行：

```bash
cd homework1
```

### 1. Grid Search

搜索以下几类设置：

- 特征：`bow` / `ngram` / `tfidf` / `char_wb_tfidf`
- 优化器：`sgd` / `adam`
- 学习率：`0.1` / `0.03` / `0.01`
- L2：`0.0` / `1e-4` / `1e-3`
- batch size：`16` / `32`

结果会保存到 `results/grid_search_results.json` 和 `results/best_config.json`。

```bash
python main.py search
```

### 2. Train With Best Config

自动加载 `results/best_config.json`；如果不存在，则使用内置默认配置。

```bash
python main.py train
```

### 3. Predict

默认按 `train/dev` 划分训练并生成 `submission.csv`：

```bash
python main.py predict
```

如果需要在选定最优配置后使用 `train + dev` 重新训练，再对测试集预测：

```bash
python main.py predict --train-on-all
```

### 4. Compare Features

快速比较几种高收益特征配置的验证集准确率：

```bash
python main.py compare_features
```

## Outputs

- `results/best_config.json`：最优配置
- `results/grid_search_results.json`：完整搜索结果
- `results/model_params.json`：训练后的模型权重
- `submission.csv`：测试集预测结果
