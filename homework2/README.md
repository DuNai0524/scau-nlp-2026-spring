# Homework 2: SLU Intent Detection with BERT

基于 `bert-base-chinese` 微调的中文口语理解意图识别，保持 `homework1` 的目录和输出风格，同时使用 Hugging Face Transformers 完成 BERT 训练与预测。

## Environment

```bash
uv sync
```

## Data

代码默认读取项目根目录下 `nlp-text-classification-experiments/` 中的以下文件：

- `train_new_5shot.csv`
- `dev_new.csv`
- `kaggle_test.csv`

模型输入使用 `sentence_sep` 列，并会把原始对话中的 `[SEP]` 标记保留为 BERT 分隔符语义。超过模型长度上限的样本会被截断到 512 tokens。

## Run

所有命令在 `homework2/` 目录下执行：

```bash
cd homework2
```

### 1. Train

在 `train` 上微调 `bert-base-chinese`，并在 `dev` 上按 epoch 评估、保存最优模型。

```bash
python main.py train
```

### 2. Predict

加载 `results/` 中已保存的模型与标签映射，生成 `submission.csv`。如果尚未执行训练，会直接报错并提示先运行训练命令。

```bash
python main.py predict
```

提交文件包含两列：

- `ID`：与 `kaggle_test.csv` 中的 `ID` 一一对应
- `c_numerical`：模型预测的意图类别数值编码

```csv
ID,c_numerical
test_00000,0
test_00001,3
```

## Outputs

- `results/model/`：最终保存的最优 BERT 模型与 tokenizer
- `results/checkpoints/`：Trainer 训练过程中保存的 checkpoint
- `results/training_config.json`：训练配置
- `results/label_metadata.json`：标签顺序与 `c_numerical` 映射
- `results/metrics.json`：训练与验证指标
- `submission.csv`：测试集预测结果
