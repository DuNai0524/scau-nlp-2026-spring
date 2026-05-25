# Homework 1: SLU Intent Detection

基于 softmax 回归的中文口语理解意图识别（numpy 实现，无深度学习框架）。

## 环境准备

```bash
# 在项目根目录安装依赖
uv sync
```

## 数据准备

将以下 CSV 文件放入 `data/` 目录：

- `train.csv` — 训练集
- `dev.csv` — 验证集
- `test.csv` — 测试集

## 运行命令

所有命令在 `homework1/` 目录下执行：

```bash
cd homework1
```

### 1. 网格搜索调参

搜索 3 种激活函数 x 2 种损失函数 x 2 种优化器 x 3 种学习率 x 5 种 dropout = 180 组超参数组合，结果保存到 `results/best_config.json`：

```bash
python main.py search
```

### 2. 使用最优配置训练

自动加载 `results/best_config.json`（若不存在则使用默认配置：relu + cross_entropy + adam, lr=0.01, dropout=0.1）：

```bash
python main.py train
```

### 3. 生成提交文件

训练模型并生成 `submission.csv`：

```bash
python main.py predict
```

### 4. 对比特征提取方法

在 BoW / N-gram / TF-IDF 三种特征下分别训练，比较验证集准确率：

```bash
python main.py compare_features
```

## 超参数搜索空间

| 维度         | 候选值                                      |
| ------------ | ------------------------------------------- |
| 激活函数     | `relu`, `leaky_relu`, `tanh`               |
| 损失函数     | `cross_entropy`, `mse`                      |
| 优化器       | `adam`, `sgd`（含 momentum）                |
| 学习率       | `0.1`, `0.01`, `0.001`                      |
| Dropout      | `0.0`, `0.1`, `0.2`, `0.3`, `0.5`          |

## 输出目录

- `results/best_config.json` — 搜索得到的最优超参数
- `results/model_params.json` — 训练后的模型权重
- `submission.csv` — 测试集预测结果
