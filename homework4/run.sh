#!/bin/bash
# 完整的端到端运行脚本：数据准备 -> 训练 -> 预测

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="/root/nlp_homework_04"

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  口语对话理解 - 意图检测 (LLaMA-Factory)${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# 解析命令行参数
MODE="${1:-all}"  # all, prepare, train, predict, eval

function show_usage() {
    echo -e "${YELLOW}用法:${NC} ./run.sh [MODE]"
    echo ""
    echo -e "${BLUE}可选模式:${NC}"
    echo "  ${GREEN}prepare${NC}  - 仅执行数据预处理"
    echo "  ${GREEN}train${NC}    - 仅执行模型训练"
    echo "  ${GREEN}predict${NC}  - 仅执行推理预测"
    echo "  ${GREEN}eval${NC}     - 在验证集上评估模型"
    echo "  ${GREEN}all${NC}      - 执行完整流程 (默认)"
    echo ""
}

if [ "$MODE" == "help" ] || [ "$MODE" == "-h" ]; then
    show_usage
    exit 0
fi

echo -e "${YELLOW}当前模式: $MODE${NC}\n"

# 激活环境
echo -e "${BLUE}>>> 激活 Python 环境${NC}"
cd "$PROJECT_DIR"
source .venv/bin/activate
echo -e "${GREEN}✓ 环境已激活${NC}\n"

# ==================== 数据预处理 ====================
if [ "$MODE" == "all" ] || [ "$MODE" == "prepare" ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  步骤1: 数据预处理${NC}"
    echo -e "${BLUE}===============================================${NC}"
    python prepare_data.py
    echo ""
fi

# ==================== 模型训练 ====================
if [ "$MODE" == "all" ] || [ "$MODE" == "train" ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  步骤2: 模型训练${NC}"
    echo -e "${BLUE}===============================================${NC}"
    bash train.sh
    echo ""
fi

# ==================== 验证集评估 ====================
if [ "$MODE" == "eval" ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  验证集评估${NC}"
    echo -e "${BLUE}===============================================${NC}"
    
    # 检查是否有训练好的模型
    LORA_PATH="/root/LlamaFactory/saves/qwen2.5-7b/lora/sft"
    if [ ! -d "$LORA_PATH" ]; then
        echo -e "${RED}错误: 未找到训练好的模型: $LORA_PATH${NC}"
        echo -e "${YELLOW}请先运行: ./run.sh train${NC}"
        exit 1
    fi
    
    python predict.py \
        --test_file nlp-text-classification-experiments/dev_new.csv \
        --output dev_predictions.csv \
        --batch_size 8
    
    # 计算准确率
    echo -e "\n${YELLOW}正在计算验证集准确率...${NC}"
    python -c "
import pandas as pd
dev_df = pd.read_csv('nlp-text-classification-experiments/dev_new.csv')
pred_df = pd.read_csv('dev_predictions.csv')
dev_df['Id'] = range(1, len(dev_df) + 1)
merged = dev_df.merge(pred_df, on='Id', suffixes=('_true', '_pred'))
acc = (merged['c_numerical'] == merged['Category']).mean()
print(f'验证集准确率: {acc:.4f} ({acc*100:.2f}%)')
"
    echo ""
fi

# ==================== 测试集预测 ====================
if [ "$MODE" == "all" ] || [ "$MODE" == "predict" ]; then
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  步骤3: 测试集预测${NC}"
    echo -e "${BLUE}===============================================${NC}"
    
    # 检查是否有训练好的模型
    LORA_PATH="/root/LlamaFactory/saves/qwen2.5-7b/lora/sft"
    if [ ! -d "$LORA_PATH" ]; then
        echo -e "${RED}错误: 未找到训练好的模型: $LORA_PATH${NC}"
        echo -e "${YELLOW}请先运行: ./run.sh train${NC}"
        exit 1
    fi
    
    python predict.py \
        --test_file nlp-text-classification-experiments/kaggle_test.csv \
        --output submission.csv \
        --batch_size 8
    
    echo ""
    echo -e "${GREEN}===============================================${NC}"
    echo -e "${GREEN}  预测完成！${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo -e "${YELLOW}提交文件: submission.csv${NC}"
    echo ""
fi

# ==================== 完整流程完成 ====================
if [ "$MODE" == "all" ]; then
    echo -e "${GREEN}===============================================${NC}"
    echo -e "${GREEN}  所有步骤完成！${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo ""
    echo -e "${BLUE}生成的文件:${NC}"
    echo "  - 数据: data/intent_classification/"
    echo "  - 模型: /root/LlamaFactory/saves/qwen2.5-7b/lora/sft"
    echo "  - 预测: submission.csv"
    echo ""
    echo -e "${YELLOW}你可以提交 submission.csv 到 Kaggle${NC}"
fi
