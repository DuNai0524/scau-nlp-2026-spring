#!/bin/bash
# LLaMA-Factory 训练脚本

set -e

# 配置
LLAMAFACTORY_DIR="/root/LlamaFactory"
PROJECT_DIR="/root/nlp_homework_04"
MODEL_NAME="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  LLaMA-Factory 训练启动脚本${NC}"
echo -e "${GREEN}================================${NC}"

# 1. 检查LLaMA-Factory目录
if [ ! -d "$LLAMAFACTORY_DIR" ]; then
    echo -e "${RED}错误: LLaMA-Factory 目录不存在: $LLAMAFACTORY_DIR${NC}"
    exit 1
fi

echo -e "${YELLOW}✓ LLaMA-Factory 目录: $LLAMAFACTORY_DIR${NC}"

# 2. 激活环境
echo -e "\n${YELLOW}>>> 步骤1: 激活 uv 环境${NC}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# 3. 数据预处理
echo -e "\n${YELLOW}>>> 步骤2: 数据预处理${NC}"
cd "$PROJECT_DIR"
python prepare_data.py

# 4. 复制数据到LLaMA-Factory
echo -e "\n${YELLOW}>>> 步骤3: 复制数据到 LLaMA-Factory${NC}"
mkdir -p "$LLAMAFACTORY_DIR/data/intent_classification"
cp -r "$PROJECT_DIR/data/intent_classification/"* "$LLAMAFACTORY_DIR/data/intent_classification/"
echo -e "${GREEN}✓ 数据已复制到 LLaMA-Factory${NC}"
echo -e "${GREEN}  - 训练集: 1465 条${NC}"
echo -e "${GREEN}  - 验证集: 3200 条${NC}"
echo -e "${GREEN}  - 测试集: 4000 条${NC}"

# 5. 检查模型缓存
#echo -e "\n${YELLOW}>>> 步骤4: 检查模型缓存${NC}"
#if [ ! -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct" ]; then
#    echo -e "${YELLOW}! 模型未缓存，首次下载需要较长时间...${NC}"
#fi

# 6. 启动训练
echo -e "\n${YELLOW}>>> 步骤5: 启动 LoRA 微调${NC}"
echo -e "${GREEN}模型: $MODEL_NAME${NC}"
echo -e "${GREEN}配置: configs/qwen2.5_lora_sft.yaml${NC}"
echo -e "${GREEN}输出: saves/qwen2.5-7b/qlora/sft${NC}"
echo -e "${YELLOW}使用 QLoRA (4-bit 量化) 训练，显存占用约 8-12GB${NC}\n"

cd "$LLAMAFACTORY_DIR"

# 使用 llamafactory-cli 启动训练
llamafactory-cli train \
    "$PROJECT_DIR/configs/qwen2.5_lora_sft.yaml"

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}  训练完成！${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "${YELLOW}模型保存在: saves/qwen2.5-7b/qlora/sft${NC}"
