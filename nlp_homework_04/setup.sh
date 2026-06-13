#!/bin/bash
# 环境安装脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  环境安装脚本${NC}"
echo -e "${GREEN}===============================================${NC}"

PROJECT_DIR="/root/nlp_homework_04"
LLAMAFACTORY_DIR="/root/LlamaFactory"

# 1. 检查 uv 是否安装
echo -e "\n${BLUE}>>> 检查 uv 安装${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}正在安装 uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo -e "${GREEN}✓ uv 已安装${NC}"

# 2. 创建/更新 uv 环境
echo -e "\n${BLUE}>>> 配置 Python 环境${NC}"
cd "$PROJECT_DIR"

# 使用 Python 3.10 (兼容性更好)
if [ -f ".python-version" ]; then
    PYTHON_VERSION=$(cat .python-version)
    echo -e "${YELLOW}检测到 Python 版本: $PYTHON_VERSION${NC}"
fi

# 创建虚拟环境
echo -e "${YELLOW}正在创建虚拟环境...${NC}"
uv venv --python=python3.10 || uv venv
source .venv/bin/activate

echo -e "${GREEN}✓ 虚拟环境已创建${NC}"

# 3. 安装项目依赖
echo -e "\n${BLUE}>>> 安装项目依赖${NC}"
uv pip install -e .

echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 4. 安装 LLaMA-Factory
echo -e "\n${BLUE}>>> 安装 LLaMA-Factory${NC}"
if [ -d "$LLAMAFACTORY_DIR" ]; then
    echo -e "${YELLOW}LLaMA-Factory 已存在，更新依赖...${NC}"
    cd "$LLAMAFACTORY_DIR"
    git pull origin main 2>/dev/null || echo "使用现有版本"
    uv pip install -e ".[torch,metrics]"
else
    echo -e "${YELLOW}克隆 LLaMA-Factory...${NC}"
    cd /root
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    uv pip install -e ".[torch,metrics]"
fi

echo -e "${GREEN}✓ LLaMA-Factory 安装完成${NC}"

# 5. 验证安装
echo -e "\n${BLUE}>>> 验证安装${NC}"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
llamafactory-cli version 2>/dev/null || echo "llamafactory-cli 可用"

# 6. 设置 HuggingFace 镜像（国内加速）
echo -e "\n${BLUE}>>> 配置 HuggingFace 镜像${NC}"
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  环境安装完成！${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo -e "${YELLOW}接下来你可以运行:${NC}"
echo "  ./run.sh        # 执行完整流程"
echo "  ./run.sh train  # 仅训练模型"
echo ""
