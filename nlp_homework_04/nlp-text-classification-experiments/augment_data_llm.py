"""
LLM-based Data Augmentation for Customer Service Intent Classification
使用大语言模型进行客服对话意图分类的数据增强

特点：
1. 解决类别不平衡问题
2. 保持口语化风格
3. 多样化的改写策略
4. 实体替换增强泛化能力

支持平台：阿里云百炼 (Bailian) - OpenAI兼容模式
"""

import pandas as pd
import json
import os
import re
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import time

# 使用 OpenAI 兼容模式调用百炼
try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai: pip install openai")
    exit(1)


# ==================== 配置 ====================
CONFIG = {
    "input_file": "train_new_5shot.csv",
    "output_file": "train_augmented.csv",
    "target_samples_per_class": 20,  # 每类目标样本数
    "min_samples_for_rare_class": 5,  # 少样本类别的最低数量

    # 百炼API配置 (OpenAI兼容模式)
    "model": "qwen-plus",  # 可选: qwen-turbo, qwen-plus, qwen-max, qwen3-72b-instruct 等
    "api_key_env": "DASHSCOPE_API_KEY",  # 环境变量名
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 百炼兼容模式endpoint

    # 增强策略配置
    "augmentation_strategies": [
        "intent_equivalent",     # 意图等价改写
        "entity_substitution",   # 实体替换
        "colloquial_variation",  # 口语化变体
        "dialogue_truncation",   # 对话截断
    ],

    # 每种策略每条原始数据生成的数量
    "augments_per_strategy": {
        "intent_equivalent": 2,
        "entity_substitution": 1,
        "colloquial_variation": 1,
        "dialogue_truncation": 1,
    },

    # API调用配置
    "max_retries": 3,
    "retry_delay": 2,
    "temperature": 0.8,
}


# ==================== Prompt模板 ====================
PROMPTS = {
    "intent_equivalent": """你是一个中国移动客服对话数据增强专家。请对以下客服对话进行【意图等价改写】。

要求：
1. 保持核心意图完全不变（类别标签不变）
2. 用不同的表达方式重新组织语言
3. 保持口语化、自然的客服对话风格
4. 可以调整句式（陈述句↔疑问句）
5. 可以替换同义词（如"办理"→"申请"，"取消"→"关掉"）
6. 保持对话的完整性和流畅性

原始对话：
{dialogue}

意图类别：{label}

请生成{num_augments}个不同的改写版本，格式如下：
【改写1】
对话内容...

【改写2】
对话内容...
""",

    "entity_substitution": """你是一个中国移动客服对话数据增强专家。请对以下客服对话进行【实体替换】。

要求：
1. 保持对话结构和意图完全不变
2. 替换具体的实体信息：
   - 电话号码（如13812345678→15987654321）
   - 金额（如30元→50元）
   - 流量数（如500兆→1G）
   - 套餐名（如18元套餐→28元套餐）
   - 地名（如长春→吉林）
   - 时间（如下个月→下个季度）
3. 保持替换后的数值合理、符合实际情况
4. 保持口语化风格

原始对话：
{dialogue}

意图类别：{label}

请生成{num_augments}个实体替换版本，格式如下：
【替换1】
对话内容...

【替换2】
对话内容...
""",

    "colloquial_variation": """你是一个中国移动客服对话数据增强专家。请对以下客服对话进行【口语化变体】。

要求：
1. 保持意图完全不变
2. 增加或调整口语化表达：
   - 添加语气词（如"呃"、"那个"、"就是"、"哎"）
   - 添加填充词（如"能不能"、"麻烦你"、"帮我看下"）
   - 模拟ASR转录风格（如"幺"→"1"、"咋"→"怎么"）
   - 调整语速感（急促/缓慢）
3. 保持对话自然流畅
4. 不要过度添加，保持真实感

原始对话：
{dialogue}

意图类别：{label}

请生成{num_augments}个口语化变体版本，格式如下：
【变体1】
对话内容...

【变体2】
对话内容...
""",

    "dialogue_truncation": """你是一个中国移动客服对话数据增强专家。请对以下客服对话进行【对话精简/扩展】。

要求：
1. 保持核心意图不变
2. 可以进行以下操作之一：
   - 精简：删除冗余对话，保留核心交互
   - 扩展：增加确认环节、礼貌用语等
   - 截断：只保留前半部分关键对话
3. 确保修改后的对话仍然完整、合理
4. 保持口语化风格

原始对话：
{dialogue}

意图类别：{label}

请生成{num_augments}个变体版本，格式如下：
【变体1】
对话内容...

【变体2】
对话内容...
""",

    "rare_class_augment": """你是一个中国移动客服对话数据增强专家。
当前类别【{label}】的训练样本非常少，请基于以下示例生成更多样本。

已有示例：
{examples}

要求：
1. 生成与示例意图一致的新对话
2. 对话场景要多样化：
   - 不同的用户语气（急躁/温和/疑惑）
   - 不同的具体业务细节
   - 不同的对话长度
3. 保持中国移动客服对话的真实风格
4. 每个生成样本要独立、有区分度

请生成{num_augments}个新的对话样本，格式如下：
【样本1】
对话内容...

【样本2】
对话内容...
""",
}


class DataAugmenter:
    """数据增强器 - 使用阿里云百炼API (OpenAI兼容模式)"""

    def __init__(self, config: Dict):
        self.config = config

        # 获取API Key
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            raise ValueError(f"请设置环境变量 {config['api_key_env']}")

        # 初始化OpenAI客户端 (指向百炼endpoint)
        self.client = OpenAI(
            api_key=api_key,
            base_url=config["base_url"]
        )

        print(f"使用模型: {config['model']}")
        print(f"API Endpoint: {config['base_url']}")

    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        df = pd.read_csv(self.config["input_file"])
        print(f"加载数据: {len(df)} 条, {df['label_raw'].nunique()} 个类别")
        return df

    def analyze_distribution(self, df: pd.DataFrame) -> Dict:
        """分析类别分布"""
        dist = df['label_raw'].value_counts().to_dict()
        print("\n类别分布:")
        for label, count in sorted(dist.items(), key=lambda x: x[1]):
            status = "⚠️ 稀少" if count < 3 else ("⚠️ 较少" if count < 5 else "✓ 正常")
            print(f"  {label}: {count} 条 {status}")
        return dist

    def call_llm(self, prompt: str) -> str:
        """调用百炼LLM API (OpenAI兼容模式)"""
        max_retries = self.config.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的客服对话数据增强助手，擅长生成自然、多样的中文客服对话。你的输出应该只包含生成的对话内容，不要添加额外的解释或注释。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.config.get("temperature", 0.8),
                    max_tokens=2000,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                delay = self.config.get("retry_delay", 2) * (2 ** attempt)
                time.sleep(delay)

        return ""

    def parse_augmentations(self, response: str) -> List[str]:
        """解析LLM返回的增强数据"""
        augments = []
        # 匹配【改写1】【替换1】【变体1】【样本1】等格式
        pattern = r'【[^】]+】\s*\n(.*?)(?=【[^】]+】|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            text = match.strip()
            if len(text) > 20:  # 过滤太短的结果
                augments.append(text)

        return augments

    def augment_sample(self, dialogue: str, label: str, strategy: str, num_augments: int = 2) -> List[str]:
        """对单个样本进行增强"""
        prompt_template = PROMPTS.get(strategy)
        if not prompt_template:
            print(f"未知策略: {strategy}")
            return []

        prompt = prompt_template.format(
            dialogue=dialogue,
            label=label,
            num_augments=num_augments
        )

        response = self.call_llm(prompt)
        augments = self.parse_augmentations(response)

        return augments[:num_augments]  # 限制数量

    def augment_rare_class(self, df: pd.DataFrame, label: str, target_count: int) -> List[Dict]:
        """对稀有类别进行批量增强"""
        samples = df[df['label_raw'] == label]['word_mf2'].tolist()
        current_count = len(samples)

        needed = target_count - current_count
        if needed <= 0:
            return []

        print(f"\n增强稀有类别 [{label}]: 当前{current_count}条, 需要增加{needed}条")

        # 构建示例文本
        examples = "\n\n".join([f"示例{i+1}:\n{s}" for i, s in enumerate(samples)])

        prompt = PROMPTS["rare_class_augment"].format(
            label=label,
            examples=examples,
            num_augments=needed
        )

        response = self.call_llm(prompt)
        augments = self.parse_augmentations(response)

        # 构建新样本
        new_samples = []
        for aug in augments[:needed]:
            new_samples.append({
                'word_mf2': aug,
                'label_raw': label,
                'c_numerical': 0,  # 占位符
                'num_cnum': 0,
                'sentence_sep': aug.replace('，', '[SEP]').replace('。', '[SEP]').replace('？', '[SEP]').replace('！', '[SEP]'),
            })

        return new_samples

    def augment_normal_class(self, df: pd.DataFrame, label: str, target_count: int) -> List[Dict]:
        """对普通类别进行增强"""
        samples = df[df['label_raw'] == label]['word_mf2'].tolist()
        current_count = len(samples)

        needed = target_count - current_count
        if needed <= 0:
            return []

        print(f"\n增强类别 [{label}]: 当前{current_count}条, 需要增加{needed}条")

        new_samples = []
        strategies = self.config["augmentation_strategies"]

        # 计算每个样本需要生成多少增强
        aug_per_sample = needed // len(samples) + 1

        for sample in tqdm(samples, desc=f"处理 {label[:10]}..."):
            # 随机选择策略
            strategy = random.choice(strategies)
            num_gen = min(aug_per_sample, self.config["augments_per_strategy"].get(strategy, 1))

            augments = self.augment_sample(sample, label, strategy, num_gen)

            for aug in augments:
                new_samples.append({
                    'word_mf2': aug,
                    'label_raw': label,
                    'c_numerical': 0,
                    'num_cnum': 0,
                    'sentence_sep': aug.replace('，', '[SEP]').replace('。', '[SEP]').replace('？', '[SEP]').replace('！', '[SEP]'),
                })

                if len(new_samples) >= needed:
                    break

            if len(new_samples) >= needed:
                break

        return new_samples[:needed]

    def run(self):
        """运行完整的增强流程"""
        print("=" * 60)
        print("LLM 数据增强开始")
        print("=" * 60)

        # 1. 加载数据
        df = self.load_data()

        # 2. 分析分布
        distribution = self.analyze_distribution(df)

        # 3. 确定目标数量
        target = self.config["target_samples_per_class"]
        min_rare = self.config["min_samples_for_rare_class"]

        # 4. 对每个类别进行增强
        all_new_samples = []

        for label, count in distribution.items():
            if count < min_rare:
                # 稀有类别，优先处理
                target_for_class = max(min_rare, target)
                new_samples = self.augment_rare_class(df, label, target_for_class)
            elif count < target:
                # 普通类别
                new_samples = self.augment_normal_class(df, label, target)
            else:
                new_samples = []

            all_new_samples.extend(new_samples)

        # 5. 合并并保存
        if all_new_samples:
            new_df = pd.DataFrame(all_new_samples)
            augmented_df = pd.concat([df, new_df], ignore_index=True)

            # 保存
            augmented_df.to_csv(self.config["output_file"], index=False)

            print("\n" + "=" * 60)
            print("增强完成!")
            print(f"原始样本: {len(df)} 条")
            print(f"新增样本: {len(all_new_samples)} 条")
            print(f"最终样本: {len(augmented_df)} 条")
            print(f"输出文件: {self.config['output_file']}")
            print("=" * 60)

            # 显示新的分布
            print("\n增强后类别分布:")
            new_dist = augmented_df['label_raw'].value_counts().to_dict()
            for label, count in sorted(new_dist.items()):
                print(f"  {label}: {count} 条")
        else:
            print("无需增强，数据已满足目标")


# ==================== 入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM数据增强 - 客服对话意图分类")
    parser.add_argument("--input", type=str, default=CONFIG["input_file"], help="输入文件")
    parser.add_argument("--output", type=str, default=CONFIG["output_file"], help="输出文件")
    parser.add_argument("--target", type=int, default=CONFIG["target_samples_per_class"], help="每类目标样本数")
    parser.add_argument("--model", type=str, default=CONFIG["model"],
                        help="百炼模型名称: qwen-turbo, qwen-plus, qwen-max, qwen3-72b-instruct 等")
    args = parser.parse_args()

    # 更新配置
    CONFIG["input_file"] = args.input
    CONFIG["output_file"] = args.output
    CONFIG["target_samples_per_class"] = args.target
    CONFIG["model"] = args.model

    print("=" * 60)
    print("阿里云百炼 - 数据增强")
    print("=" * 60)
    print(f"输入文件: {CONFIG['input_file']}")
    print(f"输出文件: {CONFIG['output_file']}")
    print(f"目标数量: {CONFIG['target_samples_per_class']} 条/类")
    print(f"使用模型: {CONFIG['model']}")
    print("=" * 60)

    # 运行
    augmenter = DataAugmenter(CONFIG)
    augmenter.run()
