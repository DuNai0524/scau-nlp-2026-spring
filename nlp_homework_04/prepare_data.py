"""
数据预处理脚本：将数据转换为LLaMA-Factory训练格式
支持新的 JSON 训练数据和 CSV 验证/测试数据
"""
import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter


def load_category_mapping(train_csv_file):
    """
    从训练CSV加载类别名称和编号的双向映射

    Args:
        train_csv_file: 包含 label_raw 和 c_numerical 列的CSV文件

    Returns:
        name_to_id: 类别名称 -> 数字编号
        id_to_name: 数字编号 -> 类别名称
    """
    df = pd.read_csv(train_csv_file)

    # 构建映射
    name_to_id = {}
    id_to_name = {}

    for _, row in df.iterrows():
        cat_name = row['label_raw']
        cat_id = int(row['c_numerical'])

        # 去重，确保唯一映射
        if cat_name not in name_to_id:
            name_to_id[cat_name] = cat_id
        if cat_id not in id_to_name:
            id_to_name[cat_id] = cat_name

    print(f"✅ 加载了 {len(name_to_id)} 个类别映射")
    return name_to_id, id_to_name


def load_json_training_data(json_file):
    """
    加载新的JSON格式训练数据

    Args:
        json_file: JSON训练数据文件

    Returns:
        训练数据列表
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ 加载了 {len(data)} 条JSON训练数据")

    # 统计标签分布
    labels = [item['output'] for item in data]
    label_dist = Counter(labels)
    print(f"   类别分布: {len(label_dist)} 个类别")

    return data


def convert_csv_to_json_format(csv_file, name_to_id, split='dev'):
    """
    将CSV数据转换为JSON格式（与训练数据格式一致）

    Args:
        csv_file: CSV文件路径
        name_to_id: 类别名称到数字的映射
        split: 数据集类型 ('dev' 或 'test')

    Returns:
        转换后的数据列表
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    data = []

    for idx, row in df.iterrows():
        # 使用 sentence_sep 列（带 [SEP] 分隔符的对话）
        # 如果没有该列，使用 word_mf2
        if 'sentence_sep' in df.columns and pd.notna(row['sentence_sep']):
            text = str(row['sentence_sep'])
        else:
            text = str(row['word_mf2'])

        # 创建与训练数据一致的格式
        instruction = "请判断下面客服对话的意图类别，只输出类别名称，不要输出解释。"

        # 构建输入格式（与训练数据一致）
        # 提取关键词（简单处理：去除 [SEP] 标记）
        keywords = text.replace('[SEP]', '')

        input_text = f"客服对话：\n{text}\n\n关键词：\n{keywords}"

        if split in ['train', 'dev']:
            # 训练集和验证集有标签
            # 直接使用类别名称
            label_name = row['label_raw']
            output = label_name
        else:
            # 测试集无标签
            output = ""

        item = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }

        data.append(item)

    print(f"✅ 转换了 {len(data)} 条 {split} 数据为JSON格式")
    return data


def create_dataset_info(output_dir):
    """
    创建LLaMA-Factory的dataset_info.json
    使用 sharegpt 格式以支持 conversations 字段
    """
    dataset_info = {
        "intent_classification": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        },
        "intent_classification_dev": {
            "file_name": "dev.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        },
        "intent_classification_test": {
            "file_name": "test.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }

    output_file = os.path.join(output_dir, "dataset_info.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"✅ 已创建 dataset_info.json -> {output_file}")


def convert_to_sharegpt_format(data):
    """
    将 alpaca 格式转换为 sharegpt 格式
    LLaMA-Factory 对 Qwen 模型推荐使用 sharegpt 格式
    """
    sharegpt_data = []

    for item in data:
        # 构建对话格式
        conversations = [
            {
                "role": "user",
                "content": f"{item['instruction']}\n{item['input']}"
            },
            {
                "role": "assistant",
                "content": item['output']
            }
        ]

        sharegpt_data.append({"conversations": conversations})

    return sharegpt_data


def save_json(data, output_file):
    """保存JSON数据"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 {len(data)} 条数据 -> {output_file}")


def main():
    # 路径配置
    base_dir = Path(__file__).parent
    data_dir = base_dir / "nlp-text-classification-experiments"
    output_dir = base_dir / "data" / "intent_classification"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📊 开始数据预处理")
    print("=" * 60)

    # 1. 加载类别映射
    print("\n🔄 步骤1: 加载类别映射...")
    name_to_id, id_to_name = load_category_mapping(
        data_dir / "train_augmented.csv"
    )

    # 保存类别映射供后续使用
    mapping_file = output_dir / "category_mapping.json"
    mapping_data = {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name
    }
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存类别映射 -> {mapping_file}")

    # 2. 加载新的JSON训练数据
    print("\n🔄 步骤2: 加载新的JSON训练数据...")
    train_data = load_json_training_data(
        data_dir / "intent_train_labelraw_aug10.json"
    )

    # 转换为 sharegpt 格式
    train_sharegpt = convert_to_sharegpt_format(train_data)
    save_json(train_sharegpt, output_dir / "train.json")

    # 3. 转换验证集
    print("\n🔄 步骤3: 转换验证集...")
    dev_data = convert_csv_to_json_format(
        data_dir / "dev_new.csv",
        name_to_id,
        split='dev'
    )

    # 转换为 sharegpt 格式
    dev_sharegpt = convert_to_sharegpt_format(dev_data)
    save_json(dev_sharegpt, output_dir / "dev.json")

    # 4. 转换测试集
    print("\n🔄 步骤4: 转换测试集...")
    test_data = convert_csv_to_json_format(
        data_dir / "kaggle_test.csv",
        name_to_id,
        split='test'
    )

    # 转换为 sharegpt 格式
    test_sharegpt = convert_to_sharegpt_format(test_data)
    save_json(test_sharegpt, output_dir / "test.json")

    # 5. 创建dataset_info.json
    print("\n🔄 步骤5: 创建dataset_info.json...")
    create_dataset_info(output_dir)

    # 6. 数据统计
    print("\n" + "=" * 60)
    print("📊 数据统计")
    print("=" * 60)
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    print(f"类别数: {len(name_to_id)} 个")

    print("\n" + "=" * 60)
    print("🎉 数据预处理完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"  - 训练集: train.json")
    print(f"  - 验证集: dev.json")
    print(f"  - 测试集: test.json")
    print(f"  - 类别映射: category_mapping.json")
    print(f"  - 数据集配置: dataset_info.json")


if __name__ == "__main__":
    main()
