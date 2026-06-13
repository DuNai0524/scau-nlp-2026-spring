"""
推理预测脚本：使用微调后的模型进行意图分类预测
支持类别名称到数字编号的自动转换
"""
import torch
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class IntentClassifier:
    """意图分类器"""

    def __init__(self, base_model_path, lora_path=None, device="cuda"):
        """
        初始化分类器

        Args:
            base_model_path: 基础模型路径
            lora_path: LoRA适配器路径（可选）
            device: 运行设备
        """
        self.device = device
        print(f"🔄 正在加载模型: {base_model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载LoRA适配器（如果提供）
        if lora_path and Path(lora_path).exists():
            print(f"🔄 正在加载 LoRA 适配器: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # 合并权重以获得更快推理

        self.model.eval()
        print(f"✅ 模型加载完成，设备: {self.device}")

    def create_prompt(self, text):
        """
        创建分类提示 - 与训练时的格式保持一致

        Args:
            text: 输入文本（带 [SEP] 分隔符的对话）

        Returns:
            格式化的prompt
        """
        instruction = "请判断下面客服对话的意图类别，只输出类别名称，不要输出解释。"

        # 提取关键词（去除 [SEP] 标记）
        keywords = text.replace('[SEP]', '')

        input_text = f"客服对话：\n{text}\n\n关键词：\n{keywords}"

        # 使用 Qwen chat 模板格式（与训练时一致）
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n{input_text}"}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def predict(self, text, max_new_tokens=20):
        """
        单条预测

        Args:
            text: 输入文本
            max_new_tokens: 最大生成token数

        Returns:
            预测的类别名称
        """
        prompt = self.create_prompt(text)

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 提取类别名称（去除可能的空格和换行）
        category_name = generated_text.strip()

        return category_name

    def predict_batch(self, texts, batch_size=8):
        """
        批量预测

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            预测结果列表（类别名称）
        """
        predictions = []

        for i in tqdm(range(0, len(texts), batch_size), desc="预测进度"):
            batch_texts = texts[i:i+batch_size]
            batch_preds = []

            for text in batch_texts:
                pred = self.predict(text)
                batch_preds.append(pred)

            predictions.extend(batch_preds)

        return predictions


def load_category_mapping(mapping_file):
    """
    加载类别映射

    Args:
        mapping_file: 类别映射JSON文件路径

    Returns:
        name_to_id: 类别名称 -> 数字编号
        id_to_name: 数字编号 -> 类别名称
    """
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    name_to_id = mapping_data['name_to_id']
    id_to_name = mapping_data['id_to_name']

    # 将 id_to_name 的 key 转换为 int
    id_to_name = {int(k): v for k, v in id_to_name.items()}

    return name_to_id, id_to_name


def convert_predictions_to_ids(predictions, name_to_id, id_to_name):
    """
    将类别名称预测结果转换为数字编号

    Args:
        predictions: 类别名称列表
        name_to_id: 类别名称 -> 数字编号映射
        id_to_name: 数字编号 -> 类别名称映射

    Returns:
        数字编号列表，转换失败为 -1
    """
    numeric_predictions = []
    failed_count = 0

    for pred in predictions:
        # 尝试直接匹配
        if pred in name_to_id:
            numeric_predictions.append(name_to_id[pred])
        else:
            # 尝试部分匹配（模型可能输出多余内容）
            matched = False
            for name in name_to_id.keys():
                if name in pred or pred in name:
                    numeric_predictions.append(name_to_id[name])
                    matched = True
                    break

            if not matched:
                # 尝试提取数字（如果模型输出了数字）
                numbers = re.findall(r'\d+', pred)
                if numbers:
                    num = int(numbers[0])
                    if num in id_to_name:
                        numeric_predictions.append(num)
                        matched = True

            if not matched:
                numeric_predictions.append(-1)
                failed_count += 1

    if failed_count > 0:
        print(f"⚠️ 警告: {failed_count} 个样本转换失败，将被标记为-1")

    return numeric_predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='意图分类推理')
    parser.add_argument('--base_model', type=str,
                        default='/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct',
                        help='基础模型路径')
    parser.add_argument('--lora_path', type=str,
                        default='/root/LlamaFactory/saves/qwen2.5-7b/qlora/sft',
                        help='LoRA适配器路径')
    parser.add_argument('--test_file', type=str,
                        default='nlp-text-classification-experiments/kaggle_test.csv',
                        help='测试集CSV文件')
    parser.add_argument('--mapping_file', type=str,
                        default='data/intent_classification/category_mapping.json',
                        help='类别映射JSON文件')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='输出预测结果文件')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小')

    args = parser.parse_args()

    # 检查文件
    if not Path(args.test_file).exists():
        print(f"❌ 错误: 测试文件不存在: {args.test_file}")
        return

    if not Path(args.mapping_file).exists():
        print(f"❌ 错误: 类别映射文件不存在: {args.mapping_file}")
        return

    # 加载类别映射
    print(f"📂 正在加载类别映射...")
    name_to_id, id_to_name = load_category_mapping(args.mapping_file)
    print(f"   发现 {len(name_to_id)} 个类别")

    # 加载测试数据
    print(f"📂 正在加载测试数据: {args.test_file}")
    test_df = pd.read_csv(args.test_file, encoding='utf-8-sig')
    print(f"   测试集大小: {len(test_df)}")

    # 初始化分类器
    classifier = IntentClassifier(
        base_model_path=args.base_model,
        lora_path=args.lora_path if Path(args.lora_path).exists() else None
    )

    # 批量预测
    print(f"\n🚀 开始预测...")

    # 使用 sentence_sep 列（如果有），否则使用 word_mf2
    if 'sentence_sep' in test_df.columns:
        texts = test_df['sentence_sep'].fillna(test_df['word_mf2']).tolist()
    else:
        texts = test_df['word_mf2'].tolist()

    # 预测类别名称
    predictions = classifier.predict_batch(texts, batch_size=args.batch_size)

    # 转换为数字编号
    print(f"\n🔄 转换类别名称为数字编号...")
    numeric_predictions = convert_predictions_to_ids(predictions, name_to_id, id_to_name)

    # 统计预测分布
    from collections import Counter
    pred_dist = Counter(numeric_predictions)
    print(f"\n📊 预测分布:")
    for k in sorted(pred_dist.keys()):
        if k == -1:
            print(f"   转换失败: {pred_dist[k]} 条")
        else:
            print(f"   类别 {k} ({id_to_name.get(k, 'Unknown')}): {pred_dist[k]} 条")

    # 保存结果
    # 读取sample_submission获取正确的ID
    sample_file = Path(args.test_file).parent / 'sample_submission.csv'
    if sample_file.exists():
        sample_df = pd.read_csv(sample_file, encoding='utf-8-sig')
        result_df = pd.DataFrame({
            'ID': sample_df['ID'],
            'c_numerical': numeric_predictions
        })
    else:
        result_df = pd.DataFrame({
            'ID': [f'test_{i:05d}' for i in range(len(numeric_predictions))],
            'c_numerical': numeric_predictions
        })

    result_df.to_csv(args.output, index=False)
    print(f"\n✅ 预测完成！结果已保存到: {args.output}")
    print(f"   总计: {len(result_df)} 条预测")


if __name__ == "__main__":
    main()
