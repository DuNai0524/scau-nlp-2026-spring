"""
调试脚本：检查模型预测输出
"""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    base_model_path = '/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct'
    lora_path = '/root/LlamaFactory/saves/qwen2.5-7b/qlora/sft'

    # 加载类别
    with open('data/intent_classification/categories.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)

    category_list = "\n".join([f"{k}: {v}" for k, v in sorted(categories.items(), key=lambda x: int(x[0]))])

    print(f"🔄 正在加载模型: {base_model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载LoRA适配器
    if Path(lora_path).exists():
        print(f"🔄 正在加载 LoRA 适配器: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    print(f"✅ 模型加载完成")

    # 测试样本
    test_text = "您好高兴为您服务哎你好哎你好哎想把这个号码设置呼叫转移咋整您是所有来电都要转移吗那边您知道密码吧我不知道那您可以通过手机上来进行设置如果呼叫转移成功以后的话他是要收费的转接到别的号码上一分钟一毛钱呃我把这个设置的方式给您发过去一会您按照短信提示就能设置还有其他的方法也发给您随时您的也可以取消好谢谢啊不客气啊请问还有什么可以帮您吗没有"

    # 预测时的 prompt 格式 (当前使用的方式)
    predict_prompt = f"""请判断以下客服对话的意图类别。从以下34个类别中选择最匹配的一个，只输出类别编号（0-33）。

可选类别：
{category_list}

请只输出类别编号，不要输出类别名称或其他内容。

用户对话：{test_text}

类别编号："""

    # 训练时的 prompt 格式 (LLaMA-Factory alpaca 格式)
    train_instruction = f"""请判断以下客服对话的意图类别。从以下34个类别中选择最匹配的一个，只输出类别编号（0-33）。

可选类别：
{category_list}

请只输出类别编号，不要输出类别名称或其他内容。"""

    train_input = f"用户对话：{test_text}"

    # Qwen 的 chat 格式
    messages = [
        {"role": "system", "content": "你是一个意图分类助手。"},
        {"role": "user", "content": f"{train_instruction}\n\n{train_input}"}
    ]

    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n" + "="*50)
    print("测试1: 使用预测脚本的 prompt 格式")
    print("="*50)
    print(f"Prompt 长度: {len(predict_prompt)}")

    inputs = tokenizer(predict_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"模型输出: '{generated}'")

    print("\n" + "="*50)
    print("测试2: 使用 Qwen chat 格式 (训练时可能使用的格式)")
    print("="*50)
    print(f"Prompt 长度: {len(chat_prompt)}")

    inputs2 = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated2 = tokenizer.decode(outputs2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"模型输出: '{generated2}'")

    print("\n" + "="*50)
    print("测试3: 使用纯 instruction + input 格式")
    print("="*50)

    # 模拟 LLaMA-Factory 的 alpaca 格式
    alpaca_prompt = f"{train_instruction}\n\n{train_input}"
    print(f"Prompt 长度: {len(alpaca_prompt)}")

    inputs3 = tokenizer(alpaca_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs3 = model.generate(
            **inputs3,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated3 = tokenizer.decode(outputs3[0][inputs3['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"模型输出: '{generated3}'")


if __name__ == "__main__":
    main()
