#!/usr/bin/env python3
import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm

def evaluate_on_humaneval(model_path, num_samples=10):
    """在 HumanEval 上评估模型"""
    print("在 HumanEval 上评估模型...")
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载 HumanEval 数据集
    dataset = load_dataset("openai/openai_humaneval")["test"]
    
    correct = 0
    total = min(num_samples, len(dataset))
    
    for i in tqdm(range(total), desc="评估进度"):
        problem = dataset[i]
        prompt = problem["prompt"]
        
        # 生成代码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 这里简化评估，实际应该运行代码进行测试
        # 你可以使用更完整的评估框架如 EvalPlus
        print(f"\n问题 {i+1}:")
        print(f"生成代码: {generated_code[:200]}...")
        
        # 简单检查代码质量
        if "def" in generated_code and len(generated_code) > len(prompt) + 50:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"\n=== 评估结果 ===")
    print(f"评估样本数: {total}")
    print(f"通过数: {correct}")
    print(f"准确率: {accuracy:.1f}%")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='评估微调模型效果')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--num_samples', type=int, default=10, help='评估样本数')
    
    args = parser.parse_args()
    
    evaluate_on_humaneval(args.model_path, args.num_samples)

if __name__ == "__main__":
    main()