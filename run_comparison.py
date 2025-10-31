#!/usr/bin/env python3
import argparse
from run_evaluation import evaluate_on_humaneval

def compare_models():
    """对比基础模型和微调模型"""
    parser = argparse.ArgumentParser(description='对比模型效果')
    parser.add_argument('--base_model', type=str, required=True, help='基础模型路径')
    parser.add_argument('--finetuned_model', type=str, required=True, help='微调模型路径')
    parser.add_argument('--num_samples', type=int, default=10, help='评估样本数')
    
    args = parser.parse_args()
    
    print("=== 模型对比评估 ===")
    
    print("\n1. 评估基础模型:")
    base_accuracy = evaluate_on_humaneval(args.base_model, args.num_samples)
    
    print("\n2. 评估微调模型:")
    finetuned_accuracy = evaluate_on_humaneval(args.finetuned_model, args.num_samples)
    
    print("\n=== 对比结果 ===")
    improvement = finetuned_accuracy - base_accuracy
    print(f"基础模型准确率: {base_accuracy:.1f}%")
    print(f"微调模型准确率: {finetuned_accuracy:.1f}%")
    print(f"提升: {improvement:+.1f}%")

if __name__ == "__main__":
    compare_models()