#!/usr/bin/env python3
import os
import sys
import yaml
import json
import argparse
from src.tree_evolution import TreeEvolution

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_seed_codes(seed_file):
    """加载种子代码"""
    if os.path.exists(seed_file):
        with open(seed_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 生成示例种子代码
        example_seeds = [
            "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def binary_search(arr, target):\n    low, high = 0, len(arr)-1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"
        ]
        return example_seeds

def main():
    parser = argparse.ArgumentParser(description='运行Tree-of-Evolution数据合成')
    parser.add_argument('--config', type=str, default='configs/small.yaml', 
                       help='配置文件路径')
    parser.add_argument('--seed_file', type=str, default='data/seeds/the_stack_seeds.txt',
                       help='种子代码文件路径')
    parser.add_argument('--output', type=str, default='data/synthesized/training_data.json',
                       help='输出文件路径')
    parser.add_argument('--seed_count', type=int, default=None,
                       help='使用的种子数量')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载种子代码
    seed_codes = load_seed_codes(args.seed_file)
    if args.seed_count:
        seed_codes = seed_codes[:args.seed_count]
    
    print(f"使用 {len(seed_codes)} 个种子代码")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 运行树进化 - 移除了外部进度条
    print("初始化进化器...")
    evolution = TreeEvolution(config)
    synthesized_data = evolution.run_evolution(seed_codes)  # 方法名已修复
    
    # 保存数据
    evolution.save_synthesized_data(args.output)
    print(f"合成数据已保存到 {args.output}")
    
    evolution.analyze_synthesized_data()
    
    # 打印统计信息
    if synthesized_data:
        challenge_scores = [item['challenge_score'] for item in synthesized_data]
        diversity_scores = [item['diversity_score'] for item in synthesized_data]
        
        print(f"\n数据统计:")
        print(f"总指令数: {len(synthesized_data)}")
        print(f"平均挑战性分数: {sum(challenge_scores)/len(challenge_scores):.2f}")
        print(f"平均多样性分数: {sum(diversity_scores)/len(diversity_scores):.2f}")
        print(f"平均质量分数: {(sum(challenge_scores) + sum(diversity_scores))/len(synthesized_data):.2f}")
    else:
        print("警告: 没有生成任何数据")

if __name__ == "__main__":
    main()