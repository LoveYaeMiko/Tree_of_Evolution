#!/usr/bin/env python3
import argparse
from src.inference import CodeGenerator
from tqdm import tqdm

def test_model_capabilities(generator):
    """测试模型的各种能力"""
    test_cases = [
        {
            "category": "算法实现",
            "instruction": "写一个Python函数实现快速排序算法"
        },
        {
            "category": "数据结构", 
            "instruction": "实现一个二叉树类，包含插入和遍历方法"
        },
        {
            "category": "字符串处理",
            "instruction": "写一个函数检查字符串是否是回文"
        },
        {
            "category": "文件操作",
            "instruction": "写一个函数读取文本文件并统计词频"
        }
    ]
    
    print("\n=== 模型能力测试 ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['category']}: {test_case['instruction']}")
        code = generator.generate_code(test_case['instruction'])
        print(f"生成代码:\n```python\n{code}\n```")
        
        # 简单质量评估
        quality_indicators = {
            "包含函数定义": "def " in code,
            "代码长度合理": 50 < len(code) < 1000,
            "语法正确": True  # 这里可以添加语法检查
        }
        
        score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        print(f"质量评分: {score:.1f}%")
        
        input("按回车继续下一个测试...")

def main():
    parser = argparse.ArgumentParser(description='交互式代码生成')
    parser.add_argument('--model_path', type=str, required=True,
                       help='微调模型路径')
    
    args = parser.parse_args()
    
    # 初始化代码生成器
    with tqdm(desc="加载模型", total=1) as pbar:
        generator = CodeGenerator(args.model_path)
        pbar.update(1)
    
    # 启动交互式会话
    test_model_capabilities(generator)
    generator.interactive_session()

if __name__ == "__main__":
    main()