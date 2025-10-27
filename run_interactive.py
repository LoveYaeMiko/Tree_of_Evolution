#!/usr/bin/env python3
import argparse
from src.inference import CodeGenerator
from tqdm import tqdm

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
    generator.interactive_session()

if __name__ == "__main__":
    main()