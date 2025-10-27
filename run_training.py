#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
from src.model_training import ModelTrainer
from tqdm import tqdm
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    parser = argparse.ArgumentParser(description='训练Tree-of-Evolution模型')
    parser.add_argument('--config', type=str, default='configs/small.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='models/finetuned_model',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件 {args.data_path} 不存在")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    with tqdm(desc="初始化训练器", total=1) as pbar:
        trainer = ModelTrainer(config)
        pbar.update(1)
    
    trainer.train_model(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()