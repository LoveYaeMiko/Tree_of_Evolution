#!/usr/bin/env python3
"""
独立运行模型训练模块
"""

import argparse
import time
from pathlib import Path
from config import get_config
from model_training.fine_tuning import CodeFineTuner

def main():
    parser = argparse.ArgumentParser(description="独立模型训练")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="自定义输出目录")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.scale)
    if args.output_dir:
        config.output_dir = args.output_dir
    
    print(f"开始模型训练 (规模: {args.scale})...")
    start_time = time.time()
    
    try:
        # 检查数据文件
        if not Path(args.data_path).exists():
            raise FileNotFoundError(f"训练数据文件不存在: {args.data_path}")
        
        # 模型微调
        fine_tuner = CodeFineTuner(config)
        train_dataset, val_dataset = fine_tuner.prepare_data(args.data_path)
        model_path = fine_tuner.fine_tune(train_dataset, val_dataset)
        
        # 保存训练信息
        fine_tuner.save_training_info(train_dataset, val_dataset, model_path)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 模型训练完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"模型保存在: {model_path}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        raise

if __name__ == "__main__":
    main()