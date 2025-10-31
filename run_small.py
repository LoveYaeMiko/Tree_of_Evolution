#!/usr/bin/env python3
"""
小规模版本运行脚本 (1×4090，10小时内完成)
"""

import os
import sys

def run_small_scale():
    """运行小规模版本"""
    print("运行小规模版本 (1×4090，10小时内)...")
    
    # 设置小规模参数
    cmd = (
        "python main.py "
        "--mode small "
        "--download_models "
        "--download_datasets "
        "--evaluate_model "
        "--num_seeds 100 "
        "--evolution_rounds 2 "
        "--paths_per_node 2 "
        "--batch_size 16 "
        "--learning_rate 5e-6 "
        "--num_epochs 2"
    )
    
    print(f"执行命令: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    run_small_scale()