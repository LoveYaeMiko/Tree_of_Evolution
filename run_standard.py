#!/usr/bin/env python3
"""
标准规模版本运行脚本
"""

import os
import sys

def run_standard_scale():
    """运行标准规模版本"""
    print("运行标准规模版本...")
    
    # 设置标准规模参数
    cmd = (
        "python main.py "
        "--mode standard "
        "--download_models "
        "--download_datasets "
        "--evaluate_model "
        "--num_seeds 500 "
        "--evolution_rounds 3 "
        "--paths_per_node 3 "
        "--batch_size 8 "
        "--learning_rate 3e-6 "
        "--num_epochs 2"
    )
    
    print(f"执行命令: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    run_standard_scale()