#!/usr/bin/env python3
"""
独立运行数据合成模块
"""

import argparse
import time
from config import get_config
from data_synthesis.tree_evolution import TreeEvolution
from data_synthesis.data_collection import DataCollector

def main():
    parser = argparse.ArgumentParser(description="独立数据合成")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="自定义输出目录")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.scale)
    if args.output_dir:
        config.output_dir = args.output_dir
    
    print(f"开始数据合成 (规模: {args.scale})...")
    start_time = time.time()
    
    try:
        # 1. 树进化合成
        print("步骤1: 树进化合成...")
        synthesizer = TreeEvolution(config)
        synthesized_data = synthesizer.synthesize_data()
        synthesizer.save_synthesized_data(synthesized_data)
        
        # 2. 数据收集
        print("步骤2: 数据收集...")
        collector = DataCollector(config)
        instruction_response_pairs = collector.collect_from_trees(synthesized_data)
        collector.save_instruction_response_pairs(instruction_response_pairs)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 数据合成完成! 耗时: {elapsed_time/60:.2f} 分钟")
        
    except Exception as e:
        print(f"❌ 数据合成失败: {e}")
        raise

if __name__ == "__main__":
    main()