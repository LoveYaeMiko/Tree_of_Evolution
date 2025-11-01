#!/usr/bin/env python3
"""
独立运行模型评估模块
"""

import argparse
import time
from pathlib import Path
from config import get_config
from model_training.evaluation import CodeEvaluator

def main():
    parser = argparse.ArgumentParser(description="独立模型评估")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["humaneval", "mbpp", "all"], default=["humaneval"],
                       help="评估基准")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.scale)
    
    print(f"开始模型评估 (规模: {args.scale})...")
    start_time = time.time()
    
    try:
        # 检查模型路径
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
        
        # 模型评估
        evaluator = CodeEvaluator(config)
        evaluator.load_model(args.model_path)
        
        benchmarks = args.benchmarks
        if "all" in benchmarks:
            benchmarks = ["humaneval", "mbpp"]
        
        results = {}
        for benchmark in benchmarks:
            print(f"\n评估基准: {benchmark}")
            if benchmark == "humaneval":
                result = evaluator.evaluate_on_humaneval()
            elif benchmark == "mbpp":
                result = evaluator.evaluate_on_mbpp()
            else:
                continue
            
            evaluator.save_evaluation_results(result, benchmark)
            results[benchmark] = result.get("average_metrics", {})
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 模型评估完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print("评估结果摘要:")
        for benchmark, metrics in results.items():
            print(f"  {benchmark}: {metrics}")
        
    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        raise

if __name__ == "__main__":
    main()