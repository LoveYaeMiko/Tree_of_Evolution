# run_paper_evaluation_fixed.py
#!/usr/bin/env python3
"""
修复版的与论文一致的评估脚本
"""

import argparse
import time
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model_training.paper_evaluation import PaperEvaluatorFixed

def main():
    parser = argparse.ArgumentParser(description="修复版的与论文一致的评估")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--model_path", type=str, required=True,
                       help="微调模型路径")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基模型名称（如果不指定则使用配置中的base_model）")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["humaneval", "mbpp", "all"], default=["humaneval"],
                       help="评估基准")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.scale)
    
    # 如果指定了基模型，则覆盖配置
    if args.base_model:
        config.base_model = args.base_model
    
    print(f"开始修复版的与论文一致的评估 (规模: {args.scale})...")
    print(f"基模型: {config.base_model}")
    print(f"微调模型: {args.model_path}")
    
    start_time = time.time()
    
    try:
        # 检查模型路径
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"微调模型路径不存在: {args.model_path}")
        
        # 创建评估器
        evaluator = PaperEvaluatorFixed(config)
        
        # 加载基模型和微调模型
        print("加载基模型...")
        evaluator.load_base_model(config.base_model)
        print("加载微调模型...")
        evaluator.load_model(args.model_path)
        
        benchmarks = args.benchmarks
        if "all" in benchmarks:
            benchmarks = ["humaneval", "mbpp"]
        
        all_results = {}
        
        for benchmark in benchmarks:
            print(f"\n{'='*50}")
            print(f"评估基准: {benchmark}")
            print(f"{'='*50}")
            
            # 评估基模型
            print(f"\n1. 评估基模型...")
            if benchmark == "humaneval":
                base_results = evaluator.evaluate_humaneval(use_base_model=True)
            elif benchmark == "mbpp":
                base_results = evaluator.evaluate_mbpp(use_base_model=True)
            else:
                continue
            
            # 评估微调模型
            print(f"\n2. 评估微调模型...")
            if benchmark == "humaneval":
                fine_tuned_results = evaluator.evaluate_humaneval(use_base_model=False)
            elif benchmark == "mbpp":
                fine_tuned_results = evaluator.evaluate_mbpp(use_base_model=False)
            else:
                continue
            
            # 保存对比结果
            evaluator.save_comparison_results(base_results, fine_tuned_results, benchmark)
            
            all_results[benchmark] = {
                "base": base_results["pass@1"],
                "fine_tuned": fine_tuned_results["pass@1"],
                "improvement": fine_tuned_results["pass@1"] - base_results["pass@1"]
            }
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print("🎉 修复版的与论文一致的评估完成!")
        print(f"总耗时: {elapsed_time/60:.2f} 分钟")
        print(f"{'='*50}")
        
        # 打印最终摘要
        print("\n📊 最终评估结果摘要:")
        for benchmark, results in all_results.items():
            print(f"  {benchmark.upper()}:")
            print(f"    基模型: {results['base']*100:.2f}%")
            print(f"    微调模型: {results['fine_tuned']*100:.2f}%")
            print(f"    改进: {results['improvement']*100:+.2f}%")
        
        print(f"\n结果保存在: {Path(config.output_dir) / 'paper_evaluation'}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供具体的解决方案
        print("\n🔧 具体解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 确保基模型可以访问")
        print("3. 检查数据集是否可用")
        
        raise

if __name__ == "__main__":
    main()