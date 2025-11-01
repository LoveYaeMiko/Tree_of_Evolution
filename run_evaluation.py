#!/usr/bin/env python3
"""
独立运行模型评估模块 - 修复codebleu问题
"""

import argparse
import time
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model_training.evaluation import CodeEvaluator

def install_codebleu_if_needed():
    """如果需要，安装codebleu"""
    try:
        # 尝试导入codebleu
        from codebleu import calc_codebleu
        print("✅ codebleu已安装")
        return True
    except ImportError:
        try:
            # 尝试从evaluate加载
            import evaluate
            evaluate.load("codebleu")
            print("✅ evaluate中的codebleu可用")
            return True
        except:
            print("⚠ codebleu不可用，尝试安装...")
            try:
                from install_codebleu import install_codebleu
                return install_codebleu()
            except:
                print("❌ 自动安装codebleu失败")
                return False

def fix_codebleu_issues():
    """修复codebleu问题"""
    try:
        from fix_codebleu_issues import main as fix_main
        fix_main()
        return True
    except Exception as e:
        print(f"❌ codebleu问题修复失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="独立模型评估")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["humaneval", "mbpp", "all"], default=["humaneval"],
                       help="评估基准")
    parser.add_argument("--install_codebleu", action="store_true",
                       help="安装codebleu")
    parser.add_argument("--fix_codebleu", action="store_true",
                       help="修复codebleu问题")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.scale)
    
    # 如果需要，修复codebleu问题
    if args.fix_codebleu:
        print("修复codebleu问题...")
        success = fix_codebleu_issues()
        if not success:
            print("❌ codebleu修复失败，评估将继续使用其他指标")
    
    # 如果需要，安装codebleu
    if args.install_codebleu:
        print("安装codebleu...")
        success = install_codebleu_if_needed()
        if not success:
            print("❌ codebleu安装失败，评估将继续使用其他指标")
    
    print(f"开始模型评估 (规模: {args.scale})...")
    start_time = time.time()
    
    try:
        # 检查模型路径
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {args.model_path}")
        
        # 检查codebleu是否可用
        install_codebleu_if_needed()
        
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
        import traceback
        traceback.print_exc()
        
        # 提供具体的解决方案
        print("\n🔧 具体解决方案:")
        print("1. 修复codebleu问题: python run_evaluation.py --fix_codebleu")
        print("2. 安装codebleu: python run_evaluation.py --install_codebleu")
        print("3. 手动安装: pip install codebleu")
        print("4. 或使用其他评估指标")
        
        raise

if __name__ == "__main__":
    main()