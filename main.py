import os
import argparse
from data_download import DatasetDownloader
from data_synthesis import TreeOfEvolution
from train_model import CodeInstructionTrainer
from evaluation import CodeEvaluator
from model_manager import ModelManager
from config import get_config

def main():
    config = get_config()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    print("=" * 60)
    print("Tree-of-Evolution 代码生成框架")
    print("=" * 60)
    print(f"运行模式: {config.mode}")
    print(f"输出目录: {config.output_dir}")
    print(f"数据目录: {config.data_dir}")
    print("=" * 60)
    
    # 步骤0: 下载模型
    if config.download_models:
        print("\n步骤0: 下载模型")
        print("-" * 40)
        manager = ModelManager(config)
        manager.download_all_models(config.mode)
    
    # 步骤1: 下载数据集
    if config.download_datasets:
        print("\n步骤1: 下载数据集")
        print("-" * 40)
        downloader = DatasetDownloader(config)
        downloader.download_all_datasets()
    
    # 步骤2: 数据合成
    if not config.skip_synthesis:
        print("\n步骤2: 数据合成")
        print("-" * 40)
        toe = TreeOfEvolution(config)
        synthesized_data = toe.synthesize_data()
        print(f"数据合成完成！生成 {len(synthesized_data)} 条指令")
    
    # 步骤3: 模型训练
    if not config.skip_training:
        print("\n步骤3: 模型训练")
        print("-" * 40)
        trainer = CodeInstructionTrainer(config)
        trainer.train()
    
    # 步骤4: 模型评估
    if config.evaluate_model:
        print("\n步骤4: 模型评估")
        print("-" * 40)
        evaluator = CodeEvaluator(config)
        humaneval_score = evaluator.evaluate_humaneval(os.path.join(config.output_dir, "fine_tuned_model"))
        mbpp_score = evaluator.evaluate_mbpp(os.path.join(config.output_dir, "fine_tuned_model"))
        
        print("\n最终评估结果:")
        print(f"HumanEval Pass@1: {humaneval_score:.4f}")
        print(f"MBPP Pass@1: {mbpp_score:.4f}")
    
    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()