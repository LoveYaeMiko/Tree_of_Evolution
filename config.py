import argparse
import os
import sys

def get_config():
    parser = argparse.ArgumentParser(description="Tree-of-Evolution Configuration")
    
    # 数据合成参数
    parser.add_argument("--mode", type=str, default="small", choices=["small", "standard"], 
                       help="运行模式: small(小规模) 或 standard(标准规模)")
    parser.add_argument("--num_seeds", type=int, default=100, help="初始种子数量")
    parser.add_argument("--evolution_rounds", type=int, default=2, help="进化轮数")
    parser.add_argument("--paths_per_node", type=int, default=2, help="每个节点的进化路径数")
    parser.add_argument("--beam_size", type=float, default=0.75, help="束搜索保留比例")
    parser.add_argument("--diversity_threshold", type=float, default=6.0, help="多样性阈值")
    
    # 模型参数
    parser.add_argument("--base_model", type=str, default="", help="基础模型路径（留空使用自动配置）")
    parser.add_argument("--synthesis_model", type=str, default="", help="数据合成模型路径")
    parser.add_argument("--embed_model", type=str, default="", help="嵌入模型路径")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    
    # 系统参数
    parser.add_argument("--download_models", action="store_true", help="是否下载模型")
    parser.add_argument("--download_datasets", action="store_true", help="是否下载数据集")
    parser.add_argument("--evaluate_model", action="store_true", help="是否评估模型")
    parser.add_argument("--force_download", action="store_true", help="强制重新下载")
    parser.add_argument("--skip_synthesis", action="store_true", help="跳过数据合成")
    parser.add_argument("--skip_training", action="store_true", help="跳过模型训练")
    
    # 路径参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    
    args = parser.parse_args()
    
    # 根据模式调整参数
    if args.mode == "standard":
        args.num_seeds = 500
        args.evolution_rounds = 3
        args.paths_per_node = 3
        args.batch_size = 16
        args.base_model = ""
        args.learning_rate = 3e-6
    
    # 设置默认路径
    if not args.base_model:
        try:
            from model_manager import ModelManager
            manager = ModelManager(args)
            model_config = manager.load_model_config()
            if model_config:
                args.base_model = model_config.get("base_model", "")
                args.synthesis_model = model_config.get("synthesis_model", "")
                args.embed_model = model_config.get("embed_model", "")
            else:
                print("警告: 未找到模型配置，请先运行 --download_models")
        except ImportError:
            print("警告: 无法导入 ModelManager")
    
    return args

if __name__ == "__main__":
    config = get_config()
    print("配置参数:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")