from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class SmallScaleConfig:
    """小规模配置 (1块4090, 10小时内完成)"""
    # 数据合成
    num_seeds: int = 100  # 种子数量
    tree_depth: int = 2   # 树深度
    beam_width: int = 2   # 束搜索宽度
    candidates_per_node: int = 2  # 每节点候选数
    
    # 模型
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B"
    embedding_model: str = "thenlper/gte-large"
    
    # 训练
    batch_size: int = 2
    max_length: int = 1024
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.1
    
    # 评估
    eval_batch_size: int = 8
    max_eval_samples: int = 100
    
    # 路径
    data_dir: str = "./resources"
    output_dir: str = "./results/small_scale"

@dataclass  
class StandardScaleConfig:
    """标准规模配置"""
    # 数据合成
    num_seeds: int = 5000
    tree_depth: int = 3
    beam_width: int = 3  
    candidates_per_node: int = 3
    
    # 模型
    base_model: str = "Qwen/Qwen2.5-Coder-7B"
    embedding_model: str = "thenlper/gte-large"
    
    # 训练
    batch_size: int = 16
    max_length: int = 4096
    learning_rate: float = 5e-6  
    num_epochs: int = 2
    warmup_ratio: float = 0.01
    
    # 评估
    eval_batch_size: int = 16
    max_eval_samples: int = -1  # 全部评估
    
    # 路径
    data_dir: str = "./resources"
    output_dir: str = "./results/standard_scale"

def get_config(scale: str = "small"):
    """获取配置"""
    if scale == "small":
        return SmallScaleConfig()
    elif scale == "standard":
        return StandardScaleConfig()
    else:
        raise ValueError(f"不支持的规模: {scale}")