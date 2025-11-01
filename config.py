import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DataConfig:
    """数据配置"""
    # 数据规模
    scale: str = "small"  # "small" 或 "standard"
    
    # Stack v1 配置
    stack_samples_small: int = 100  # 小规模种子数
    stack_samples_standard: int = 5000  # 标准规模种子数
    
    # ToE 配置
    evolution_rounds_small: int = 2  # 小规模演化轮数
    evolution_rounds_standard: int = 3  # 标准规模演化轮数
    paths_per_node_small: int = 2  # 小规模每节点路径数
    paths_per_node_standard: int = 3  # 标准规模每节点路径数
    beam_size_small: int = 1  # 小规模beam大小
    beam_size_standard: int = 2  # 标准规模beam大小
    
    # 数据收集
    target_samples_small: int = 5000  # 小规模目标样本数
    target_samples_standard: int = 75000  # 标准规模目标样本数
    similarity_threshold: float = 6.0  # 相似度阈值

@dataclass  
class ModelConfig:
    """模型配置"""
    # 基础模型
    base_model_small: str = "Qwen/Qwen2.5-Coder-1.5B"
    base_model_standard: str = "Qwen/Qwen2.5-Coder-7B"
    
    # 合成模型
    synthesis_model: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 嵌入模型
    embedding_model: str = "thenlper/gte-large-en-v1.5"
    
    # 本地路径
    local_base_path: str = "models"
    local_data_path: str = "data"

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    learning_rate: float = 5e-6
    num_epochs: int = 2
    batch_size_small: int = 32
    batch_size_standard: int = 256
    max_length: int = 4096
    warmup_ratio: float = 0.01
    
    # 硬件
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"

@dataclass
class ToEConfig:
    """ToE框架配置"""
    # 质量评估
    challenge_weight: float = 1.0
    diversity_weight: float = 1.0
    
    # 演化控制
    max_depth: int = 3
    min_challenge_score: float = 5.0
    
    # 评估参数
    challenge_temperature: float = 0.0
    diversity_top_k: int = 1

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, scale: str = "small"):
        self.scale = scale
        self.data_config = DataConfig()
        self.model_config = ModelConfig() 
        self.training_config = TrainingConfig()
        self.toe_config = ToEConfig()
        
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        if self.scale == "small":
            return {
                "stack_samples": self.data_config.stack_samples_small,
                "evolution_rounds": self.data_config.evolution_rounds_small,
                "paths_per_node": self.data_config.paths_per_node_small,
                "beam_size": self.data_config.beam_size_small,
                "target_samples": self.data_config.target_samples_small
            }
        else:
            return {
                "stack_samples": self.data_config.stack_samples_standard,
                "evolution_rounds": self.data_config.evolution_rounds_standard, 
                "paths_per_node": self.data_config.paths_per_node_standard,
                "beam_size": self.data_config.beam_size_standard,
                "target_samples": self.data_config.target_samples_standard
            }
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        if self.scale == "small":
            return {
                "batch_size": self.training_config.batch_size_small,
                "base_model": self.model_config.base_model_small
            }
        else:
            return {
                "batch_size": self.training_config.batch_size_standard, 
                "base_model": self.model_config.base_model_standard
            }