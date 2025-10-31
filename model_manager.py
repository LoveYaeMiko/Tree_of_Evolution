import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login
import subprocess
import sys
from tqdm import tqdm
import requests
import json
from config import get_config

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.model_dir = os.path.join(config.data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 定义模型配置
        self.model_configs = {
            "small": {
                "base_model": "Qwen/Qwen2.5-Coder-1.5B",
                "synthesis_model": "Qwen/Qwen2.5-Coder-7B",
                "local_base_path": "Qwen2.5-Coder-1.5B",
                "local_synthesis_path": "Qwen2.5-Coder-7B"
            },
            "standard": {
                "base_model": "Qwen/Qwen2.5-Coder-7B", 
                "synthesis_model": "Qwen/Qwen2.5-14B-Instruct",
                "local_base_path": "Qwen2.5-Coder-7B",
                "local_synthesis_path": "Qwen2.5-14B-Instruct"
            }
        }
    
    def download_model(self, model_name, local_name=None, force_download=False):
        """下载模型"""
        if local_name is None:
            local_name = model_name.split('/')[-1]
        
        local_path = os.path.join(self.model_dir, local_name)
        
        # 检查模型是否已下载
        if os.path.exists(local_path) and not force_download:
            print(f"模型 {model_name} 已存在于 {local_path}")
            return local_path
        
        print(f"开始下载模型: {model_name}")
        print(f"保存到: {local_path}")
        
        try:
            # 使用snapshot_download下载模型
            downloaded_path = snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=True
            )
            print(f"模型下载完成: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            print(f"下载模型失败: {e}")
            print("尝试使用git lfs下载...")
            return self._download_with_git_lfs(model_name, local_path)
    
    def _download_with_git_lfs(self, model_name, local_path):
        """使用git lfs下载模型（备用方法）"""
        try:
            # 克隆仓库
            clone_cmd = f"git clone https://huggingface.co/{model_name} {local_path}"
            subprocess.run(clone_cmd, shell=True, check=True)
            
            # 进入目录并拉取LFS文件
            lfs_cmd = f"cd {local_path} && git lfs pull"
            subprocess.run(lfs_cmd, shell=True, check=True)
            
            print(f"模型下载完成: {local_path}")
            return local_path
            
        except subprocess.CalledProcessError as e:
            print(f"git lfs下载失败: {e}")
            return None
    
    def download_all_models(self, mode="small"):
        """下载所有需要的模型"""
        config = self.model_configs.get(mode, self.model_configs["small"])
        
        print(f"下载 {mode} 模式所需模型...")
        
        models = {}
        
        # 下载基础模型
        print("\n1. 下载基础模型...")
        base_path = self.download_model(config["base_model"], config["local_base_path"])
        models["base_model"] = base_path
        
        # 下载合成模型
        print("\n2. 下载数据合成模型...")
        synthesis_path = self.download_model(config["synthesis_model"], config["local_synthesis_path"])
        models["synthesis_model"] = synthesis_path
        
        # 下载评估模型（使用基础模型）
        models["eval_model"] = base_path
        
        # 下载嵌入模型
        print("\n3. 下载嵌入模型...")
        embed_path = self.download_model("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2")
        models["embed_model"] = embed_path
        
        # 保存模型路径配置
        model_config_file = os.path.join(self.model_dir, "model_paths.json")
        with open(model_config_file, 'w', encoding='utf-8') as f:
            json.dump(models, f, ensure_ascii=False, indent=2)
        
        print(f"\n所有模型下载完成！配置保存在: {model_config_file}")
        return models
    
    def load_model_config(self):
        """加载模型路径配置"""
        model_config_file = os.path.join(self.model_dir, "model_paths.json")
        if os.path.exists(model_config_file):
            with open(model_config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_model_path(self, model_type, mode="small"):
        """获取模型路径"""
        config = self.load_model_config()
        if config and model_type in config:
            return config[model_type]
        
        # 如果配置不存在，使用默认路径
        model_configs = self.model_configs.get(mode, self.model_configs["small"])
        local_name = model_configs[f"local_{model_type.split('_')[0]}_path"]
        return os.path.join(self.model_dir, local_name)
    
    def check_model_exists(self, model_type, mode="small"):
        """检查模型是否存在"""
        model_path = self.get_model_path(model_type, mode)
        return model_path and os.path.exists(model_path)
    
    def setup_models(self, mode="small", force_download=False):
        """设置所有需要的模型"""
        if force_download or not self.check_model_exists("base_model", mode):
            return self.download_all_models(mode)
        else:
            return self.load_model_config()

def main():
    config = get_config()
    manager = ModelManager(config)
    
    print("模型管理器")
    print("1. 下载小规模模型")
    print("2. 下载标准规模模型") 
    print("3. 检查模型状态")
    
    choice = input("请选择操作 (1/2/3): ").strip()
    
    if choice == "1":
        manager.download_all_models("small")
    elif choice == "2":
        manager.download_all_models("standard")
    elif choice == "3":
        models = manager.setup_models("small")
        if models:
            print("模型状态:")
            for model_type, path in models.items():
                exists = os.path.exists(path)
                status = "✓ 已下载" if exists else "✗ 缺失"
                print(f"  {model_type}: {status} ({path})")
        else:
            print("未找到模型配置，请先下载模型")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()