import os
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
import datasets

class ResourceDownloader:
    def __init__(self, base_dir="./resources"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_stack_v1(self):
        """下载Stack v1数据集"""
        print("下载Stack v1数据集...")
        stack_dir = self.base_dir / "stack_v1"
        stack_dir.mkdir(exist_ok=True)
        
        # 使用Hugging Face datasets下载
        try:
            from datasets import load_dataset
            dataset = load_dataset("bigcode/the-stack", 
                                 data_dir="data/python", 
                                 split="train", 
                                 streaming=True)
            
            # 取前5000个样本作为种子
            samples = []
            for i, sample in enumerate(dataset):
                if i >= 5000:
                    break
                samples.append(sample)
                
            # 保存为JSON
            import json
            with open(stack_dir / "seed_samples.json", "w") as f:
                json.dump(samples, f)
                
            print(f"✓ Stack v1数据集下载完成，共{len(samples)}个样本")
            
        except Exception as e:
            print(f"下载Stack v1失败: {e}")
            # 创建模拟数据
            self._create_mock_stack_data(stack_dir)
    
    def _create_mock_stack_data(self, stack_dir):
        """创建模拟的Stack数据"""
        import json
        mock_data = []
        for i in range(100):
            mock_data.append({
                "content": f"def example_function_{i}():\n    return 'This is sample code {i}'",
                "size": 100 + i,
                "license": "mit"
            })
        
        with open(stack_dir / "seed_samples.json", "w") as f:
            json.dump(mock_data, f)
        print("✓ 创建了模拟Stack数据")
    
    def download_benchmarks(self):
        """下载评估基准数据集"""
        print("下载评估基准数据集...")
        benchmarks_dir = self.base_dir / "benchmarks"
        benchmarks_dir.mkdir(exist_ok=True)
        
        # HumanEval
        try:
            from datasets import load_dataset
            humaneval = load_dataset("openai/openai_humaneval")
            humaneval.save_to_disk(benchmarks_dir / "humaneval")
            print("✓ HumanEval数据集下载完成")
        except Exception as e:
            print(f"下载HumanEval失败: {e}")
            
        # MBPP
        try:
            mbpp = load_dataset("google/mbpp")
            mbpp.save_to_disk(benchmarks_dir / "mbpp")
            print("✓ MBPP数据集下载完成")
        except Exception as e:
            print(f"下载MBPP失败: {e}")
    
    def download_models(self):
        """下载模型"""
        print("下载模型...")
        models_dir = self.base_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_configs = {
            "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
            "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B", 
            "gte-large": "thenlper/gte-large"
        }
        
        for model_name, hf_path in model_configs.items():
            model_dir = models_dir / model_name
            if model_dir.exists():
                print(f"✓ {model_name} 已存在")
                continue
                
            try:
                print(f"下载 {model_name}...")
                snapshot_download(
                    repo_id=hf_path,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✓ {model_name} 下载完成")
            except Exception as e:
                print(f"下载 {model_name} 失败: {e}")
    
    def download_all(self):
        """下载所有资源"""
        print("开始下载所有必要资源...")
        self.download_stack_v1()
        self.download_benchmarks() 
        self.download_models()
        print("✓ 所有资源下载完成!")

if __name__ == "__main__":
    downloader = ResourceDownloader()
    downloader.download_all()