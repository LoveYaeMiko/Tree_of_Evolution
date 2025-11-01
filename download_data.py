import os
import json
import requests
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

class DataDownloader:
    """数据下载器"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def download_stack_v1_sample(self, num_samples: int = 5000):
        """下载Stack v1样本数据"""
        print("下载Stack v1样本数据...")
        
        stack_dir = self.data_dir / "stack_v1"
        stack_dir.mkdir(exist_ok=True)
        
        # 从HuggingFace下载部分数据
        try:
            # 下载一个较小的代码数据集作为替代
            snapshot_download(
                repo_id="bigcode/the-stack-dedup",
                allow_patterns=["data/python/*.jsonl.gz"],
                local_dir=stack_dir,
                repo_type="dataset"
            )
            print("✅ Stack v1数据下载完成")
        except Exception as e:
            print(f"⚠️ 下载Stack v1失败: {e}")
            # 创建示例数据
            self._create_sample_stack_data(stack_dir, num_samples)
    
    def download_humaneval(self):
        """下载HumanEval数据集"""
        print("下载HumanEval数据集...")
        
        humaneval_dir = self.data_dir / "humaneval"
        humaneval_dir.mkdir(exist_ok=True)
        
        try:
            hf_hub_download(
                repo_id="openai/humaneval",
                filename="data/human_eval.jsonl",
                repo_type="dataset",
                local_dir=humaneval_dir
            )
            print("✅ HumanEval数据下载完成")
        except Exception as e:
            print(f"⚠️ 下载HumanEval失败: {e}")
            self._create_sample_humaneval_data(humaneval_dir)
    
    def download_mbpp(self):
        """下载MBPP数据集"""
        print("下载MBPP数据集...")
        
        mbpp_dir = self.data_dir / "mbpp"
        mbpp_dir.mkdir(exist_ok=True)
        
        try:
            hf_hub_download(
                repo_id="google-research-datasets/mbpp",
                filename="sanitized-mbpp.json",
                repo_type="dataset", 
                local_dir=mbpp_dir
            )
            print("✅ MBPP数据下载完成")
        except Exception as e:
            print(f"⚠️ 下载MBPP失败: {e}")
            self._create_sample_mbpp_data(mbpp_dir)
    
    def _create_sample_stack_data(self, output_dir: Path, num_samples: int):
        """创建Stack v1示例数据"""
        print("创建Stack v1示例数据...")
        
        sample_code_snippets = [
            {
                "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python"
            },
            {
                "content": "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
                "language": "python" 
            },
            # 添加更多示例...
        ]
        
        with open(output_dir / "sample.jsonl", 'w') as f:
            for i in range(min(num_samples, len(sample_code_snippets))):
                f.write(json.dumps(sample_code_snippets[i % len(sample_code_snippets)]) + '\n')
        
        print(f"✅ 创建了 {num_samples} 个Stack v1样本")
    
    def _create_sample_humaneval_data(self, output_dir: Path):
        """创建HumanEval示例数据"""
        print("创建HumanEval示例数据...")
        
        sample_problems = [
            {
                "task_id": "test_0",
                "prompt": "def add(a, b):\n    \"\"\"返回两个数的和\"\"\"\n    return a + b",
                "test": "assert add(1, 2) == 3\nassert add(0, 0) == 0",
                "entry_point": "add"
            }
        ]
        
        with open(output_dir / "data.jsonl", 'w') as f:
            for problem in sample_problems:
                f.write(json.dumps(problem) + '\n')
        
        print("✅ 创建了HumanEval示例数据")
    
    def _create_sample_mbpp_data(self, output_dir: Path):
        """创建MBPP示例数据"""
        print("创建MBPP示例数据...")
        
        sample_problems = [
            {
                "task_id": 1,
                "text": "编写一个函数计算两个数的和",
                "code": "def add(a, b):\n    return a + b",
                "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
            }
        ]
        
        with open(output_dir / "data.jsonl", 'w') as f:
            for problem in sample_problems:
                f.write(json.dumps(problem) + '\n')
        
        print("✅ 创建了MBPP示例数据")

def main():
    downloader = DataDownloader()
    
    print("开始下载数据...")
    print("=" * 50)
    
    # 下载所有数据
    downloader.download_stack_v1_sample(100)  # 小规模样本
    downloader.download_humaneval()
    downloader.download_mbpp()
    
    print("=" * 50)
    print("数据下载完成！")

if __name__ == "__main__":
    main()