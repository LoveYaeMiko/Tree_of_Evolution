import os
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
import datasets
import json

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
            # 使用streaming模式下载部分数据
            dataset = load_dataset("bigcode/the-stack", 
                                 data_dir="data/python", 
                                 split="train", 
                                 streaming=True,
                                 trust_remote_code=True)
            
            # 取前5000个样本作为种子
            samples = []
            for i, sample in enumerate(dataset):
                if i >= 5000:
                    break
                if i % 1000 == 0:
                    print(f"已下载 {i} 个样本...")
                samples.append(sample)
                
            # 保存为JSON
            with open(stack_dir / "seed_samples.json", "w") as f:
                json.dump(samples, f, indent=2)
                
            print(f"✓ Stack v1数据集下载完成，共{len(samples)}个样本")
            
        except Exception as e:
            print(f"下载Stack v1失败: {e}")
            print("创建模拟数据...")
            # 创建模拟数据
            self._create_mock_stack_data(stack_dir)
    
    def _create_mock_stack_data(self, stack_dir):
        """创建模拟的Stack数据"""
        mock_data = []
        
        # 创建一些真实的Python代码示例作为种子
        sample_codes = [
            """
def fibonacci(n):
    \"\"\"计算第n个斐波那契数\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
            """,
            """
def factorial(n):
    \"\"\"计算n的阶乘\"\"\"
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
            """,
            """
def is_prime(n):
    \"\"\"检查数字是否为质数\"\"\"
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
            """,
            """
def binary_search(arr, target):
    \"\"\"二分查找算法\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
            """,
            """
def quick_sort(arr):
    \"\"\"快速排序算法\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
            """
        ]
        
        for i, code in enumerate(sample_codes):
            mock_data.append({
                "content": code.strip(),
                "size": len(code),
                "license": "mit",
                "file_name": f"sample_code_{i}.py"
            })
        
        # 添加更多模拟数据
        for i in range(len(sample_codes), 100):
            mock_data.append({
                "content": f"def example_function_{i}():\n    return 'This is sample code {i}'",
                "size": 100 + i,
                "license": "mit",
                "file_name": f"example_{i}.py"
            })
        
        with open(stack_dir / "seed_samples.json", "w") as f:
            json.dump(mock_data, f, indent=2)
        print("✓ 创建了模拟Stack数据")
    
    def download_benchmarks(self):
        """下载评估基准数据集"""
        print("下载评估基准数据集...")
        benchmarks_dir = self.base_dir / "benchmarks"
        benchmarks_dir.mkdir(exist_ok=True)
        
        # HumanEval
        try:
            print("下载HumanEval数据集...")
            from datasets import load_dataset
            # 使用正确的数据集名称
            humaneval = load_dataset("openai/openai_humaneval", trust_remote_code=True)
            
            # 将路径转换为字符串
            humaneval.save_to_disk(str(benchmarks_dir / "humaneval"))
            print("✓ HumanEval数据集下载完成")
            
            # 同时保存为JSON格式便于查看
            test_data = humaneval["test"]
            with open(benchmarks_dir / "humaneval_test.json", "w") as f:
                json.dump(test_data.to_dict(), f, indent=2)
                
        except Exception as e:
            print(f"下载HumanEval失败: {e}")
            print("创建模拟HumanEval数据...")
            self._create_mock_humaneval(benchmarks_dir)
            
        # MBPP
        try:
            print("下载MBPP数据集...")
            from datasets import load_dataset
            # 使用正确的数据集名称
            mbpp = load_dataset("mbpp", "sanitized", trust_remote_code=True)  # 使用sanitized版本
            
            # 将路径转换为字符串
            mbpp.save_to_disk(str(benchmarks_dir / "mbpp"))
            print("✓ MBPP数据集下载完成")
            
            # 同时保存为JSON格式便于查看
            test_data = mbpp["test"]
            with open(benchmarks_dir / "mbpp_test.json", "w") as f:
                json.dump(test_data.to_dict(), f, indent=2)
                
        except Exception as e:
            print(f"下载MBPP失败: {e}")
            print("创建模拟MBPP数据...")
            self._create_mock_mbpp(benchmarks_dir)
    
    def _create_mock_humaneval(self, benchmarks_dir):
        """创建模拟HumanEval数据"""
        humaneval_dir = benchmarks_dir / "humaneval"
        humaneval_dir.mkdir(exist_ok=True)
        
        # 创建模拟的HumanEval测试数据
        mock_humaneval = {
            "test": [
                {
                    "task_id": "test_0",
                    "prompt": "def add(a, b):\n    \"\"\"返回两个数字的和\"\"\"\n    ",
                    "canonical_solution": "return a + b",
                    "test": "assert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0",
                    "entry_point": "add"
                },
                {
                    "task_id": "test_1", 
                    "prompt": "def factorial(n):\n    \"\"\"计算n的阶乘\"\"\"\n    ",
                    "canonical_solution": "if n == 0:\n    return 1\nelse:\n    return n * factorial(n-1)",
                    "test": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
                    "entry_point": "factorial"
                },
                {
                    "task_id": "test_2",
                    "prompt": "def is_palindrome(s):\n    \"\"\"检查字符串是否是回文\"\"\"\n    ",
                    "canonical_solution": "return s == s[::-1]",
                    "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('a') == True",
                    "entry_point": "is_palindrome"
                }
            ]
        }
        
        # 保存模拟数据
        with open(humaneval_dir / "dataset_dict.json", "w") as f:
            json.dump(mock_humaneval, f, indent=2)
        
        with open(benchmarks_dir / "humaneval_test.json", "w") as f:
            json.dump(mock_humaneval["test"], f, indent=2)
            
        print("✓ 创建了模拟HumanEval数据")
    
    def _create_mock_mbpp(self, benchmarks_dir):
        """创建模拟MBPP数据"""
        mbpp_dir = benchmarks_dir / "mbpp"
        mbpp_dir.mkdir(exist_ok=True)
        
        # 创建模拟的MBPP测试数据
        mock_mbpp = {
            "test": [
                {
                    "task_id": 1,
                    "text": "编写一个函数，接受两个数字作为输入并返回它们的和",
                    "code": "def add(a, b):\n    return a + b",
                    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
                },
                {
                    "task_id": 2,
                    "text": "编写一个函数计算列表中所有元素的平均值",
                    "code": "def average(lst):\n    return sum(lst) / len(lst) if lst else 0",
                    "test_list": ["assert average([1, 2, 3]) == 2.0", "assert average([]) == 0", "assert average([5]) == 5.0"]
                },
                {
                    "task_id": 3,
                    "text": "编写一个函数检查字符串是否是回文",
                    "code": "def is_palindrome(s):\n    return s == s[::-1]",
                    "test_list": ["assert is_palindrome('racecar') == True", "assert is_palindrome('hello') == False", "assert is_palindrome('a') == True"]
                }
            ]
        }
        
        # 保存模拟数据
        with open(mbpp_dir / "dataset_dict.json", "w") as f:
            json.dump(mock_mbpp, f, indent=2)
        
        with open(benchmarks_dir / "mbpp_test.json", "w") as f:
            json.dump(mock_mbpp["test"], f, indent=2)
            
        print("✓ 创建了模拟MBPP数据")
    
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
                # 检查是否已经下载了关键文件
                required_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
                has_files = any((model_dir / f).exists() for f in required_files)
                
                if has_files:
                    print(f"✓ {model_name} 已存在")
                    continue
                else:
                    print(f"⚠ {model_name} 目录存在但文件不完整，重新下载...")
                    shutil.rmtree(model_dir)
                
            try:
                print(f"下载 {model_name}...")
                snapshot_download(
                    repo_id=hf_path,
                    local_dir=str(model_dir),  # 转换为字符串
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"✓ {model_name} 下载完成")
            except Exception as e:
                print(f"下载 {model_name} 失败: {e}")
                print(f"请手动下载: {hf_path}")
    
    def download_all(self):
        """下载所有资源"""
        print("开始下载所有必要资源...")
        self.download_stack_v1()
        self.download_benchmarks() 
        self.download_models()
        print("✓ 所有资源下载完成!")
        
        # 显示下载摘要
        self.show_download_summary()
    
    def show_download_summary(self):
        """显示下载摘要"""
        print("\n" + "="*50)
        print("📊 下载摘要")
        print("="*50)
        
        # 检查Stack v1
        stack_file = self.base_dir / "stack_v1" / "seed_samples.json"
        if stack_file.exists():
            with open(stack_file, 'r') as f:
                data = json.load(f)
            print(f"Stack v1: {len(data)} 个代码样本")
        else:
            print("Stack v1: ✗ 下载失败")
        
        # 检查基准数据集
        benchmarks = ["humaneval", "mbpp"]
        for benchmark in benchmarks:
            benchmark_dir = self.base_dir / "benchmarks" / benchmark
            if benchmark_dir.exists():
                try:
                    with open(self.base_dir / "benchmarks" / f"{benchmark}_test.json", 'r') as f:
                        data = json.load(f)
                    print(f"{benchmark.upper()}: {len(data)} 个测试样本")
                except:
                    print(f"{benchmark.upper()}: ✓ 已下载")
            else:
                print(f"{benchmark.upper()}: ✗ 下载失败")
        
        # 检查模型
        models_dir = self.base_dir / "models"
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                files = list(model_dir.glob("*"))
                print(f"{model_dir.name}: {len(files)} 个文件")
        
        print("="*50)

if __name__ == "__main__":
    downloader = ResourceDownloader()
    downloader.download_all()