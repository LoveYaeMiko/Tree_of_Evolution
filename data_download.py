import os
import json
import requests
import tarfile
import zipfile
from datasets import load_dataset
from tqdm import tqdm
import random
from config import get_config

class DatasetDownloader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_stack_v1(self, num_samples=5000):
        """下载Stack v1数据集并提取代码片段"""
        print("下载Stack v1数据集...")
        
        try:
            # 使用Hugging Face datasets库加载Stack数据集
            dataset = load_dataset("bigcode/the-stack", 
                                 data_dir="data/python", 
                                 split=f"train[:{num_samples}]",
                                 streaming=True)
            
            code_snippets = []
            progress = tqdm(total=num_samples, desc="提取代码片段")
            
            for sample in dataset:
                if len(code_snippets) >= num_samples:
                    break
                    
                # 提取Python代码
                if 'content' in sample and sample['content']:
                    code = sample['content']
                    # 基本过滤：至少包含一个函数定义
                    if 'def ' in code and len(code) > 50 and len(code) < 2000:
                        code_snippets.append(code)
                        progress.update(1)
            
            progress.close()
            
            # 保存代码片段
            output_file = os.path.join(self.data_dir, 'stack_v1_snippets.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(code_snippets, f, ensure_ascii=False, indent=2)
            
            print(f"成功提取 {len(code_snippets)} 个代码片段")
            return code_snippets
            
        except Exception as e:
            print(f"下载Stack v1失败: {e}")
            return self._generate_fallback_snippets(num_samples)
    
    def download_humaneval(self):
        """下载HumanEval数据集"""
        print("下载HumanEval数据集...")
        
        try:
            dataset = load_dataset("openai/humaneval")
            
            problems = []
            for split in ['test']:
                for problem in dataset[split]:
                    problems.append({
                        'task_id': problem['task_id'],
                        'prompt': problem['prompt'],
                        'canonical_solution': problem['canonical_solution'],
                        'test': problem['test'],
                        'entry_point': problem['entry_point']
                    })
            
            # 保存HumanEval数据
            output_file = os.path.join(self.data_dir, 'humaneval.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(problems, f, ensure_ascii=False, indent=2)
            
            print(f"成功下载 {len(problems)} 个HumanEval问题")
            return problems
            
        except Exception as e:
            print(f"下载HumanEval失败: {e}")
            return []
    
    def download_mbpp(self):
        """下载MBPP数据集"""
        print("下载MBPP数据集...")
        
        try:
            dataset = load_dataset("google-research-datasets/mbpp")
            
            problems = []
            for split in ['train', 'test', 'validation']:
                if split in dataset:
                    for problem in dataset[split]:
                        problems.append({
                            'task_id': problem['task_id'],
                            'text': problem['text'],
                            'code': problem['code'],
                            'test_list': problem['test_list'],
                            'test_setup_code': problem.get('test_setup_code', ''),
                            'challenge_test_list': problem.get('challenge_test_list', [])
                        })
            
            # 保存MBPP数据
            output_file = os.path.join(self.data_dir, 'mbpp.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(problems, f, ensure_ascii=False, indent=2)
            
            print(f"成功下载 {len(problems)} 个MBPP问题")
            return problems
            
        except Exception as e:
            print(f"下载MBPP失败: {e}")
            return []
    
    def download_evalplus(self):
        """下载EvalPlus数据集（HumanEval+和MBPP+）"""
        print("下载EvalPlus数据集...")
        
        try:
            # HumanEval+
            humaneval_plus = load_dataset("evalplus/humanevalplus")
            # MBPP+
            mbpp_plus = load_dataset("evalplus/mbppplus")
            
            evalplus_data = {
                'humaneval_plus': [],
                'mbpp_plus': []
            }
            
            # 处理HumanEval+
            for problem in humaneval_plus['test']:
                evalplus_data['humaneval_plus'].append({
                    'task_id': problem['task_id'],
                    'prompt': problem['prompt'],
                    'canonical_solution': problem['canonical_solution'],
                    'test': problem['test'],
                    'base_test': problem['base_test'],
                    'plus_test': problem['plus_test']
                })
            
            # 处理MBPP+
            for problem in mbpp_plus['test']:
                evalplus_data['mbpp_plus'].append({
                    'task_id': problem['task_id'],
                    'text': problem['text'],
                    'code': problem['code'],
                    'test_list': problem['test_list'],
                    'base_test_list': problem['base_test_list'],
                    'plus_test_list': problem['plus_test_list']
                })
            
            # 保存EvalPlus数据
            output_file = os.path.join(self.data_dir, 'evalplus.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evalplus_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功下载 EvalPlus: {len(evalplus_data['humaneval_plus'])} HumanEval+, {len(evalplus_data['mbpp_plus'])} MBPP+")
            return evalplus_data
            
        except Exception as e:
            print(f"下载EvalPlus失败: {e}")
            return {'humaneval_plus': [], 'mbpp_plus': []}
    
    def _generate_fallback_snippets(self, num_samples):
        """生成备用代码片段（当下载失败时使用）"""
        print("生成备用代码片段...")
        
        fallback_snippets = [
            # 基础算法
            "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            
            "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
            
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b",
            
            # 数据处理
            "def read_csv_file(filepath):\n    import csv\n    data = []\n    with open(filepath, 'r', encoding='utf-8') as file:\n        reader = csv.reader(file)\n        for row in reader:\n            data.append(row)\n    return data",
            
            "def filter_data(data, condition):\n    return [item for item in data if condition(item)]",
            
            # 字符串处理
            "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
            
            "def count_words(text):\n    words = text.split()\n    word_count = {}\n    for word in words:\n        word = word.lower().strip('.,!?;:')\n        if word:\n            word_count[word] = word_count.get(word, 0) + 1\n    return word_count",
            
            # 数学计算
            "def calculate_stats(numbers):\n    if not numbers:\n        return None\n    mean = sum(numbers) / len(numbers)\n    sorted_nums = sorted(numbers)\n    mid = len(sorted_nums) // 2\n    median = (sorted_nums[mid] + sorted_nums[~mid]) / 2 if len(sorted_nums) % 2 == 0 else sorted_nums[mid]\n    return {'mean': mean, 'median': median, 'min': min(numbers), 'max': max(numbers)}",
            
            # 文件操作
            "def copy_file(source, destination):\n    with open(source, 'r', encoding='utf-8') as src:\n        content = src.read()\n    with open(destination, 'w', encoding='utf-8') as dst:\n        dst.write(content)",
            
            # 网络请求
            "def make_api_request(url, headers=None):\n    import requests\n    response = requests.get(url, headers=headers or {})\n    if response.status_code == 200:\n        return response.json()\n    else:\n        raise Exception(f'API request failed with status {response.status_code}')"
        ]
        
        # 扩展备用片段到所需数量
        expanded_snippets = []
        while len(expanded_snippets) < num_samples:
            for snippet in fallback_snippets:
                if len(expanded_snippets) >= num_samples:
                    break
                # 对片段进行轻微修改以增加多样性
                modified_snippet = self._modify_snippet(snippet)
                expanded_snippets.append(modified_snippet)
        
        # 保存备用片段
        output_file = os.path.join(self.data_dir, 'fallback_snippets.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(expanded_snippets[:num_samples], f, ensure_ascii=False, indent=2)
        
        return expanded_snippets[:num_samples]
    
    def _modify_snippet(self, snippet):
        """对代码片段进行轻微修改以增加多样性"""
        modifications = [
            lambda s: s.replace('def ', 'def custom_'),
            lambda s: s.replace('arr', 'array'),
            lambda s: s.replace('target', 'search_target'),
            lambda s: s.replace('data', 'dataset'),
            lambda s: s.replace('filepath', 'filename'),
            lambda s: s + '\n# Additional functionality can be added here',
            lambda s: s.replace('return ', '# Process result\n    return '),
        ]
        
        modifier = random.choice(modifications)
        return modifier(snippet)
    
    def download_all_datasets(self):
        """下载所有必要的数据集"""
        print("开始下载所有数据集...")
        
        datasets = {}
        
        # 下载Stack v1用于种子
        datasets['stack_v1'] = self.download_stack_v1(self.config.num_seeds)
        
        # 下载基准数据集用于评估
        datasets['humaneval'] = self.download_humaneval()
        datasets['mbpp'] = self.download_mbpp()
        datasets['evalplus'] = self.download_evalplus()
        
        print("所有数据集下载完成！")
        return datasets

def main():
    config = get_config()
    downloader = DatasetDownloader(config)
    datasets = downloader.download_all_datasets()

if __name__ == "__main__":
    main()