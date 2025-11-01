import torch
import datetime  # 添加 datetime 导入
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import ast
import signal

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")

class PaperEvaluatorFixed:
    """修复版的论文评估器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载完成")
    
    def load_base_model(self, model_name):
        """加载未微调的基模型"""
        print(f"加载基模型: {model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        print("基模型加载完成")
    
    def evaluate_humaneval(self, use_base_model=False):
        """在HumanEval上评估"""
        print("在HumanEval上评估...")
        
        try:
            dataset = load_dataset("openai/openai_humaneval", split="test")
            return self._evaluate_humaneval_dataset(dataset, use_base_model)
        except Exception as e:
            print(f"HumanEval评估失败: {e}")
            return self._evaluate_mock_humaneval(use_base_model)
    
    def evaluate_mbpp(self, use_base_model=False):
        """在MBPP上评估 - 修复字段问题"""
        print("在MBPP上评估...")
        
        try:
            # 尝试不同的MBPP数据集加载方式
            try:
                dataset = load_dataset("mbpp", "sanitized", split="test")
            except:
                dataset = load_dataset("mbpp", split="test")
            
            return self._evaluate_mbpp_dataset_fixed(dataset, use_base_model)
        except Exception as e:
            print(f"MBPP评估失败: {e}")
            return self._evaluate_mock_mbpp(use_base_model)
    
    def _evaluate_humaneval_dataset(self, dataset, use_base_model=False):
        """评估HumanEval数据集"""
        results = []
        total = min(self.config.max_eval_samples, len(dataset)) if self.config.max_eval_samples > 0 else len(dataset)
        
        for i in tqdm(range(total), desc=f"评估HumanEval ({'基模型' if use_base_model else '微调模型'})"):
            example = dataset[i]
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]
            test_cases = example["test"]
            task_id = example["task_id"]
            
            # 生成代码
            if use_base_model:
                generated_code = self._generate_code_with_model(prompt, self.base_model, self.base_tokenizer)
            else:
                generated_code = self._generate_code_with_model(prompt, self.model, self.tokenizer)
            
            # 执行测试用例
            passed = self._execute_test_cases(generated_code, test_cases, canonical_solution)
            
            results.append({
                "task_id": task_id,
                "passed": passed,
                "generated_code": generated_code,
                "canonical_solution": canonical_solution
            })
        
        # 计算pass@1
        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results) if results else 0
        
        return {
            "dataset": "humaneval",
            "model_type": "base" if use_base_model else "fine_tuned",
            "pass@1": pass_rate,
            "total_samples": len(results),
            "passed_samples": pass_count,
            "detailed_results": results
        }
    
    def _evaluate_mbpp_dataset_fixed(self, dataset, use_base_model=False):
        """修复版的MBPP数据集评估"""
        results = []
        total = min(self.config.max_eval_samples, len(dataset)) if self.config.max_eval_samples > 0 else len(dataset)
        
        for i in tqdm(range(total), desc=f"评估MBPP ({'基模型' if use_base_model else '微调模型'})"):
            example = dataset[i]
            
            # 修复：检查并适配不同的字段名
            if "text" in example:
                prompt = example["text"]
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "description" in example:
                prompt = example["description"]
            else:
                # 打印可用的字段以便调试
                print(f"可用的字段: {list(example.keys())}")
                prompt = str(example)  # 回退方案
            
            if "code" in example:
                canonical_solution = example["code"]
            elif "canonical_solution" in example:
                canonical_solution = example["canonical_solution"]
            else:
                canonical_solution = ""
            
            if "test_list" in example:
                test_list = example["test_list"]
            elif "test" in example:
                test_list = example["test"]
            else:
                test_list = []
            
            task_id = example.get("task_id", f"mbpp_{i}")
            
            # 生成代码
            if use_base_model:
                generated_code = self._generate_code_with_model(prompt, self.base_model, self.base_tokenizer)
            else:
                generated_code = self._generate_code_with_model(prompt, self.model, self.tokenizer)
            
            # 执行测试用例
            passed = self._execute_mbpp_test_cases(generated_code, test_list, canonical_solution)
            
            results.append({
                "task_id": task_id,
                "passed": passed,
                "generated_code": generated_code,
                "canonical_solution": canonical_solution
            })
        
        # 计算pass@1
        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results) if results else 0
        
        return {
            "dataset": "mbpp",
            "model_type": "base" if use_base_model else "fine_tuned",
            "pass@1": pass_rate,
            "total_samples": len(results),
            "passed_samples": pass_count,
            "detailed_results": results
        }
    
    def _generate_code_with_model(self, prompt, model, tokenizer):
        """使用指定模型生成代码"""
        # 格式化提示 - 与论文中的格式一致
        formatted_prompt = f"### 指令:\n{prompt}\n\n### 响应:\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分
        if "### 响应:" in generated_text:
            code = generated_text.split("### 响应:")[-1].strip()
        else:
            code = generated_text[len(formatted_prompt):].strip()
        
        return code
    
    def _execute_test_cases(self, generated_code, test_cases, canonical_solution):
        """执行HumanEval测试用例"""
        try:
            # 创建完整的Python代码
            full_code = generated_code + "\n\n" + test_cases
            
            # 在隔离环境中执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                # 设置超时（5秒）
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                # 执行代码
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                signal.alarm(0)  # 取消超时
                
                # 检查执行结果
                return result.returncode == 0
                    
            except TimeoutException:
                return False
            except subprocess.TimeoutExpired:
                return False
            finally:
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
        except Exception:
            return False
    
    def _execute_mbpp_test_cases(self, generated_code, test_list, canonical_solution):
        """执行MBPP测试用例"""
        try:
            # 创建完整的Python代码
            test_code = "\n".join([f"    {test}" for test in test_list])
            full_code = f"""
{generated_code}

def test_function():
{test_code}

if __name__ == "__main__":
    test_function()
    print("All tests passed!")
"""
            
            # 在隔离环境中执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                # 设置超时（5秒）
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                # 执行代码
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                signal.alarm(0)  # 取消超时
                
                # 检查执行结果
                return result.returncode == 0 and "All tests passed!" in result.stdout
                    
            except TimeoutException:
                return False
            except subprocess.TimeoutExpired:
                return False
            finally:
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
        except Exception:
            return False
    
    def _evaluate_mock_humaneval(self, use_base_model=False):
        """模拟HumanEval评估"""
        print("使用模拟HumanEval数据...")
        
        mock_data = [
            {
                "task_id": "mock_1",
                "prompt": "def add(a, b):\n    \"\"\"返回两个数字的和\"\"\"\n    ",
                "canonical_solution": "return a + b",
                "test": "assert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0"
            }
        ]
        
        results = []
        for example in mock_data:
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]
            test_cases = example["test"]
            task_id = example["task_id"]
            
            if use_base_model:
                generated_code = self._generate_code_with_model(prompt, self.base_model, self.base_tokenizer)
            else:
                generated_code = self._generate_code_with_model(prompt, self.model, self.tokenizer)
            
            # 简单检查生成的代码是否包含关键部分
            passed = "return" in generated_code and ("a + b" in generated_code or "a+b" in generated_code)
            
            results.append({
                "task_id": task_id,
                "passed": passed,
                "generated_code": generated_code,
                "canonical_solution": canonical_solution
            })
        
        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results) if results else 0
        
        return {
            "dataset": "humaneval_mock",
            "model_type": "base" if use_base_model else "fine_tuned",
            "pass@1": pass_rate,
            "total_samples": len(results),
            "passed_samples": pass_count,
            "detailed_results": results
        }
    
    def _evaluate_mock_mbpp(self, use_base_model=False):
        """模拟MBPP评估"""
        print("使用模拟MBPP数据...")
        
        mock_data = [
            {
                "task_id": 1,
                "text": "编写一个函数，接受两个数字作为输入并返回它们的和",
                "code": "def add(a, b):\n    return a + b",
                "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
            }
        ]
        
        results = []
        for example in mock_data:
            prompt = example["text"]
            canonical_solution = example["code"]
            test_list = example["test_list"]
            task_id = example["task_id"]
            
            if use_base_model:
                generated_code = self._generate_code_with_model(prompt, self.base_model, self.base_tokenizer)
            else:
                generated_code = self._generate_code_with_model(prompt, self.model, self.tokenizer)
            
            # 简单检查生成的代码是否包含关键部分
            passed = "return" in generated_code and ("a + b" in generated_code or "a+b" in generated_code)
            
            results.append({
                "task_id": task_id,
                "passed": passed,
                "generated_code": generated_code,
                "canonical_solution": canonical_solution
            })
        
        pass_count = sum(1 for r in results if r["passed"])
        pass_rate = pass_count / len(results) if results else 0
        
        return {
            "dataset": "mbpp_mock",
            "model_type": "base" if use_base_model else "fine_tuned",
            "pass@1": pass_rate,
            "total_samples": len(results),
            "passed_samples": pass_count,
            "detailed_results": results
        }
    
    def save_comparison_results(self, base_results, fine_tuned_results, benchmark_name):
        """保存对比结果"""
        output_dir = Path(self.config.output_dir) / "paper_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建对比结果
        comparison = {
            "benchmark": benchmark_name,
            "base_model": self.config.base_model,
            "fine_tuned_model": "custom_fine_tuned",
            "results": {
                "base_model": {
                    "pass@1": base_results["pass@1"],
                    "total_samples": base_results["total_samples"],
                    "passed_samples": base_results["passed_samples"]
                },
                "fine_tuned_model": {
                    "pass@1": fine_tuned_results["pass@1"],
                    "total_samples": fine_tuned_results["total_samples"],
                    "passed_samples": fine_tuned_results["passed_samples"]
                },
                "improvement": fine_tuned_results["pass@1"] - base_results["pass@1"]
            },
            "timestamp": str(datetime.datetime.now())  # 修复：使用正确的 datetime
        }
        
        output_file = output_dir / f"{benchmark_name}_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果
        detailed_file = output_dir / f"{benchmark_name}_detailed.json"
        detailed_results = {
            "base_model_detailed": base_results["detailed_results"],
            "fine_tuned_model_detailed": fine_tuned_results["detailed_results"]
        }
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"对比结果已保存到: {output_file}")
        
        # 打印摘要
        print(f"\n📊 {benchmark_name.upper()} 评估结果对比:")
        print(f"  基模型 ({self.config.base_model}): {base_results['pass@1']*100:.2f}% pass@1")
        print(f"  微调模型: {fine_tuned_results['pass@1']*100:.2f}% pass@1")
        print(f"  改进: {comparison['results']['improvement']*100:+.2f}%")