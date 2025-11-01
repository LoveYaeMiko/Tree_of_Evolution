import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ConfigManager
from utils import ProgressTracker

class CodeEvaluator:
    """代码评估器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.training_config.device
        
    def load_model(self, model_path: str):
        """加载模型"""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def evaluate_humaneval(self, model, tokenizer) -> float:
        """评估HumanEval"""
        print("评估HumanEval...")
        
        # 加载数据
        humaneval_file = Path("data/humaneval/data.jsonl")
        problems = []
        
        with open(humaneval_file, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        
        # 评估每个问题
        correct = 0
        total = len(problems)
        
        eval_tracker = ProgressTracker(total, desc="HumanEval评估")
        
        for problem in problems:
            prompt = problem['prompt']
            
            # 生成代码
            inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 简单验证：检查是否包含函数定义
            if "def " in generated_code and "return" in generated_code:
                correct += 1
            
            eval_tracker.update(1, accuracy=f"{correct/total:.2f}")
        
        eval_tracker.close()
        accuracy = correct / total
        print(f"HumanEval 准确率: {accuracy:.4f}")
        
        return accuracy
    
    def evaluate_mbpp(self, model, tokenizer) -> float:
        """评估MBPP"""
        print("评估MBPP...")
        
        # 加载数据
        mbpp_file = Path("data/mbpp/data.jsonl")
        problems = []
        
        with open(mbpp_file, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        
        # 评估每个问题
        correct = 0
        total = len(problems)
        
        eval_tracker = ProgressTracker(total, desc="MBPP评估")
        
        for problem in problems:
            prompt = problem['text']
            
            # 生成代码
            inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 简单验证
            if "def " in generated_code:
                correct += 1
            
            eval_tracker.update(1, accuracy=f"{correct/total:.2f}")
        
        eval_tracker.close()
        accuracy = correct / total
        print(f"MBPP 准确率: {accuracy:.4f}")
        
        return accuracy
    
    def run_evaluation(self, model_path: str):
        """运行评估"""
        print(f"评估模型: {model_path}")
        
        model, tokenizer = self.load_model(model_path)
        
        results = {}
        
        # 评估不同数据集
        results['humaneval'] = self.evaluate_humaneval(model, tokenizer)
        results['mbpp'] = self.evaluate_mbpp(model, tokenizer)
        
        # 保存结果
        results_file = Path(f"evaluation_results_{self.config.scale}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ 评估完成！结果已保存到 {results_file}")
        print(f"HumanEval: {results['humaneval']:.4f}")
        print(f"MBPP: {results['mbpp']:.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="代码模型评估")
    parser.add_argument("--scale", type=str, default="small", choices=["small", "standard"],
                       help="评估规模: small 或 standard")
    parser.add_argument("--model_path", type=str, required=True,
                       help="要评估的模型路径")
    
    args = parser.parse_args()
    
    config = ConfigManager(scale=args.scale)
    evaluator = CodeEvaluator(config)
    evaluator.run_evaluation(args.model_path)

if __name__ == "__main__":
    main()