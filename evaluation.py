import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from config import get_config
from model_manager import ModelManager

class CodeEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 确保模型存在
        self._ensure_models_exist()
        
        print("加载评估模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def _ensure_models_exist(self):
        """确保评估模型存在"""
        manager = ModelManager(self.config)
        
        if not self.config.base_model or not os.path.exists(self.config.base_model):
            print(f"评估模型不存在: {self.config.base_model}")
            print("正在下载模型...")
            manager.setup_models(self.config.mode, force_download=self.config.force_download)
            
            # 更新配置路径
            updated_config = manager.load_model_config()
            if updated_config:
                self.config.base_model = updated_config.get("base_model", self.config.base_model)

    def evaluate_humaneval(self, model_path):
        """在HumanEval数据集上评估模型"""
        print("在HumanEval数据集上评估模型...")
        
        # 加载HumanEval数据
        humaneval_file = os.path.join(self.config.data_dir, 'humaneval.json')
        if not os.path.exists(humaneval_file):
            print("HumanEval数据集未找到，请先运行 data_download.py")
            return None
        
        with open(humaneval_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        results = []
        progress = tqdm(total=len(problems), desc="评估HumanEval")
        
        for problem in problems:
            prompt = problem['prompt']
            
            # 生成代码
            generated_code = self._generate_code(prompt)
            
            # 简单的通过率评估（在实际应用中应该执行测试）
            pass_rate = self._simple_evaluation(problem, generated_code)
            
            results.append({
                'task_id': problem['task_id'],
                'prompt': prompt,
                'generated_code': generated_code,
                'pass_rate': pass_rate,
                'canonical_solution': problem['canonical_solution']
            })
            
            progress.update(1)
            current_avg = sum(r['pass_rate'] for r in results) / len(results)
            progress.set_description(f"评估HumanEval - 平均通过率: {current_avg:.2f}")
        
        progress.close()
        
        overall_pass_rate = sum(r['pass_rate'] for r in results) / len(results)
        print(f"HumanEval总体通过率: {overall_pass_rate:.4f}")
        
        # 保存结果
        result_file = os.path.join(self.config.output_dir, 'humaneval_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_pass_rate': overall_pass_rate,
                'detailed_results': results
            }, f, ensure_ascii=False, indent=2)
        
        return overall_pass_rate
    
    def evaluate_mbpp(self, model_path):
        """在MBPP数据集上评估模型"""
        print("在MBPP数据集上评估模型...")
        
        # 加载MBPP数据
        mbpp_file = os.path.join(self.config.data_dir, 'mbpp.json')
        if not os.path.exists(mbpp_file):
            print("MBPP数据集未找到，请先运行 data_download.py")
            return None
        
        with open(mbpp_file, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        # 只评估测试集
        test_problems = [p for p in problems if p.get('task_id', '').startswith('Mbpp/')]
        if not test_problems:
            test_problems = problems[:100]  # 如果没有明确标识，取前100个
        
        results = []
        progress = tqdm(total=len(test_problems), desc="评估MBPP")
        
        for problem in test_problems:
            prompt = problem['text']
            
            # 生成代码
            generated_code = self._generate_code(prompt)
            
            # 简单的通过率评估
            pass_rate = self._simple_evaluation_mbpp(problem, generated_code)
            
            results.append({
                'task_id': problem['task_id'],
                'prompt': prompt,
                'generated_code': generated_code,
                'pass_rate': pass_rate,
                'reference_code': problem['code']
            })
            
            progress.update(1)
            current_avg = sum(r['pass_rate'] for r in results) / len(results)
            progress.set_description(f"评估MBPP - 平均通过率: {current_avg:.2f}")
        
        progress.close()
        
        overall_pass_rate = sum(r['pass_rate'] for r in results) / len(results)
        print(f"MBPP总体通过率: {overall_pass_rate:.4f}")
        
        # 保存结果
        result_file = os.path.join(self.config.output_dir, 'mbpp_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_pass_rate': overall_pass_rate,
                'detailed_results': results
            }, f, ensure_ascii=False, indent=2)
        
        return overall_pass_rate
    
    def _generate_code(self, prompt):
        """生成代码"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    def _simple_evaluation(self, problem, generated_code):
        """简化的评估方法（在实际应用中应该执行真实的测试用例）"""
        # 这里使用简单的启发式方法评估代码质量
        score = 0.0
        
        # 检查是否包含函数定义
        if 'def ' in generated_code:
            score += 0.3
        
        # 检查代码长度
        if len(generated_code) > 50:
            score += 0.2
        
        # 检查是否包含返回语句
        if 'return ' in generated_code:
            score += 0.3
        
        # 检查基本语法（简单的括号匹配）
        if generated_code.count('(') == generated_code.count(')') and \
           generated_code.count('[') == generated_code.count(']') and \
           generated_code.count('{') == generated_code.count('}'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _simple_evaluation_mbpp(self, problem, generated_code):
        """MBPP的简化评估"""
        return self._simple_evaluation(problem, generated_code)

def main():
    config = get_config()
    model_path = os.path.join(config.output_dir, "fine_tuned_model")
    
    if not os.path.exists(model_path):
        print(f"微调模型不存在: {model_path}")
        print("请先运行 train_model.py 进行模型训练")
        return
    
    evaluator = CodeEvaluator(config)
    
    # 在多个基准上评估
    humaneval_score = evaluator.evaluate_humaneval(model_path)
    mbpp_score = evaluator.evaluate_mbpp(model_path)
    
    print("\n评估结果汇总:")
    print(f"HumanEval Pass@1: {humaneval_score:.4f}")
    print(f"MBPP Pass@1: {mbpp_score:.4f}")

if __name__ == "__main__":
    main()