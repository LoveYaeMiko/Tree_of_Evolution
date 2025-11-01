import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
from pathlib import Path
import evaluate

class CodeEvaluator:
    """代码评估器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载评估指标
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.codebleu = evaluate.load("codebleu")
    
    def load_model(self, model_path):
        """加载微调后的模型"""
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_on_benchmark(self, benchmark_name, split="test"):
        """在基准测试上评估"""
        print(f"在 {benchmark_name} 上评估模型...")
        
        # 加载基准数据集
        if benchmark_name == "humaneval":
            dataset = load_dataset("openai/humaneval", split=split)
        elif benchmark_name == "mbpp":
            dataset = load_dataset("google/mbpp", split=split)
        else:
            raise ValueError(f"不支持的基准: {benchmark_name}")
        
        results = []
        total = min(self.config.max_eval_samples, len(dataset)) if self.config.max_eval_samples > 0 else len(dataset)
        
        for i in tqdm(range(total), desc=f"评估 {benchmark_name}"):
            example = dataset[i]
            
            if benchmark_name == "humaneval":
                prompt = example["prompt"]
                reference = example["canonical_solution"]
            else:  # mbpp
                prompt = example["text"]
                reference = example["code"]
            
            # 生成代码
            generated_code = self._generate_code(prompt)
            
            # 计算指标
            metrics = self._compute_metrics(generated_code, reference)
            metrics["problem_id"] = example.get("task_id", i)
            results.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in results[0].keys():
            if key != "problem_id":
                avg_metrics[f"avg_{key}"] = sum(r[key] for r in results) / len(results)
        
        return results, avg_metrics
    
    def _generate_code(self, prompt: str) -> str:
        """生成代码"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分
        if "### 响应:" in generated_text:
            code = generated_text.split("### 响应:")[-1].strip()
        else:
            code = generated_text[len(prompt):].strip()
        
        return code
    
    def _compute_metrics(self, generated: str, reference: str) -> dict:
        """计算评估指标"""
        # BLEU
        bleu_score = self.bleu.compute(
            predictions=[generated], 
            references=[[reference]]
        )["bleu"]
        
        # ROUGE
        rouge_scores = self.rouge.compute(
            predictions=[generated], 
            references=[reference]
        )
        
        # CodeBLEU
        codebleu_score = self.codebleu.compute(
            predictions=[generated], 
            references=[[reference]]
        )["codebleu"]
        
        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"], 
            "rougeL": rouge_scores["rougeL"],
            "codebleu": codebleu_score
        }
    
    def save_evaluation_results(self, results: dict, benchmark_name: str):
        """保存评估结果"""
        output_dir = Path(self.config.output_dir) / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{benchmark_name}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {output_file}")