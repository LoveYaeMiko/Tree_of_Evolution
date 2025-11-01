import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class DataCollector:
    """数据收集器 - 从进化树中收集高质量数据"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型用于生成响应
        print("加载响应生成模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 生成配置 - 使用确定性生成
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,  # 确定性生成
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
    def collect_from_trees(self, trees_data: List[Dict]) -> List[Dict[str, Any]]:
        """从树结构中收集数据"""
        print("开始从进化树收集数据...")
        
        all_instruction_pairs = []
        
        for tree_idx, tree in enumerate(tqdm(trees_data, desc="处理进化树")):
            tree_instructions = self._collect_tree_instructions(tree)
            all_instruction_pairs.extend(tree_instructions)
        
        # 生成响应
        print("为指令生成代码响应...")
        instruction_response_pairs = []
        
        for i in tqdm(range(0, len(all_instruction_pairs), self.config.eval_batch_size)):
            batch = all_instruction_pairs[i:i + self.config.eval_batch_size]
            batch_responses = self._generate_responses_batch(batch)
            instruction_response_pairs.extend(batch_responses)
        
        print(f"数据收集完成，共 {len(instruction_response_pairs)} 个指令-响应对")
        return instruction_response_pairs
    
    def _collect_tree_instructions(self, tree: Dict) -> List[Dict]:
        """从单个树中收集指令"""
        instructions = []
        
        def traverse_node(node, depth=0):
            if isinstance(node, dict):
                # 添加当前节点
                instructions.append({
                    "instruction": node.get("instruction", ""),
                    "challenge_score": node.get("challenge_score", 0),
                    "diversity_score": node.get("diversity_score", 0),
                    "quality_score": node.get("quality_score", 0),
                    "depth": depth
                })
                
                # 遍历子节点
                children = node.get("children", [])
                for child in children:
                    traverse_node(child, depth + 1)
        
        traverse_node(tree)
        return instructions
    
    def _generate_responses_batch(self, instructions_batch: List[Dict]) -> List[Dict]:
        """批量生成响应"""
        responses = []
        
        for item in instructions_batch:
            instruction = item["instruction"]
            response = self._generate_single_response(instruction)
            
            responses.append({
                "instruction": instruction,
                "response": response,
                "challenge_score": item["challenge_score"],
                "diversity_score": item["diversity_score"],
                "quality_score": item["quality_score"]
            })
        
        return responses
    
    def _generate_single_response(self, instruction: str) -> str:
        """为单个指令生成响应"""
        prompt = f"""请为以下编程问题生成Python代码解决方案:

问题: {instruction}

代码:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = response.split("代码:")[-1].strip()
        
        return code
    
    def filter_high_quality_data(self, data: List[Dict], threshold: float = 10.0) -> List[Dict]:
        """过滤高质量数据"""
        print(f"过滤高质量数据 (阈值: {threshold})...")
        
        filtered_data = [
            item for item in data 
            if item.get("quality_score", 0) >= threshold
        ]
        
        print(f"过滤后数据量: {len(filtered_data)} / {len(data)}")
        return filtered_data
    
    def save_instruction_response_pairs(self, data: List[Dict[str, Any]]):
        """保存指令-响应对"""
        output_dir = Path(self.config.output_dir) / "synthesized_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整数据
        full_output_file = output_dir / "instruction_response_pairs_full.json"
        with open(full_output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存高质量数据
        high_quality_data = self.filter_high_quality_data(data)
        hq_output_file = output_dir / "instruction_response_pairs_high_quality.json"
        with open(hq_output_file, 'w', encoding='utf-8') as f:
            json.dump(high_quality_data, f, ensure_ascii=False, indent=2)
        
        # 保存训练格式数据
        training_data = self._format_for_training(high_quality_data)
        training_file = output_dir / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_dir}")
        print(f"- 完整数据: {len(data)} 样本")
        print(f"- 高质量数据: {len(high_quality_data)} 样本")
    
    def _format_for_training(self, data: List[Dict]) -> List[Dict]:
        """格式化为训练数据"""
        training_data = []
        
        for item in data:
            training_data.append({
                "instruction": item["instruction"],
                "input": "",
                "output": item["response"],
                "challenge_score": item["challenge_score"],
                "diversity_score": item["diversity_score"]
            })
        
        return training_data