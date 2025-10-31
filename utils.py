import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from transformers import AutoTokenizer

class QualityEvaluator:
    def __init__(self, embed_model_path):
        print(f"加载嵌入模型: {embed_model_path}")
        self.embedding_model = SentenceTransformer(embed_model_path)
        self.diversity_db = []
        
    def evaluate_challenge(self, instruction, tokenizer, model):
        """评估指令的挑战性"""
        prompt = f"""评估以下编程问题的挑战性（1-10分）：
{instruction}

请考虑：
1. 问题复杂度
2. 概念深度  
3. 所需技能水平
4. 边界情况和优化需求

输出格式：分数: [1-10]"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 简单解析分数
        try:
            score_text = response.split("分数:")[-1].strip()
            score = float(score_text.split()[0])
            return min(max(score, 1), 10)
        except:
            return 5.0  # 默认分数
    
    def evaluate_diversity(self, instruction, current_round_trees):
        """评估指令的多样性"""
        if not current_round_trees:
            return 5.0
            
        # 编码当前指令
        current_embedding = self.embedding_model.encode([instruction])
        
        # 构建其他树的挑战性指令数据库
        external_instructions = []
        for tree_id, tree_instructions in current_round_trees.items():
            if tree_instructions:
                # 取每个树中最具挑战性的指令
                challenging_instruction = max(tree_instructions, 
                                           key=lambda x: x.get('challenge_score', 0))
                external_instructions.append(challenging_instruction['instruction'])
        
        if not external_instructions:
            return 5.0
            
        # 编码外部指令
        external_embeddings = self.embedding_model.encode(external_instructions)
        
        # 使用FAISS计算最近邻距离
        index = faiss.IndexFlatIP(external_embeddings.shape[1])
        index.add(external_embeddings.astype('float32'))
        
        distances, _ = index.search(current_embedding.astype('float32'), k=1)
        diversity_score = 10 * (1 - distances[0][0])  # 转换为0-10分
        
        return min(max(diversity_score, 0), 10)

class ProgressTracker:
    def __init__(self, total_steps, desc="进度"):
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = tqdm(total=total_steps, desc=desc)
        
    def update(self, step=1, status=""):
        self.current_step += step
        self.pbar.update(step)
        if status:
            self.pbar.set_description(f"进度: {status}")
            
    def close(self):
        self.pbar.close()

def save_checkpoint(data, filepath):
    """保存检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_checkpoint(filepath):
    """加载检查点"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def print_progress(current, total, prefix="", suffix=""):
    """打印进度信息"""
    progress = current / total * 100
    print(f"\r{prefix} [{current}/{total}] {progress:.1f}% {suffix}", end="", flush=True)
    if current == total:
        print()