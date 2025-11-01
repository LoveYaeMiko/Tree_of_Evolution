import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int, desc: str = "Processing"):
        self.pbar = tqdm(total=total_steps, desc=desc)
        self.current_step = 0
        
    def update(self, steps: int = 1, **kwargs):
        """更新进度"""
        self.current_step += steps
        self.pbar.update(steps)
        if kwargs:
            self.pbar.set_postfix(**kwargs)
            
    def close(self):
        """关闭进度条"""
        self.pbar.close()

class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self, embedding_model_path: str, device: str = "cuda"):
        self.device = device
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        
        # 初始化FAISS索引
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.instruction_db = []
        
    def evaluate_challenge(self, instruction: str, evaluator_model, evaluator_tokenizer) -> float:
        """评估挑战性"""
        prompt = f"""你是一个编程问题挑战性评估专家。请评估以下编程问题的挑战性，打分范围1-10：

{instruction}

请按以下标准评估：
1. 问题复杂度：是否涉及复杂逻辑或多步骤解决？
2. 概念深度：是否需要理解高级编程概念？
3. 技能要求：是否适合中级以上程序员？
4. 边界情况：是否需要考虑边界情况或优化？

请严格按照以下格式回复：
评估理由：[你的评估理由]
分数：[1-10的整数]

"""
        
        inputs = evaluator_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = evaluator_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                pad_token_id=evaluator_tokenizer.eos_token_id
            )
            
        response = evaluator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析分数
        try:
            score_text = response.split("分数:")[-1].strip()
            score = float(score_text.split()[0])
            return max(1.0, min(10.0, score))
        except:
            return 5.0  # 默认分数
    
    def evaluate_diversity(self, instruction: str, external_db: List[str]) -> float:
        """评估多样性"""
        if not external_db:
            return 5.0  # 默认多样性分数
            
        # 编码当前指令
        current_embedding = self.embedding_model.encode([instruction], convert_to_tensor=True)
        current_embedding = current_embedding.cpu().numpy()
        
        # 编码外部数据库
        db_embeddings = self.embedding_model.encode(external_db, convert_to_tensor=True)
        db_embeddings = db_embeddings.cpu().numpy()
        
        # 计算相似度
        similarities = []
        for db_embedding in db_embeddings:
            similarity = np.dot(current_embedding[0], db_embedding) / (
                np.linalg.norm(current_embedding[0]) * np.linalg.norm(db_embedding)
            )
            similarities.append(similarity)
            
        # 多样性分数 = 10 * (1 - 最大相似度)
        max_similarity = max(similarities) if similarities else 0
        diversity_score = 10 * (1 - max_similarity)
        
        return max(0.0, min(10.0, diversity_score))
    
    def update_database(self, instructions: List[str]):
        """更新多样性数据库"""
        if instructions:
            embeddings = self.embedding_model.encode(instructions, convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)
            self.instruction_db.extend(instructions)

def save_checkpoint(data: Any, filepath: str):
    """保存检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(data, (list, dict)):
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(data))

def load_checkpoint(filepath: str) -> Any:
    """加载检查点"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith('.json'):
                return json.load(f)
            else:
                return f.read()
    return None