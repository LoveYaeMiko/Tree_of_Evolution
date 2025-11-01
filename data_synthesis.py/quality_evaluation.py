import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
import faiss
import numpy as np
from typing import List, Tuple

class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载挑战性评估模型
        print("加载挑战性评估模型...")
        self.challenge_tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.challenge_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载多样性评估模型
        print("加载多样性评估模型...")
        self.diversity_model = SentenceTransformer(config.embedding_model)
        
        # 挑战性评估提示
        self.challenge_prompt = """评估以下编程问题的挑战级别(1-10分):

{instruction}

评分标准:
1-3分: 简单问题，基本编程概念
4-6分: 中等难度，需要算法思维  
7-8分: 困难问题，需要高级概念
9-10分: 极难问题，需要创新思维

只输出分数数字:"""
    
    def evaluate_challenge(self, instruction: str) -> float:
        """评估挑战性"""
        prompt = self.challenge_prompt.format(instruction=instruction)
        
        inputs = self.challenge_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.challenge_model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.challenge_tokenizer.eos_token_id
            )
        
        response = self.challenge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取分数
        try:
            score_text = response.split("分数数字:")[-1].strip()
            score = float(score_text.split()[0])
            return max(1.0, min(10.0, score))  # 限制在1-10之间
        except:
            return 5.0  # 默认分数
    
    def evaluate_diversity(self, instruction: str, existing_instructions: List) -> float:
        """评估多样性"""
        if not existing_instructions:
            return 10.0  # 如果没有现有指令，多样性最高
        
        # 编码所有指令
        all_instructions = [instruction] + [node.instruction for node in existing_instructions]
        embeddings = self.diversity_model.encode(all_instructions)
        
        # 使用FAISS计算相似度
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # 归一化向量
        faiss.normalize_L2(embeddings)
        index.add(embeddings[1:])  # 添加现有指令
        
        # 查询当前指令与现有指令的相似度
        query = embeddings[0:1]
        distances, indices = index.search(query, k=min(5, len(existing_instructions)))
        
        # 计算多样性分数 (1 - 最大相似度)
        max_similarity = distances[0][0] if len(distances[0]) > 0 else 0
        diversity_score = 10 * (1 - max_similarity)  # 转换为0-10分
        
        return max(0.0, min(10.0, diversity_score))
    
    def evaluate_instruction(self, instruction: str, existing_nodes: List) -> Tuple[float, float]:
        """评估指令质量"""
        challenge_score = self.evaluate_challenge(instruction)
        diversity_score = self.evaluate_diversity(instruction, existing_nodes)
        
        return challenge_score, diversity_score