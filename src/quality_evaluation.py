import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import faiss
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

class QualityEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.syn_config = config['synthesis']
        
        # 使用 transformers 加载嵌入模型
        print("Loading embedding model for diversity evaluation...")
        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_dim = 384
            print("✓ 嵌入模型通过 transformers 加载成功")
        except Exception as e:
            print(f"使用 transformers 加载嵌入模型失败: {e}")
            # 备选方案：使用更简单的模型
            try:
                self.embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.embedding_model = AutoModel.from_pretrained('bert-base-uncased')
                self.embedding_dim = 768
                print("✓ 使用 BERT-base 作为备选嵌入模型")
            except Exception as e2:
                print(f"备选嵌入模型也失败: {e2}")
                raise
        
        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.existing_embeddings = []
        
        # 加载挑战性评估模型
        print("Loading challenge evaluation model...")
        self.challenge_tokenizer = AutoTokenizer.from_pretrained(
            self.syn_config.get('eval_model', 'Qwen/Qwen2.5-Coder-1.5B')
        )
        self.challenge_model = AutoModelForCausalLM.from_pretrained(
            self.syn_config.get('eval_model', 'Qwen/Qwen2.5-Coder-1.5B'),
            torch_dtype=torch.float16,  # 保持 torch_dtype
            device_map="auto"
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        try:
            # 使用 transformers 模型获取嵌入
            inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            # 使用 [CLS] token 的嵌入作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            # 归一化
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding[0]
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 返回随机嵌入作为备选
            return np.random.randn(self.embedding_dim)
    
    def evaluate_challenge(self, instruction: str) -> float:
        """评估指令的挑战性 (1-10分)"""
        prompt = f"""评估以下编程问题的挑战性（1-10分，1=非常简单，10=极其困难）：

问题: {instruction}

评估标准：
1. 问题复杂度：是否涉及复杂逻辑或多步骤解决？
2. 概念深度：是否需要理解高级编程概念？
3. 技能要求：是否适合中级到高级学习者？
4. 边界情况：是否需要考虑边界情况或优化？

请直接输出分数（1-10的整数）："""
        
        inputs = self.challenge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = self.challenge_model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.challenge_tokenizer.eos_token_id
            )
        
        response = self.challenge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 从响应中提取分数
        score_text = response.split("请直接输出分数（1-10的整数）：")[-1].strip()
        try:
            # 尝试提取数字
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = min(10, max(1, int(numbers[0])))
                return float(score)
        except:
            pass
        
        # 默认返回中等分数
        return 5.0
    
    def evaluate_diversity(self, instruction: str, existing_instructions: List[str]) -> float:
        """评估指令的多样性 (基于嵌入相似度)"""
        if not existing_instructions:
            return 5.0  # 如果没有现有指令，返回中等多样性分数
        
        # 将新指令转换为嵌入
        new_embedding = self.get_embedding(instruction)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        
        # 将现有指令转换为嵌入
        existing_embeddings = []
        for existing_instruction in existing_instructions:
            embedding = self.get_embedding(existing_instruction)
            existing_embeddings.append(embedding)
        
        if not existing_embeddings:
            return 5.0
            
        existing_embeddings = np.array(existing_embeddings)
        existing_embeddings = existing_embeddings / np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
        
        # 计算与最近邻居的相似度
        similarities = np.dot(existing_embeddings, new_embedding)
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        
        # 将相似度转换为多样性分数 (相似度越低，多样性分数越高)
        diversity_score = 10 * (1 - max_similarity)
        return min(10.0, max(0.0, diversity_score))
    
    def evaluate_instruction(self, instruction: str, existing_instructions: List[str]) -> Tuple[float, float]:
        """评估指令的质量"""
        challenge_score = self.evaluate_challenge(instruction)
        diversity_score = self.evaluate_diversity(instruction, existing_instructions)
        
        return challenge_score, diversity_score