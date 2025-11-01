import json
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .quality_evaluation import QualityEvaluator

class TreeNode:
    """树节点"""
    def __init__(self, instruction: str, code_snippet: str = "", parent=None):
        self.instruction = instruction
        self.code_snippet = code_snippet
        self.parent = parent
        self.children = []
        self.quality_score = 0.0
        self.challenge_score = 0.0
        self.diversity_score = 0.0
        
    def to_dict(self):
        return {
            "instruction": self.instruction,
            "code_snippet": self.code_snippet,
            "quality_score": self.quality_score,
            "challenge_score": self.challenge_score, 
            "diversity_score": self.diversity_score
        }

class TreeEvolution:
    """树进化合成器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        print("加载数据合成模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化评估器
        self.evaluator = QualityEvaluator(config)
        
        # 进化提示
        self.evolution_prompt = """你擅长创建高质量的[编程问题]。请基于给定的[旧问题]创建一个新的[编程问题]。

旧问题:
{old_instruction}

专家评估:
- 挑战级别: {challenge_score}/10
- 多样性级别: {diversity_score}/10

请创建一个新的编程问题，要求:
1. 比旧问题更具挑战性
2. 比旧问题更加多样化
3. 不能包含旧问题中的代码片段

只输出新的编程问题，不要额外解释:"""
        
    def load_seed_data(self) -> List[str]:
        """加载种子数据"""
        seed_file = Path(self.config.data_dir) / "stack_v1" / "seed_samples.json"
        with open(seed_file, 'r') as f:
            data = json.load(f)
        
        # 提取代码片段
        code_snippets = []
        for item in data[:self.config.num_seeds]:
            if 'content' in item:
                code_snippets.append(item['content'])
            elif 'code' in item:
                code_snippets.append(item['code'])
            else:
                # 取第一个字段的值
                code_snippets.append(str(list(item.values())[0]))
                
        return code_snippets
    
    def generate_instruction_from_code(self, code_snippet: str) -> str:
        """从代码片段生成指令"""
        prompt = f"""基于以下代码片段创建一个编程问题:

```python
{code_snippet}
编程问题:"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        instruction = response.split("编程问题:")[-1].strip()
        
        return instruction

    def evolve_instruction(self, node: TreeNode) -> List[TreeNode]:
        """进化指令"""
        candidates = []
        
        for _ in range(self.config.candidates_per_node):
            prompt = self.evolution_prompt.format(
                old_instruction=node.instruction,
                challenge_score=node.challenge_score,
                diversity_score=node.diversity_score
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_instruction = response.split("编程问题:")[-1].strip()
            
            candidate_node = TreeNode(new_instruction, parent=node)
            candidates.append(candidate_node)
            
        return candidates

    def beam_search_evolution(self, root_nodes: List[TreeNode]) -> List[TreeNode]:
        """束搜索进化"""
        current_level = root_nodes
        
        for depth in range(self.config.tree_depth):
            print(f"正在进行第 {depth + 1} 轮进化...")
            next_level = []
            
            for node in tqdm(current_level, desc=f"深度 {depth + 1}"):
                # 生成候选
                candidates = self.evolve_instruction(node)
                
                # 评估质量
                for candidate in candidates:
                    challenge, diversity = self.evaluator.evaluate_instruction(
                        candidate.instruction, current_level
                    )
                    candidate.challenge_score = challenge
                    candidate.diversity_score = diversity
                    candidate.quality_score = challenge + diversity
                
                # 选择top-k候选
                candidates.sort(key=lambda x: x.quality_score, reverse=True)
                selected_candidates = candidates[:self.config.beam_width]
                
                # 只保留质量高于父节点的候选
                for candidate in selected_candidates:
                    if candidate.quality_score > node.quality_score:
                        node.children.append(candidate)
                        next_level.append(candidate)
            
            # 选择下一轮的top节点
            if next_level:
                next_level.sort(key=lambda x: x.quality_score, reverse=True)
                current_level = next_level[:self.config.beam_width * len(current_level)]
            else:
                break
                
        return root_nodes

    def synthesize_data(self) -> List[Dict[str, Any]]:
        """合成数据主函数"""
        print("开始数据合成...")
        
        # 加载种子数据
        code_snippets = self.load_seed_data()
        print(f"加载了 {len(code_snippets)} 个种子代码片段")
        
        all_instructions = []
        root_nodes = []
        
        # 为每个种子创建根节点
        for i, code_snippet in enumerate(tqdm(code_snippets, desc="创建根节点")):
            instruction = self.generate_instruction_from_code(code_snippet)
            root_node = TreeNode(instruction, code_snippet)
            
            # 评估根节点质量
            challenge, diversity = self.evaluator.evaluate_instruction(instruction, [])
            root_node.challenge_score = challenge
            root_node.diversity_score = diversity  
            root_node.quality_score = challenge + diversity
            
            root_nodes.append(root_node)
            all_instructions.append(root_node.to_dict())
        
        # 进行树进化
        evolved_nodes = self.beam_search_evolution(root_nodes)
        
        # 收集所有节点并转换为字典格式
        synthesized_data = self._convert_trees_to_dicts(evolved_nodes)
        
        print(f"数据合成完成，共生成 {len(synthesized_data)} 个指令")
        return synthesized_data
    
    def _convert_trees_to_dicts(self, root_nodes: List[TreeNode]) -> List[Dict[str, Any]]:
        """将树节点转换为字典格式"""
        all_data = []
        
        def node_to_dict(node):
            """递归将节点转换为字典"""
            node_dict = node.to_dict()
            if node.children:
                node_dict["children"] = [node_to_dict(child) for child in node.children]
            return node_dict
        
        for root in root_nodes:
            all_data.append(node_to_dict(root))
        
        return all_data

    def save_synthesized_data(self, data: List[Dict[str, Any]]):
        """保存合成数据"""
        output_dir = Path(self.config.output_dir) / "synthesized_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "synthesized_instructions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"合成数据已保存到: {output_file}")