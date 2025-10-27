import torch
import json
import time
from tqdm import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from .quality_evaluation import QualityEvaluator

@dataclass
class EvolutionNode:
    instruction: str
    parent: Any = None
    children: List = None
    quality_score: float = 0.0
    challenge_score: float = 0.0
    diversity_score: float = 0.0
    round: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeEvolution:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.syn_config = config['synthesis']
        
        # 初始化模型
        print("Loading models for tree evolution...")
        model_name = self.syn_config.get('synthesis_model', 'Qwen/Qwen2.5-Coder-1.5B')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 初始化质量评估器
        self.evaluator = QualityEvaluator(config)
        
        # 进化树存储
        self.trees = []
        self.synthesized_data = []
        
    def _generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, 
                      do_sample: bool = True, extraction_keyword: str = "直接输出") -> str:
        """统一的文本生成方法"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True  # 添加padding
        )
        
        # 移动到GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 添加重复惩罚
                top_p=0.9,  # 添加核采样
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 更健壮的文本提取
        if extraction_keyword in response:
            instruction = response.split(extraction_keyword)[-1].strip()
        else:
            # 如果找不到关键字，尝试其他分割方式
            parts = response.split("：")
            if len(parts) > 1:
                instruction = parts[-1].strip()
            else:
                instruction = response.strip()
        
        # 清理可能的标记
        instruction = instruction.replace("<|endoftext|>", "").strip()
        return instruction

    def generate_instruction_from_seed(self, seed_code: str) -> str:
        """从种子代码生成初始指令"""
        prompt = f"""你擅长创建高质量的编程问题。基于以下代码片段创建一个编程问题：

```python
{seed_code}
请创建一个包含以下内容的编程问题：

清晰的题目描述

具体的任务要求

输入输出说明

示例代码（如果需要）

请直接输出编程问题内容："""
        return self._generate_text(prompt, temperature=0.7, extraction_keyword="请直接输出编程问题内容：")

    def evolve_instruction(self, parent_node: EvolutionNode, evolution_round: int) -> List[EvolutionNode]:
        """进化指令 - 生成候选指令"""
        prompt = f"""基于以下编程问题，创建一个更具挑战性和多样性的新问题：
        原问题: {parent_node.instruction}

        原问题挑战性评分: {parent_node.challenge_score}/10
        原问题多样性评分: {parent_node.diversity_score}/10

        请创建一个：

        挑战性更高的新问题（目标: {min(10, parent_node.challenge_score + 1)}/10）

        更具多样性的新问题（目标: {min(10, parent_node.diversity_score + 1)}/10）

        直接输出新问题："""
        candidates = []
        for i in range(self.syn_config['candidates_per_node']):
            new_instruction = self._generate_text(
                prompt, 
                temperature=0.8, 
                extraction_keyword="直接输出新问题："
            )
            
            if new_instruction and len(new_instruction) > 50:
                candidate_node = EvolutionNode(
                    instruction=new_instruction,
                    parent=parent_node,
                    round=evolution_round
                )
                candidates.append(candidate_node)
        
        return candidates

    def beam_search_selection(self, candidates: List[EvolutionNode], beam_size: int) -> List[EvolutionNode]:
        """束搜索选择 - 选择质量最高的候选"""
        if not candidates:
            return []
            
        # 批量评估候选的质量（避免重复调用）
        instructions_to_evaluate = [candidate.instruction for candidate in candidates]
        existing_instructions = [data['instruction'] for data in self.synthesized_data]
        
        # 使用评估器的批量评估方法（如果支持）
        evaluation_results = []
        for instruction in tqdm(instructions_to_evaluate, desc="评估候选指令质量"):
            challenge_score, diversity_score = self.evaluator.evaluate_instruction(
                instruction, 
                existing_instructions
            )
            evaluation_results.append((challenge_score, diversity_score))
        
        # 分配分数
        for candidate, (challenge_score, diversity_score) in zip(candidates, evaluation_results):
            candidate.challenge_score = challenge_score
            candidate.diversity_score = diversity_score
            candidate.quality_score = challenge_score + diversity_score
        
        # 按质量分数排序并选择top-k
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        return candidates[:beam_size]

    def collect_data_from_tree(self, tree_root: EvolutionNode) -> List[Dict]:
        """从树中收集数据"""
        collected_data = []
        
        def traverse(node: EvolutionNode):
            if node.instruction and node.quality_score >= self.syn_config['quality_threshold']:
                # 生成代码响应
                code_response = self.generate_code_response(node.instruction)
                if code_response and len(code_response) > 10:  # 基本验证
                    collected_data.append({
                        'instruction': node.instruction,
                        'response': code_response,
                        'challenge_score': node.challenge_score,
                        'diversity_score': node.diversity_score,
                        'quality_score': node.quality_score,
                        'evolution_round': node.round
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(tree_root)
        return collected_data

    def generate_code_response(self, instruction: str) -> str:
        """为指令生成代码响应"""
        prompt = f"""请为以下编程问题提供Python代码解决方案：
        {instruction}

        要求：

        代码要正确解决问题

        包含必要的注释

        代码风格良好

        处理边界情况

        请直接输出完整的Python代码："""
        return self._generate_text(
            prompt, 
            max_new_tokens=1024, 
            temperature=0.3,  # 降低温度提高代码质量
            do_sample=True,   # 保持采样以获得多样性
            extraction_keyword="请直接输出完整的Python代码："
        )

    def run_evolution(self, seed_codes: List[str]):
        """运行完整的树进化流程"""
        print(f"开始树进化流程，使用 {len(seed_codes)} 个种子...")
        
        total_start_time = time.time()
        
        # 主进度条
        with tqdm(total=len(seed_codes), desc="总体进度") as main_pbar:
            for seed_idx, seed_code in enumerate(seed_codes):
                print(f"\n处理种子 {seed_idx + 1}/{len(seed_codes)}")
                
                # 从种子生成初始指令
                start_time = time.time()
                initial_instruction = self.generate_instruction_from_seed(seed_code)
                initial_node = EvolutionNode(instruction=initial_instruction, round=0)
                
                # 评估初始指令质量
                existing_instructions = [data['instruction'] for data in self.synthesized_data]
                challenge_score, diversity_score = self.evaluator.evaluate_instruction(
                    initial_instruction, existing_instructions
                )
                initial_node.challenge_score = challenge_score
                initial_node.diversity_score = diversity_score
                initial_node.quality_score = challenge_score + diversity_score
                
                print(f"初始指令完成 - 质量分: {initial_node.quality_score:.1f}")
                
                # 构建进化树
                current_level = [initial_node]
                tree_data = [initial_node]
                
                for round_num in range(1, self.syn_config['evolution_rounds'] + 1):
                    print(f"  第 {round_num} 轮进化")
                    
                    round_candidates = []
                    
                    # 为当前层的每个节点生成候选
                    for node_idx, node in enumerate(current_level):
                        if node.quality_score >= self.syn_config['quality_threshold']:
                            candidates = self.evolve_instruction(node, round_num)
                            round_candidates.extend(candidates)
                    
                    if not round_candidates:
                        print(f"    第 {round_num} 轮无合格候选，停止进化")
                        break
                    
                    # 束搜索选择
                    selected_candidates = self.beam_search_selection(
                        round_candidates, 
                        self.syn_config['beam_size']
                    )
                    
                    # 建立父子关系
                    for candidate in selected_candidates:
                        candidate.parent.children.append(candidate)
                        tree_data.append(candidate)
                    
                    current_level = selected_candidates
                    print(f"    选中 {len(selected_candidates)} 个候选")
                
                # 从树中收集数据
                tree_collected_data = self.collect_data_from_tree(initial_node)
                self.synthesized_data.extend(tree_collected_data)
                
                elapsed_time = time.time() - start_time
                print(f"种子 {seed_idx + 1} 完成 - 收集 {len(tree_collected_data)} 个指令 - 用时 {elapsed_time:.1f}s")
                
                # 更新总体进度
                main_pbar.update(1)
                main_pbar.set_postfix({
                    '总指令数': len(self.synthesized_data),
                    '当前种子': f'{seed_idx + 1}/{len(seed_codes)}'
                })
                
                # 检查是否达到目标数据量
                if len(self.synthesized_data) >= self.syn_config['target_data_size']:
                    print(f"达到目标数据量: {len(self.synthesized_data)}")
                    break
        
        total_time = time.time() - total_start_time
        print(f"\n树进化完成，总用时 {total_time:.2f}s")
        print(f"总共合成指令: {len(self.synthesized_data)}")
        
        return self.synthesized_data

    def save_synthesized_data(self, filepath: str):
        """保存合成的数据"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.synthesized_data, f, ensure_ascii=False, indent=2)