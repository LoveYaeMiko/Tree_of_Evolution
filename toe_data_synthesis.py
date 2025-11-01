import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ConfigManager
from utils import ProgressTracker, QualityEvaluator, save_checkpoint, load_checkpoint

class TreeOfEvolution:
    """Tree-of-Evolution 数据合成器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.training_config.device
        
        # 加载模型
        self._load_models()
        
        # 初始化质量评估器
        self.quality_evaluator = QualityEvaluator(
            embedding_model_path=config.model_config.embedding_model,
            device=self.device
        )
        
        # 初始化树结构
        self.trees = []
        self.current_round = 0
        
    def _load_models(self):
        """加载模型"""
        print("加载模型...")
        
        # 加载合成模型
        synthesis_model_path = os.path.join(
            self.config.model_config.local_base_path, 
            "Qwen2.5-7B-Instruct"
        )
        
        self.synthesis_tokenizer = AutoTokenizer.from_pretrained(
            synthesis_model_path, 
            trust_remote_code=True
        )
        self.synthesis_model = AutoModelForCausalLM.from_pretrained(
            synthesis_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载评估模型（使用同一个模型）
        self.evaluator_tokenizer = self.synthesis_tokenizer
        self.evaluator_model = self.synthesis_model
        
        print("✅ 模型加载完成")
    
    def load_seed_data(self) -> List[str]:
        """加载种子数据"""
        data_config = self.config.get_data_config()
        stack_file = Path("data/stack_v1/sample.jsonl")
        
        seeds = []
        with open(stack_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= data_config['stack_samples']:
                    break
                data = json.loads(line)
                seeds.append(data['content'])
        
        print(f"✅ 加载了 {len(seeds)} 个种子代码片段")
        return seeds
    
    def generate_instruction(self, code_snippet: str, parent_instruction: str = None, 
                           parent_scores: Dict[str, float] = None) -> str:
        """生成指令"""
        
        if parent_instruction is None:
            # 初始生成
            prompt = f"""你是一个擅长创建高质量编程问题的专家。你的任务是根据给定的代码片段创建一个高质量的编程问题。

代码片段:
```python
{code_snippet}
请按照以下步骤进行:

分析代码片段，识别关键概念，创建一个概念列表。

使用概念列表制定创建高质量编程问题的计划。

按照计划创建编程问题。问题可以涉及代码生成、代码编辑、代码调试、数据科学等任务。注意：代码片段中的代码不能直接出现在编程问题中。

请严格按照以下格式回复:
步骤1 [概念列表]:
步骤2 [计划]:
步骤3 [编程问题]:
"""
else:
# 优化演化
prompt = f"""你是一个擅长创建高质量编程问题的专家。你的任务是根据旧的编程问题创建一个新的编程问题。

旧问题:
{parent_instruction}

专家对旧问题的挑战性评估为 {parent_scores['challenge']}/10，多样性评估为 {parent_scores['diversity']}/10。

请创建一个新的编程问题，要求:

具有更高的挑战性：涉及更复杂的逻辑、更高级的概念、需要处理更多边界情况或优化。

具有更高的多样性：提供更独特或更不常见的编码任务。

请严格按照以下格式回复:
步骤1 [计划]:
步骤2 [编程问题]:
"""
        inputs = self.synthesis_tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.synthesis_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.synthesis_tokenizer.eos_token_id
            )
        
        response = self.synthesis_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 解析响应，提取编程问题
        if "步骤3 [编程问题]:" in response:
            question = response.split("步骤3 [编程问题]:")[-1].strip()
        elif "步骤2 [编程问题]:" in response:
            question = response.split("步骤2 [编程问题]:")[-1].strip()
        else:
            question = response.strip()
        
        return question

    def evolve_tree(self, seed: str) -> List[Dict[str, Any]]:
        """演化一个树"""
        data_config = self.config.get_data_config()
        
        # 初始化树
        tree = {
            'seed': seed,
            'nodes': [],
            'rounds': []
        }
        
        # 生成根节点
        root_instruction = self.generate_instruction(seed)
        root_challenge = self.quality_evaluator.evaluate_challenge(
            root_instruction, self.evaluator_model, self.evaluator_tokenizer
        )
        root_diversity = self.quality_evaluator.evaluate_diversity(root_instruction, [])
        
        root_node = {
            'instruction': root_instruction,
            'challenge': root_challenge,
            'diversity': root_diversity,
            'score': root_challenge + root_diversity,
            'depth': 0,
            'parent': None,
            'children': []
        }
        
        tree['nodes'].append(root_node)
        current_nodes = [root_node]
        
        # 多轮演化
        for round_idx in range(data_config['evolution_rounds']):
            print(f"  演化轮次 {round_idx + 1}/{data_config['evolution_rounds']}")
            round_tracker = ProgressTracker(len(current_nodes), desc=f"轮次 {round_idx+1}")
            
            next_nodes = []
            external_db = self._get_external_db(tree, round_idx)
            
            for node in current_nodes:
                # 为每个节点生成多个候选
                candidates = []
                for path_idx in range(data_config['paths_per_node']):
                    candidate_instruction = self.generate_instruction(
                        seed, 
                        node['instruction'],
                        {'challenge': node['challenge'], 'diversity': node['diversity']}
                    )
                    
                    candidate_challenge = self.quality_evaluator.evaluate_challenge(
                        candidate_instruction, self.evaluator_model, self.evaluator_tokenizer
                    )
                    candidate_diversity = self.quality_evaluator.evaluate_diversity(
                        candidate_instruction, external_db
                    )
                    candidate_score = candidate_challenge + candidate_diversity
                    
                    candidate_node = {
                        'instruction': candidate_instruction,
                        'challenge': candidate_challenge,
                        'diversity': candidate_diversity,
                        'score': candidate_score,
                        'depth': round_idx + 1,
                        'parent': node['instruction'][:100],  # 存储父节点指令的前100字符作为标识
                        'children': []
                    }
                    
                    candidates.append(candidate_node)
                
                # Beam Search: 选择top-k候选
                candidates.sort(key=lambda x: x['score'], reverse=True)
                selected_candidates = candidates[:data_config['beam_size']]
                
                # 只保留比父节点好的候选
                for candidate in selected_candidates:
                    if candidate['score'] > node['score']:
                        node['children'].append(candidate)
                        tree['nodes'].append(candidate)
                        next_nodes.append(candidate)
                
                round_tracker.update(1, candidates=len(selected_candidates))
            
            round_tracker.close()
            current_nodes = next_nodes
            
            # 记录本轮结果
            tree['rounds'].append({
                'round': round_idx + 1,
                'node_count': len(next_nodes),
                'avg_score': sum(node['score'] for node in next_nodes) / len(next_nodes) if next_nodes else 0
            })
            
            if not current_nodes:
                break  # 没有合格节点，提前终止
        
        return tree

    def _get_external_db(self, current_tree: Dict[str, Any], round_idx: int) -> List[str]:
        """获取外部数据库用于多样性评估"""
        external_instructions = []
        
        for tree in self.trees:
            if tree != current_tree:
                # 从其他树中选取挑战性最高的指令
                high_challenge_nodes = [node for node in tree['nodes'] if node['depth'] == round_idx]
                if high_challenge_nodes:
                    best_node = max(high_challenge_nodes, key=lambda x: x['challenge'])
                    external_instructions.append(best_node['instruction'])
        
        return external_instructions

    def collect_data(self) -> List[Dict[str, Any]]:
        """收集数据"""
        data_config = self.config.get_data_config()
        all_instructions = []
        
        print("收集数据...")
        collection_tracker = ProgressTracker(len(self.trees), desc="收集数据")
        
        for tree in self.trees:
            # 按轮次收集
            for round_idx in range(data_config['evolution_rounds'] + 1):
                round_nodes = [node for node in tree['nodes'] if node['depth'] == round_idx]
                round_nodes.sort(key=lambda x: x['score'], reverse=True)
                
                selected_in_round = []
                for node in round_nodes:
                    # 检查与同轮次已选节点的相似度
                    if not selected_in_round:
                        selected_in_round.append(node)
                        continue
                    
                    # 计算与已选节点的最小相似度
                    min_similarity = min(
                        self.quality_evaluator.evaluate_diversity(
                            node['instruction'], [selected['instruction']]
                        ) for selected in selected_in_round
                    )
                    
                    # 如果多样性足够高，则选择
                    if min_similarity > data_config['similarity_threshold'] / 10.0:  # 调整阈值
                        selected_in_round.append(node)
                
                all_instructions.extend([
                    {
                        'instruction': node['instruction'],
                        'challenge': node['challenge'],
                        'diversity': node['diversity'],
                        'score': node['score'],
                        'tree_seed': tree['seed'][:100],
                        'depth': node['depth']
                    }
                    for node in selected_in_round
                ])
            
            collection_tracker.update(1)
        
        collection_tracker.close()
        
        # 按分数排序并选择目标数量的样本
        all_instructions.sort(key=lambda x: x['score'], reverse=True)
        final_instructions = all_instructions[:data_config['target_samples']]
        
        print(f"✅ 收集了 {len(final_instructions)} 条指令")
        return final_instructions

    def synthesize_responses(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为指令合成响应"""
        print("合成响应...")
        response_tracker = ProgressTracker(len(instructions), desc="合成响应")
        
        instruction_response_pairs = []
        
        for instruction_data in instructions:
            instruction = instruction_data['instruction']
            
            prompt = f"""请为以下编程问题编写代码解决方案：
{instruction}

请提供完整、正确且高效的代码。"""
            inputs = self.synthesis_tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.synthesis_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.synthesis_tokenizer.eos_token_id
                )
            
            response = self.synthesis_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_code = response.split("```python")[-1].split("```")[0] if "```python" in response else response
            
            pair = {
                'instruction': instruction,
                'response': response_code,
                'challenge': instruction_data['challenge'],
                'diversity': instruction_data['diversity'],
                'score': instruction_data['score']
            }
            instruction_response_pairs.append(pair)
            
            response_tracker.update(1)
        
        response_tracker.close()
        print(f"✅ 合成了 {len(instruction_response_pairs)} 个响应")
        
        return instruction_response_pairs

    def run(self) -> List[Dict[str, Any]]:
        """运行完整的ToE流程"""
        seeds = self.load_seed_data()
        data_config = self.config.get_data_config()
        
        print(f"开始Tree-of-Evolution合成，使用 {len(seeds)} 个种子")
        print("=" * 50)
        
        # 演化多个树
        tree_tracker = ProgressTracker(len(seeds), desc="演化树")
        
        for i, seed in enumerate(seeds):
            print(f"\n演化树 {i+1}/{len(seeds)}")
            tree = self.evolve_tree(seed)
            self.trees.append(tree)
            tree_tracker.update(1)
        
        tree_tracker.close()
        
        # 收集数据
        instructions = self.collect_data()
        
        # 合成响应
        instruction_response_pairs = self.synthesize_responses(instructions)
        
        # 保存最终数据
        output_file = Path("data/synthesized_instructions.jsonl")
        with open(output_file, 'w') as f:
            for pair in instruction_response_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"✅ 合成完成！数据已保存到 {output_file}")
        return instruction_response_pairs
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tree-of-Evolution 数据合成")
    parser.add_argument("--scale", type=str, default="small", choices=["small", "standard"],
                    help="数据规模: small (10小时) 或 standard")

    args = parser.parse_args()

    config = ConfigManager(scale=args.scale)
    toe = TreeOfEvolution(config)
    toe.run()
    
if __name__ == "main":
    main()