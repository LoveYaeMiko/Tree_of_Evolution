import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import QualityEvaluator, ProgressTracker, save_checkpoint, load_checkpoint
from config import get_config
from model_manager import ModelManager

class TreeOfEvolution:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 确保模型存在
        self._ensure_models_exist()
        
        # 加载模型和tokenizer
        print("加载数据合成模型...")
        self.synthesis_tokenizer = AutoTokenizer.from_pretrained(config.synthesis_model)
        if self.synthesis_tokenizer.pad_token is None:
            self.synthesis_tokenizer.pad_token = self.synthesis_tokenizer.eos_token
        
        self.synthesis_model = AutoModelForCausalLM.from_pretrained(
            config.synthesis_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 初始化质量评估器
        self.evaluator = QualityEvaluator(config.embed_model)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
    
    def _ensure_models_exist(self):
        """确保所需模型存在"""
        manager = ModelManager(self.config)
        
        required_models = {
            "synthesis_model": self.config.synthesis_model,
            "embed_model": self.config.embed_model
        }
        
        for model_type, model_path in required_models.items():
            if not model_path or not os.path.exists(model_path):
                print(f"模型 {model_type} 不存在: {model_path}")
                print("正在下载模型...")
                manager.setup_models(self.config.mode, force_download=self.config.force_download)
                break
        
        # 更新配置路径
        updated_config = manager.load_model_config()
        if updated_config:
            self.config.synthesis_model = updated_config.get("synthesis_model", self.config.synthesis_model)
            self.config.embed_model = updated_config.get("embed_model", self.config.embed_model)

    def generate_instruction_prompt(self, code_snippet):
        """生成指令的提示模板"""
        prompt = f"""你擅长为代码生成创建高质量的[编程问题]。你的任务是根据给定的[代码片段]创建一个高质量的[编程问题]。
步骤1：分析以下[代码片段]：
```python
{code_snippet}
识别关键概念并创建[概念列表]。

步骤2：使用[概念列表]制定创建高质量[编程问题]的详细[计划]。

步骤3：按照[计划]创建[编程问题]。问题可以涉及代码生成、代码编辑、代码调试、数据科学等任务。步骤1中的代码片段"不能"出现在[编程问题]中。

请严格按照以下格式回复：
步骤1 [概念列表]：
步骤2 [计划]：
步骤3 [编程问题]："""
        
        return prompt
    
    def evolve_instruction_prompt(self, old_instruction, challenge_score, diversity_score):
        """进化指令的提示模板"""
        prompt = f"""你擅长为代码生成创建高质量的[编程问题]。你的任务是根据提供的[旧问题]创建一个新的[编程问题]。
        步骤1：回顾提供的[旧问题]：
{old_instruction}

专家根据以下标准评估了[旧问题]的挑战级别：

问题复杂度：问题是否需要高级概念或多个步骤？

概念深度：是否以新的方式挑战基本概念应用？

技能水平：难度是否适合目标受众，并推动他们的能力？

边界情况和优化：是否需要处理边界情况或超越暴力破解的优化？
[旧问题]的挑战级别被评为 {challenge_score}（满分10分）。

另一位专家根据以下标准评估了[旧问题]的多样性：

多样性：问题是否呈现了较少见的编码任务或提供独特的问题解决场景？
[旧问题]的多样性级别被评为 {diversity_score}（满分10分）。

分析[旧问题]并制定详细的[计划]，以创建一个在挑战级别和多样性级别上都超越[旧问题]的新[编程问题]。

步骤2：按照[计划]创建[编程问题]。新问题应该提供：

基于评估标准的更高挑战级别

通过融入独特或较少见的编码任务来实现更大的多样性
你的最终目标是创建一个比提供的[旧问题]更具挑战性和多样性的[编程问题]。

请严格按照以下格式回复：
步骤1 [计划]：
步骤2 [编程问题]："""

        return prompt
    
    def generate_instruction(self, prompt):
        """使用模型生成指令"""
        inputs = self.synthesis_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.synthesis_model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.synthesis_tokenizer.eos_token_id
            )
        
        response = self.synthesis_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_instruction(response)

    def _extract_instruction(self, response):
        """从模型响应中提取指令"""
        if "步骤3 [编程问题]：" in response:
            return response.split("步骤3 [编程问题]：")[-1].strip()
        elif "步骤2 [编程问题]：" in response:
            return response.split("步骤2 [编程问题]：")[-1].strip()
        else:
            # 简单清理响应
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['题目', '问题', 'task', 'question']):
                    return '\n'.join(lines[i:]).strip()
            return response.strip()

    def synthesize_data(self):
        """主数据合成流程"""
        print("开始数据合成流程...")
        
        # 生成或加载种子代码片段
        seed_snippets = self._generate_seed_snippets()
        
        trees = {}
        total_instructions = 0
        total_steps = len(seed_snippets) * self.config.evolution_rounds * self.config.paths_per_node
        progress = ProgressTracker(total_steps, "数据合成")
        
        for seed_id, seed in enumerate(seed_snippets):
            tree = self._build_evolution_tree(seed, seed_id, progress)
            trees[seed_id] = tree
            
            # 保存检查点
            if (seed_id + 1) % 10 == 0:
                checkpoint_data = {
                    'trees': trees,
                    'total_instructions': total_instructions
                }
                save_checkpoint(checkpoint_data, 
                            os.path.join(self.config.output_dir, f'checkpoint_{seed_id}.json'))
        
        progress.close()
        
        # 收集最终数据
        final_data = self._collect_final_data(trees)
        print(f"数据合成完成！共生成 {len(final_data)} 条指令")
        
        # 保存最终数据
        with open(os.path.join(self.config.data_dir, 'synthesized_instructions.json'), 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        return final_data

    def _generate_seed_snippets(self):
        """从Stack v1数据集生成种子代码片段"""
        stack_file = os.path.join(self.config.data_dir, 'stack_v1_snippets.json')
        fallback_file = os.path.join(self.config.data_dir, 'fallback_snippets.json')
        
        if os.path.exists(stack_file):
            print("加载Stack v1代码片段...")
            with open(stack_file, 'r', encoding='utf-8') as f:
                seed_snippets = json.load(f)
        elif os.path.exists(fallback_file):
            print("加载备用代码片段...")
            with open(fallback_file, 'r', encoding='utf-8') as f:
                seed_snippets = json.load(f)
        else:
            print("未找到代码片段文件，正在下载数据集...")
            from data_download import DatasetDownloader
            downloader = DatasetDownloader(self.config)
            seed_snippets = downloader.download_stack_v1(self.config.num_seeds)
        
        # 确保有足够数量的种子
        if len(seed_snippets) < self.config.num_seeds:
            print(f"警告: 只找到 {len(seed_snippets)} 个代码片段，需要 {self.config.num_seeds} 个")
            # 重复使用现有片段或生成更多
            while len(seed_snippets) < self.config.num_seeds:
                seed_snippets.extend(seed_snippets[:self.config.num_seeds - len(seed_snippets)])
        
        return seed_snippets[:self.config.num_seeds]

    def _build_evolution_tree(self, seed, tree_id, progress):
        """为单个种子构建进化树"""
        tree = {'nodes': [], 'edges': []}
        
        # 创建初始指令
        initial_instruction = self._create_initial_instruction(seed)
        initial_node = {
            'instruction': initial_instruction,
            'parent': None, 
            'round': 0,
            'challenge_score': 5.0,
            'diversity_score': 5.0,
            'quality_score': 10.0
        }
        tree['nodes'].append(initial_node)
        
        current_round_nodes = [initial_node]
        
        for round_num in range(1, self.config.evolution_rounds + 1):
            next_round_nodes = []
            current_round_trees = {tree_id: [node for node in tree['nodes'] if node['round'] == round_num - 1]}
            
            for parent_node in current_round_nodes:
                # 为每个节点生成多个进化路径
                for path_num in range(self.config.paths_per_node):
                    progress.update(1, f"树 {tree_id+1}, 轮次 {round_num}")
                    
                    # 生成进化指令
                    evolved_instruction = self._evolve_instruction(parent_node, round_num)
                    
                    if evolved_instruction and evolved_instruction != parent_node['instruction']:
                        # 评估新指令的质量
                        challenge_score = self.evaluator.evaluate_challenge(
                            evolved_instruction, self.synthesis_tokenizer, self.synthesis_model
                        )
                        diversity_score = self.evaluator.evaluate_diversity(
                            evolved_instruction, current_round_trees
                        )
                        quality_score = challenge_score + diversity_score
                        
                        node = {
                            'instruction': evolved_instruction,
                            'parent': parent_node,
                            'round': round_num,
                            'tree_id': tree_id,
                            'challenge_score': challenge_score,
                            'diversity_score': diversity_score,
                            'quality_score': quality_score
                        }
                        next_round_nodes.append(node)
                        tree['nodes'].append(node)
                        tree['edges'].append((parent_node, node))
            
            # 应用束搜索选择最佳节点
            if next_round_nodes:
                current_round_nodes = self._beam_search(next_round_nodes)
            else:
                break
        
        return tree

    def _create_initial_instruction(self, code_snippet):
        """从代码片段创建初始指令"""
        prompt = self.generate_instruction_prompt(code_snippet)
        return self.generate_instruction(prompt)

    def _evolve_instruction(self, parent_node, round_num):
        """进化指令"""
        # 评估父节点的质量分数
        parent_challenge = parent_node.get('challenge_score', 5.0)
        parent_diversity = parent_node.get('diversity_score', 5.0)
        
        prompt = self.evolve_instruction_prompt(
            parent_node['instruction'], parent_challenge, parent_diversity
        )
        
        return self.generate_instruction(prompt)

    def _beam_search(self, nodes):
        """束搜索选择最佳节点"""
        # 按质量分数排序并选择前beam_size%
        nodes.sort(key=lambda x: x['quality_score'], reverse=True)
        beam_size = max(1, int(len(nodes) * self.config.beam_size))
        
        return nodes[:beam_size]

    def _collect_final_data(self, trees):
        """收集最终数据"""
        all_instructions = []
        
        for tree_id, tree in trees.items():
            # 收集所有节点的指令
            for node in tree['nodes']:
                if node.get('quality_score', 0) >= self.config.diversity_threshold:
                    instruction_data = {
                        'instruction': node['instruction'],
                        'challenge_score': node.get('challenge_score', 0),
                        'diversity_score': node.get('diversity_score', 0),
                        'quality_score': node.get('quality_score', 0),
                        'round': node['round'],
                        'tree_id': tree_id
                    }
                    all_instructions.append(instruction_data)
        
        # 按质量分数排序
        all_instructions.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 限制最终数据量
        max_final_data = 75000  # 论文中使用的数量
        if len(all_instructions) > max_final_data:
            all_instructions = all_instructions[:max_final_data]
        
        return all_instructions
    
def main():
    config = get_config()
    print(f"运行模式: {config.mode}")
    print(f"初始种子: {config.num_seeds}")
    print(f"进化轮数: {config.evolution_rounds}")
    toe = TreeOfEvolution(config)
    synthesized_data = toe.synthesize_data()

    print("数据合成完成！")
    print(f"生成指令数量: {len(synthesized_data)}")

if __name__ == "main":
    main()