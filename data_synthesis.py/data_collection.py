import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from .quality_evaluation import QualityEvaluator

class DataCollector:
    """数据收集器"""
    
    def __init__(self, config):
        self.config = config
        self.evaluator = QualityEvaluator(config)
    
    def collect_data(self, trees: List) -> List[Dict[str, Any]]:
        """从树中收集数据"""
        print("开始收集数据...")
        
        # 收集所有节点
        all_nodes = []
        for tree in trees:
            all_nodes.extend(self._flatten_tree(tree))
        
        # 按轮次分组
        rounds = {}
        for node in all_nodes:
            depth = self._get_node_depth(node)
            if depth not in rounds:
                rounds[depth] = []
            rounds[depth].append(node)
        
        # 按轮次收集数据
        collected_data = []
        for depth in sorted(rounds.keys()):
            round_nodes = rounds[depth]
            # 按质量分数排序
            round_nodes.sort(key=lambda x: x.quality_score, reverse=True)
            
            # 选择高质量节点
            selected_nodes = self._select_diverse_nodes(round_nodes, depth)
            collected_data.extend(selected_nodes)
            
            print(f"轮次 {depth}: 从 {len(round_nodes)} 个节点中选择了 {len(selected_nodes)} 个")
        
        # 转换为指令-响应对
        instruction_response_pairs = self._generate_responses(collected_data)
        
        print(f"数据收集完成，共 {len(instruction_response_pairs)} 个样本")
        return instruction_response_pairs
    
    def _flatten_tree(self, node):
        """展平树结构"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._flatten_tree(child))
        return nodes
    
    def _get_node_depth(self, node):
        """获取节点深度"""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _select_diverse_nodes(self, nodes: List, depth: int) -> List:
        """选择多样化的节点"""
        if depth == 0:
            # 根节点直接选择前50%
            return nodes[:len(nodes)//2]
        
        # 计算节点间的相似度
        selected_nodes = []
        node_instructions = [node.instruction for node in nodes]
        
        # 使用多样性模型编码
        embeddings = self.evaluator.diversity_model.encode(node_instructions)
        
        # 使用FAISS进行多样性选择
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        
        # 贪婪选择：每次选择与已选节点最不相似的节点
        selected_indices = []
        remaining_indices = list(range(len(nodes)))
        
        # 首先选择质量最高的节点
        if remaining_indices:
            selected_indices.append(0)
            remaining_indices.remove(0)
            index.add(embeddings[0:1])
        
        # 选择剩余节点
        while remaining_indices and len(selected_indices) < len(nodes) * 0.75:
            # 计算剩余节点与已选节点的最大相似度
            max_similarities = []
            for idx in remaining_indices:
                query = embeddings[idx:idx+1]
                similarities, _ = index.search(query, k=1)
                max_similarities.append(similarities[0][0])
            
            # 选择相似度最小的节点
            min_sim_idx = np.argmin(max_similarities)
            selected_idx = remaining_indices[min_sim_idx]
            
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
            index.add(embeddings[selected_idx:selected_idx+1])
        
        return [nodes[i] for i in selected_indices]
    
    def _generate_responses(self, nodes: List) -> List[Dict[str, Any]]:
        """为指令生成代码响应"""
        print("生成代码响应...")
        pairs = []
        
        for node in tqdm(nodes, desc="生成响应"):
            response = self._generate_code_response(node.instruction)
            pairs.append({
                "instruction": node.instruction,
                "response": response,
                "challenge_score": node.challenge_score,
                "diversity_score": node.diversity_score,
                "quality_score": node.quality_score
            })
        
        return pairs
    
    def _generate_code_response(self, instruction: str) -> str:
        """生成代码响应"""
        # 使用与进化相同的模型生成响应
        tokenizer = self.evaluator.challenge_tokenizer
        model = self.evaluator.challenge_model
        device = self.evaluator.device
        
        prompt = f"请为以下编程问题生成代码:\n\n{instruction}\n\n代码:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = response.split("代码:")[-1].strip()
        
        return code
    
    def save_collected_data(self, data: List[Dict[str, Any]]):
        """保存收集的数据"""
        output_dir = Path(self.config.output_dir) / "synthesized_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "instruction_response_pairs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"指令-响应对已保存到: {output_file}")