import torch
from transformers import DataCollatorForLanguageModeling
from typing import Dict, List, Any

class CustomDataCollator(DataCollatorForLanguageModeling):
    """自定义数据整理器，处理变长序列"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 首先找到批次中的最大长度
        max_length = max(len(feature['input_ids']) for feature in features)
        
        # 确保所有序列都有相同的长度
        for feature in features:
            current_length = len(feature['input_ids'])
            if current_length < max_length:
                # 填充到最大长度
                padding_length = max_length - current_length
                feature['input_ids'] = feature['input_ids'] + [self.tokenizer.pad_token_id] * padding_length
                feature['attention_mask'] = feature['attention_mask'] + [0] * padding_length
                feature['labels'] = feature['labels'] + [-100] * padding_length  # 忽略填充位置的损失
        
        # 现在所有序列长度相同，可以安全地转换为张量
        batch = {}
        for key in features[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']:
                batch[key] = torch.tensor([feature[key] for feature in features])
        
        return batch