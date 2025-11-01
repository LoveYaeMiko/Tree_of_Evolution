import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
import json
from tqdm import tqdm
from pathlib import Path

class CodeInstructionDataset(Dataset):
    """代码指令数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # 格式化数据
        self.formatted_data = []
        for item in self.data:
            instruction = item['instruction']
            response = item['response']
            
            # 格式化输入
            text = f"### 指令:\n{instruction}\n\n### 响应:\n{response}"
            self.formatted_data.append(text)
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        text = self.formatted_data[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': encoding['input_ids'].copy()
        }

class CodeFineTuner:
    """代码微调器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和分词器
        print("加载基础模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 设置分词器
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, data_path):
        """准备数据"""
        print("准备训练数据...")
        dataset = CodeInstructionDataset(data_path, self.tokenizer, self.config.max_length)
        
        # 分割训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        return train_dataset, val_dataset
    
    def fine_tune(self, train_dataset, val_dataset):
        """微调模型"""
        print("开始微调...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=Path(self.config.output_dir) / "fine_tuned_model",
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            logging_dir=Path(self.config.output_dir) / "logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print(f"模型微调完成，保存在: {training_args.output_dir}")
        
        return training_args.output_dir