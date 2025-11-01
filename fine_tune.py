import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from config import ConfigManager
from utils import ProgressTracker

class CodeFineTuner:
    """代码微调器"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.training_config.device
        
    def load_data(self) -> Dataset:
        """加载数据"""
        data_file = Path("data/synthesized_instructions.jsonl")
        
        instructions = []
        responses = []
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                instructions.append(data['instruction'])
                responses.append(data['response'])
        
        # 构建训练样本
        training_data = []
        for instruction, response in zip(instructions, responses):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}</s>"
            training_data.append({'text': text})
        
        return Dataset.from_list(training_data)
    
    def tokenize_data(self, dataset: Dataset, tokenizer) -> Dataset:
        """标记化数据"""
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.training_config.max_length,
                return_tensors=None
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def setup_lora(self, model):
        """设置LoRA"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        return get_peft_model(model, lora_config)
    
    def train(self):
        """训练模型"""
        training_config = self.config.training_config
        data_config = self.config.get_training_config()
        
        print("加载模型和标记器...")
        
        # 加载基础模型
        model_path = data_config['base_model']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 应用LoRA
        model = self.setup_lora(model)
        model.print_trainable_parameters()
        
        # 加载数据
        print("加载数据...")
        dataset = self.load_data()
        tokenized_dataset = self.tokenize_data(dataset, tokenizer)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=f"outputs/qwen-coder-finetuned-{self.config.scale}",
            overwrite_output_dir=True,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=data_config['batch_size'],
            per_device_eval_batch_size=data_config['batch_size'],
            gradient_accumulation_steps=1,
            warmup_ratio=training_config.warmup_ratio,
            learning_rate=training_config.learning_rate,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            bf16=training_config.mixed_precision == "bf16",
            dataloader_pin_memory=False,
            report_to=None  # 禁用wandb
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
        # 保存模型
        output_dir = f"models/qwen-coder-finetuned-{self.config.scale}"
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 训练完成！模型已保存到 {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="代码模型微调")
    parser.add_argument("--scale", type=str, default="small", choices=["small", "standard"],
                       help="训练规模: small (10小时) 或 standard")
    
    args = parser.parse_args()
    
    config = ConfigManager(scale=args.scale)
    finetuner = CodeFineTuner(config)
    finetuner.train()

if __name__ == "__main__":
    main()