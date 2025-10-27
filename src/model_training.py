import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from tqdm import tqdm
import os

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.train_config = config['training']
        if isinstance(self.train_config['learning_rate'], str):
            self.train_config['learning_rate'] = float(self.train_config['learning_rate'])
        
    def load_and_prepare_data(self, data_path: str):
        """加载和准备训练数据"""
        print("Loading training data...")
        with open(data_path, 'r', encoding='utf-8') as f:
            synthesized_data = json.load(f)
        
        # 格式化训练样本
        formatted_data = []
        with tqdm(desc="格式化训练数据", total=len(synthesized_data)) as pbar:
            for item in synthesized_data:
                instruction = item['instruction']
                response = item['response']
                
                # 创建训练文本
                training_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                formatted_data.append({'text': training_text})
                pbar.update(1)
        
        return Dataset.from_list(formatted_data)
    
    def setup_lora_config(self):
        """设置LoRA配置"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    
    def train_model(self, data_path: str, output_dir: str):
        """训练模型"""
        # 加载模型和tokenizer
        print("Loading base model...")
        with tqdm(desc="加载基础模型", total=3) as pbar:
            model_name = self.train_config['model_name']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pbar.update(1)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            model.gradient_checkpointing_enable()
            pbar.update(1)
            
            # 设置pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            pbar.update(1)
        
        # 应用LoRA
        with tqdm(desc="配置LoRA", total=1) as pbar:
            lora_config = self.setup_lora_config()
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            pbar.update(1)
        
        # 准备数据
        dataset = self.load_and_prepare_data(data_path)
        
        def tokenize_function(examples):
            # Tokenize文本
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.train_config['max_length'],
                return_tensors=None
            )
            return tokenized
        
        with tqdm(desc="Tokenizing数据", total=1) as pbar:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            pbar.update(1)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.train_config['num_epochs'],
            per_device_train_batch_size=self.train_config['batch_size'],
            per_device_eval_batch_size=self.train_config['batch_size'],
            gradient_accumulation_steps=self.train_config['gradient_accumulation_steps'],
            learning_rate=float(self.train_config['learning_rate']),
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to=None,  # 禁用wandb等记录器
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("Starting training...")
        trainer.train()
        
        # 保存最终模型
        with tqdm(desc="保存模型", total=2) as pbar:
            trainer.save_model()
            pbar.update(1)
            tokenizer.save_pretrained(output_dir)
            pbar.update(1)
        
        print(f"Training completed. Model saved to {output_dir}")
        
        return model, tokenizer