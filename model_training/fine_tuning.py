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
from transformers.trainer_utils import get_last_checkpoint
import json
from tqdm import tqdm
from pathlib import Path
import datasets
import numpy as np

class CodeInstructionDataset(Dataset):
    """代码指令数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载了 {len(self.data)} 个训练样本")
        
        # 打印第一个样本以检查格式
        if len(self.data) > 0:
            print("第一个样本的键:", list(self.data[0].keys()))
        
        self.samples = []
        for item in self.data:
            # 检查数据格式
            if "instruction" not in item:
                print(f"警告: 样本缺少 'instruction' 键: {item.keys()}")
                continue
                
            if "output" not in item:
                print(f"警告: 样本缺少 'output' 键: {item.keys()}")
                # 尝试使用其他可能的键
                if "response" in item:
                    output_content = item["response"]
                elif "code" in item:
                    output_content = item["code"]
                else:
                    print(f"错误: 样本没有输出内容，跳过: {item.keys()}")
                    continue
            else:
                output_content = item["output"]
            
            # 格式化对话
            conversation = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": output_content}
            ]
            
            # 转换为模型输入格式
            text = self.format_conversation(conversation)
            self.samples.append(text)
        
        print(f"有效样本数量: {len(self.samples)}")
    
    def format_conversation(self, conversation):
        """格式化对话"""
        formatted_text = ""
        for turn in conversation:
            if turn["role"] == "user":
                formatted_text += f"### 指令:\n{turn['content']}\n\n"
            elif turn["role"] == "assistant":
                formatted_text += f"### 响应:\n{turn['content']}\n\n"
        
        return formatted_text.strip()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # 编码文本 - 启用填充和截断
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # 不在数据集级别进行填充
            max_length=self.max_length,
            return_tensors=None,
            return_attention_mask=True
        )
        
        # 确保所有值都是整数，并且是平坦的列表
        input_ids = [int(x) for x in encoding['input_ids']]
        attention_mask = [int(x) for x in encoding['attention_mask']]
        
        # 返回平坦的列表，确保没有嵌套结构
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.copy()  # 对于因果语言建模，标签就是输入ID
        }


class SmartDataCollator:
    """智能数据整理器，处理变长序列和数据类型问题"""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # 找到批次中的最大长度
        max_length = max(len(feature['input_ids']) for feature in features)
        
        # 限制最大长度，避免内存问题
        max_length = min(max_length, self.max_length)
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for feature in features:
            # 确保所有值都是整数列表
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            labels = feature['labels']
            
            # 如果超过最大长度，进行截断
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
            # 如果短于最大长度，进行填充
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length  # 忽略填充位置的损失
            
            # 转换为张量
            batch['input_ids'].append(torch.tensor(input_ids, dtype=torch.long))
            batch['attention_mask'].append(torch.tensor(attention_mask, dtype=torch.long))
            batch['labels'].append(torch.tensor(labels, dtype=torch.long))
        
        # 堆叠张量
        batch = {k: torch.stack(v) for k, v in batch.items()}
        return batch


class CodeFineTuner:
    """代码微调器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和分词器
        print("加载基础模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # 确保分词器有填充token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 根据硬件能力选择适当的精度
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到GPU: {gpu_name}")
            
            # 对于较新的GPU（如4090），使用BF16可以获得更好的性能
            if "4090" in gpu_name or "A100" in gpu_name or "H100" in gpu_name:
                self.torch_dtype = torch.bfloat16
                print("使用BF16精度")
            else:
                self.torch_dtype = torch.float16
                print("使用FP16精度")
        else:
            self.torch_dtype = torch.float32
            print("使用FP32精度（CPU模式）")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )
        
        print(f"模型参数量: {self.model.num_parameters():,}")
    
    def prepare_data(self, data_path):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
        
        # 检查数据文件格式
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"训练数据应该是列表格式，但得到的是 {type(data)}")
        
        print(f"训练数据包含 {len(data)} 个样本")
        
        # 创建数据集
        dataset = CodeInstructionDataset(data_path, self.tokenizer, self.config.max_length)
        
        # 分割训练集和验证集
        if len(dataset) > 10:
            train_size = int(0.95 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        else:
            print("警告: 数据集太小，使用全部数据作为训练集")
            train_dataset = dataset
            val_dataset = dataset
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        
        return train_dataset, val_dataset
    
    def fine_tune(self, train_dataset, val_dataset):
        """微调模型"""
        print("开始模型微调...")
        
        # 输出目录
        output_dir = Path(self.config.output_dir) / "fine_tuned_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据精度选择适当的训练参数
        if self.torch_dtype == torch.bfloat16:
            fp16 = False
            bf16 = True
        elif self.torch_dtype == torch.float16:
            fp16 = True
            bf16 = False
        else:
            fp16 = False
            bf16 = False
        
        # 训练参数 - 修复混合精度问题
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            report_to=[],  # 禁用wandb等记录器
            
            # 修复混合精度设置
            fp16=fp16,
            bf16=bf16,
            tf32=True if not fp16 and not bf16 else False,  # 启用TF32以获得更好的性能
            
            dataloader_drop_last=True,
            
            # 梯度裁剪设置
            max_grad_norm=1.0,
            
            # 优化器设置
            optim="adamw_torch",
        )
        
        # 使用智能数据整理器
        data_collator = SmartDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer
        )
        
        # 训练前检查
        if get_last_checkpoint(output_dir) is not None:
            print("发现检查点，从检查点继续训练...")
        
        # 开始训练
        print("开始训练...")
        try:
            trainer.train()
            
            # 保存最终模型
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"模型微调完成，保存在: {output_dir}")
            
            return str(output_dir)
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 提供具体的解决方案
            self._handle_training_error(e, trainer, output_dir)
            raise
    
    def _handle_training_error(self, error, trainer, output_dir):
        """处理训练错误"""
        error_msg = str(error)
        
        if "Attempting to unscale FP16 gradients" in error_msg:
            print("\n🔧 检测到混合精度训练问题，尝试修复...")
            print("解决方案: 使用BF16而不是FP16，或者完全禁用混合精度")
            
            # 尝试保存当前进度
            try:
                trainer.save_model()
                print(f"已保存当前进度到: {output_dir}")
            except:
                print("无法保存当前进度")
        
        elif "CUDA out of memory" in error_msg:
            print("\n🔧 检测到显存不足问题")
            print("解决方案:")
            print("1. 减少批量大小")
            print("2. 使用梯度累积")
            print("3. 使用更小的模型")
            print("4. 启用梯度检查点")
        
        else:
            print("\n🔧 一般性训练错误")
            print("请检查:")
            print("1. 数据格式是否正确")
            print("2. 模型配置是否兼容")
            print("3. 硬件资源是否充足")
    
    def save_training_info(self, train_dataset, val_dataset, output_dir):
        """保存训练信息"""
        info = {
            "base_model": self.config.base_model,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.num_epochs,
            "max_length": self.config.max_length,
            "precision": str(self.torch_dtype)
        }
        
        info_file = Path(output_dir) / "training_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)