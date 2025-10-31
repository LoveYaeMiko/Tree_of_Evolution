import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import os
from utils import ProgressTracker
from config import get_config
from model_manager import ModelManager

class CodeInstructionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 确保模型存在
        self._ensure_models_exist()
        
        print("加载基础模型和tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def _ensure_models_exist(self):
        """确保基础模型存在"""
        manager = ModelManager(self.config)
        
        if not self.config.base_model or not os.path.exists(self.config.base_model):
            print(f"基础模型不存在: {self.config.base_model}")
            print("正在下载模型...")
            manager.setup_models(self.config.mode, force_download=self.config.force_download)
            
            # 更新配置路径
            updated_config = manager.load_model_config()
            if updated_config:
                self.config.base_model = updated_config.get("base_model", self.config.base_model)

    def prepare_training_data(self):
        """准备训练数据"""
        data_path = os.path.join(self.config.data_dir, 'synthesized_instructions.json')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"训练数据不存在: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            instructions_data = json.load(f)
        
        # 为每个指令生成代码响应
        training_samples = []
        progress = ProgressTracker(len(instructions_data), "准备训练数据")
        
        for i, item in enumerate(instructions_data):
            progress.update(1, f"准备训练样本 {i+1}/{len(instructions_data)}")
            
            instruction = item['instruction']
            response = self._generate_code_response(instruction)
            
            training_sample = {
                'instruction': instruction,
                'response': response,
                'text': f"### 指令:\n{instruction}\n\n### 响应:\n{response}"
            }
            training_samples.append(training_sample)
        
        progress.close()
        return training_samples
    
    def _generate_code_response(self, instruction):
        """为指令生成代码响应"""
        prompt = f"""请为以下编程问题生成Python代码解决方案：

{instruction}

请提供完整的、可运行的代码："""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取代码部分
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.replace(prompt, "").strip()
        
        return code
    
    def train(self):
        """训练模型"""
        print("准备训练数据...")
        training_data = self.prepare_training_data()
        
        # 创建数据集
        dataset = Dataset.from_list(training_data)
        
        def tokenize_function(examples):
            # 对文本进行分词
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            # 对于因果语言建模，标签与输入相同
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=0.01,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # 禁用wandb等记录器
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("开始训练...")
        trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.output_dir, "fine_tuned_model")
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print(f"训练完成！模型已保存到: {final_model_path}")

def main():
    config = get_config()
    trainer = CodeInstructionTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()