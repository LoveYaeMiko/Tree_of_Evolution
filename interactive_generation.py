import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from config import get_config
from model_manager import ModelManager

class CodeGenerator:
    def __init__(self, model_path, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        
        print("加载微调模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate_code(self, instruction, max_length=1024, temperature=0.1):
        """根据指令生成代码"""
        prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code_response = response.replace(prompt, "").strip()
        
        return code_response

def main():
    config = get_config()
    model_path = os.path.join(config.output_dir, "fine_tuned_model")
    
    if not os.path.exists(model_path):
        print(f"微调模型不存在: {model_path}")
        print("请先运行 train_model.py 进行模型训练")
        return
    
    generator = CodeGenerator(model_path, config)
    
    print("代码生成器已启动！")
    print("输入 'quit' 退出程序")
    print("-" * 50)
    
    while True:
        try:
            instruction = input("\n请输入编程问题: ").strip()
            
            if instruction.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not instruction:
                continue
            
            print("\n生成代码中...")
            code = generator.generate_code(instruction)
            
            print("\n生成的代码:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()