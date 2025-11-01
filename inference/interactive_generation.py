import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import readline  # 用于改进输入体验

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载完成!")
    
    def generate_code(self, instruction: str, max_length: int = 1024) -> str:
        """生成代码"""
        prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = response.split("### 响应:")[-1].strip()
        
        return code
    
    def interactive_session(self):
        """交互式会话"""
        print("=== 代码生成交互模式 ===")
        print("输入 'quit' 退出")
        print("输入 'reset' 清空对话历史")
        print("-" * 50)
        
        while True:
            try:
                instruction = input("\n请输入编程指令: ").strip()
                
                if instruction.lower() == 'quit':
                    break
                elif instruction.lower() == 'reset':
                    print("对话历史已清空")
                    continue
                
                if instruction:
                    print("\n生成代码中...")
                    code = self.generate_code(instruction)
                    print(f"\n生成的代码:\n```python\n{code}\n```")
                    
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python interactive_generation.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    generator = CodeGenerator(model_path)
    generator.interactive_session()