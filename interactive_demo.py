import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ConfigManager

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载完成")
    
    def generate_code(self, instruction: str, max_length: int = 1024) -> str:
        """生成代码"""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取响应部分
        if "### Response:" in response:
            code = response.split("### Response:")[-1].strip()
        else:
            code = response
        
        return code

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="交互式代码生成")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    
    args = parser.parse_args()
    
    generator = CodeGenerator(args.model_path)
    
    print("=" * 50)
    print("代码生成器已启动！")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            instruction = input("\n📝 请输入编程问题: ")
            
            if instruction.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not instruction.strip():
                continue
            
            print("\n🔄 生成中...")
            code = generator.generate_code(instruction)
            
            print(f"\n💻 生成的代码:\n```python\n{code}\n```")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()