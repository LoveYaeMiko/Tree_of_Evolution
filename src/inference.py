import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import readline  # 用于改善输入体验

class CodeGenerator:
    def __init__(self, model_path: str):
        print("Loading fine-tuned model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_code(self, instruction: str, max_length: int = 1024) -> str:
        """为给定指令生成代码"""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=max_length,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取响应部分
        if "### Response:" in response:
            code = response.split("### Response:")[-1].strip()
        else:
            code = response
        
        return code
    
    def interactive_session(self):
        """启动交互式代码生成会话"""
        print("=" * 60)
        print("Tree-of-Evolution 代码生成器")
        print("输入 'quit' 退出, 'help' 显示帮助")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n请输入编程问题: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'help':
                    print("\n使用说明:")
                    print("- 输入具体的编程问题描述")
                    print("- 问题越详细，生成的代码越准确")
                    print("- 示例: '写一个Python函数来计算斐波那契数列'")
                    print("- 输入 'quit' 退出")
                    continue
                elif not user_input:
                    continue
                
                print("\n生成代码中...")
                code = self.generate_code(user_input)
                print(f"\n生成的代码:\n```python\n{code}\n```")
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")