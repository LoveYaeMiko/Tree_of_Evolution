#!/usr/bin/env python3
import os
import sys

def verify_models():
    """验证所有需要的模型是否可用"""
    print("验证 Tree-of-Evolution 模型可用性")
    print("=" * 50)
    
    models_to_verify = [
        {
            'name': 'Qwen/Qwen2.5-Coder-1.5B',
            'type': 'coder',
            'purpose': '代码生成和数据合成'
        },
        {
            'name': 'Qwen/Qwen2.5-Coder-7B', 
            'type': 'coder',
            'purpose': '代码生成和数据合成（标准规模）'
        },
        {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'type': 'embedding',
            'purpose': '多样性评估'
        }
    ]
    
    all_verified = True
    
    for model_info in models_to_verify:
        print(f"\n验证 {model_info['name']}")
        print(f"用途: {model_info['purpose']}")
        
        try:
            if model_info['type'] == 'embedding':
                # 对于嵌入模型，尝试用 transformers 加载
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                model = AutoModel.from_pretrained(model_info['name'])
                print("✓ 嵌入模型可用 (通过 transformers)")
            else:
                # 对于代码模型，加载 tokenizer 和配置
                from transformers import AutoTokenizer, AutoConfig
                tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                config = AutoConfig.from_pretrained(model_info['name'])
                print("✓ 代码模型 tokenizer 和配置可用")
                
                # 尝试轻量级模型加载
                try:
                    from transformers import AutoModelForCausalLM
                    # 只加载到 CPU 并立即卸载，避免内存问题
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['name'],
                        torch_dtype="auto",
                        device_map="cpu"
                    )
                    del model  # 立即释放内存
                    print("✓ 代码模型完整加载成功")
                except Exception as e:
                    print(f"⚠ 代码模型完整加载警告: {e}")
                    print("  (这可能在内存不足时发生，但不影响基本功能)")
            
        except Exception as e:
            print(f"✗ 模型验证失败: {e}")
            all_verified = False
            
            # 提供具体建议
            if "Qwen2ForCausalLM" in str(e) and "torch_dump" in str(e):
                print("  建议: 更新 transformers 库")
                print("  运行: pip install transformers --upgrade")
            elif "Keras" in str(e):
                print("  建议: 安装 tf-keras")
                print("  运行: pip install tf-keras")
            elif "sentence-transformers" in str(e):
                print("  建议: 使用 transformers 替代加载")
    
    print(f"\n模型验证总结: {'所有模型可用 ✓' if all_verified else '部分模型有问题 ⚠'}")
    return all_verified

if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)