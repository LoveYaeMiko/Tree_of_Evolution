#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
from huggingface_hub import login, HfFolder
import json

def setup_huggingface_auth(token: str = None):
    """设置 Hugging Face 认证"""
    if not token:
        # 尝试从环境变量获取
        token = os.getenv('HF_TOKEN')
        if not token:
            # 尝试从 huggingface cli 的缓存中获取
            token = HfFolder.get_token()
    
    if token:
        try:
            login(token=token)
            print("✓ Hugging Face 认证成功")
            return True
        except Exception as e:
            print(f"✗ Hugging Face 认证失败: {e}")
            return False
    else:
        print("未找到 Hugging Face 认证 token")
        print("请通过以下方式之一提供 token:")
        print("1. 设置 HF_TOKEN 环境变量")
        print("2. 使用 huggingface-cli login 命令登录")
        print("3. 在命令行参数中提供 --hf-token")
        return False

def download_stack_seeds(output_file: str, num_samples: int = 5000, hf_token: str = None):
    """从 the-stack 数据集下载种子代码"""
    print(f"下载 {num_samples} 个种子代码从 the-stack 数据集...")
    
    # 设置认证
    if not setup_huggingface_auth(hf_token):
        print("无法访问 the-stack 数据集，请检查认证")
        return []
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    seeds = []
    try:
        # 加载 the-stack 数据集
        print("正在加载 the-stack 数据集...")
        dataset = load_dataset(
            'bigcode/the-stack', 
            data_dir='data/python', 
            split='train', 
            streaming=True,
            token=hf_token
        )
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            if 'content' in sample and sample['content'].strip():
                # 清理代码内容
                code_content = sample['content'].strip()
                # 移除可能的空字符和特殊字符
                code_content = code_content.replace('\x00', '').replace('\ufffd', '')
                
                if len(code_content) > 100:  # 确保有足够的内容
                    seeds.append(code_content)
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1} 个样本，收集到 {len(seeds)} 个种子...")
        
        # 保存种子代码
        with open(output_file, 'w', encoding='utf-8') as f:
            for seed in seeds:
                f.write(seed + '\n')
        
        print(f"成功保存 {len(seeds)} 个种子代码到 {output_file}")
        
    except Exception as e:
        print(f"下载 the-stack 数据集时出错: {e}")
        print("这可能是由于:")
        print("1. 认证 token 无效或过期")
        print("2. 网络连接问题")
        print("3. 数据集访问权限变更")
        return []
    
    return seeds

def download_benchmark_datasets(hf_token: str = None):
    """下载评估基准数据集"""
    print("下载评估基准数据集...")
    
    benchmarks = {
        'humaneval': 'openai/openai_humaneval',
        'mbpp': 'mbpp'
    }
    
    for name, dataset_id in benchmarks.items():
        try:
            print(f"下载 {name}...")
            if hf_token:
                dataset = load_dataset(dataset_id, token=hf_token)
            else:
                dataset = load_dataset(dataset_id)
            print(f"✓ {name}: {len(dataset['test'])} 个问题")
        except Exception as e:
            print(f"✗ 下载 {name} 失败: {e}")

def preload_models_safe(hf_token: str = None):
    """安全地预加载模型文件 - 修复参数错误和依赖问题"""
    print("安全预加载模型文件...")
    
    models_to_check = [
        'Qwen/Qwen2.5-Coder-1.5B',
        'Qwen/Qwen2.5-Coder-7B',
        'sentence-transformers/all-MiniLM-L6-v2'
    ]
    
    for model_name in models_to_check:
        print(f"\n检查 {model_name}...")
        try:
            if 'sentence-transformers' in model_name:
                # 使用 transformers 替代 sentence-transformers 来避免 Keras 冲突
                from transformers import AutoTokenizer, AutoModel
                print("使用 transformers 加载嵌入模型...")
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    model = AutoModel.from_pretrained(model_name, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                print(f"✓ {model_name} 通过 transformers 加载成功")
            else:
                # 修复：移除 torch_dump 参数，使用正确的参数
                from transformers import AutoTokenizer, AutoModelForCausalLM
                print("加载代码生成模型...")
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    # 使用正确的参数
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        token=hf_token,
                        torch_dtype="auto",  # 使用 auto 而不是 torch_dump
                        device_map="auto"
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                print(f"✓ {model_name} 加载成功")
                
        except Exception as e:
            print(f"✗ 加载 {model_name} 失败: {e}")
            
            # 提供具体解决方案
            if "torch_dump" in str(e):
                print("  问题: 使用了不存在的 torch_dump 参数")
                print("  解决方案: 已修复为使用 torch_dtype='auto'")
            elif "Keras" in str(e):
                print("  问题: Keras 版本冲突")
                print("  解决方案: 使用 transformers 替代 sentence-transformers")
            else:
                print("  尝试简化加载...")
                try:
                    # 只加载 tokenizer 来验证模型可访问性
                    from transformers import AutoTokenizer
                    if hf_token:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    print(f"  ✓ {model_name} tokenizer 加载成功 (简化验证)")
                except Exception as e2:
                    print(f"  ✗ 简化加载也失败: {e2}")

def check_huggingface_login():
    """检查是否已登录 Hugging Face"""
    try:
        token = HfFolder.get_token()
        if token:
            print("✓ 检测到 Hugging Face 登录状态")
            return True
        else:
            print("✗ 未检测到 Hugging Face 登录")
            return False
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='下载 Tree-of-Evolution 所需数据和模型')
    parser.add_argument('--seeds-only', action='store_true', help='仅下载种子数据')
    parser.add_argument('--models-only', action='store_true', help='仅预加载模型')
    parser.add_argument('--num-seeds', type=int, default=5000, help='种子代码数量')
    parser.add_argument('--output-dir', type=str, default='data/seeds', help='输出目录')
    parser.add_argument('--hf-token', type=str, help='Hugging Face 认证 token')
    parser.add_argument('--force-login', action='store_true', help='强制重新登录')
    parser.add_argument('--skip-models', action='store_true', help='跳过模型预加载')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs('data/seeds', exist_ok=True)
    os.makedirs('models/pretrained', exist_ok=True)
    
    # 检查登录状态
    if not check_huggingface_login() or args.force_login:
        if args.hf_token:
            setup_huggingface_auth(args.hf_token)
        else:
            print("请提供 Hugging Face 认证 token")
            print("方法1: 使用 --hf-token 参数")
            print("方法2: 运行: huggingface-cli login")
            print("方法3: 设置 HF_TOKEN 环境变量")
            return
    
    if not args.models_only:
        # 下载种子数据
        seed_file = os.path.join(args.output_dir, 'the_stack_seeds.txt')
        download_stack_seeds(seed_file, args.num_seeds, args.hf_token)
    
    if not args.seeds_only and not args.skip_models:
        # 安全预加载模型
        preload_models_safe(args.hf_token)
    
    print("\n下载完成！文件结构：")
    print("data/seeds/the_stack_seeds.txt    # the-stack 种子代码")
    print("~/.cache/huggingface/hub/         # 模型缓存（自动）")
    print("models/pretrained/                # 模型目录")

if __name__ == "__main__":
    main()