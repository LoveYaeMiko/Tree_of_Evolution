#!/usr/bin/env python3
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfFolder

def check_file(path, description):
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_model(model_name, description):
    try:
        if 'sentence-transformers' in model_name:
            model = SentenceTransformer(model_name)
            print(f"✓ {description}: {model_name}")
            return True
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✓ {description}: {model_name}")
            return True
    except Exception as e:
        print(f"✗ {description}: {model_name} - {e}")
        return False

def check_huggingface_auth():
    """检查 Hugging Face 认证状态"""
    try:
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face 认证: 已登录")
            
            # 测试数据集访问
            from datasets import load_dataset
            try:
                # 尝试访问一个小型公开数据集来验证权限
                dataset = load_dataset('mbpp', split='test', streaming=True)
                sample = next(iter(dataset))
                print("✓ 数据集访问权限: 正常")
                return True
            except Exception as e:
                print(f"✗ 数据集访问权限: 异常 - {e}")
                return False
        else:
            print("✗ Hugging Face 认证: 未登录")
            return False
    except Exception as e:
        print(f"✗ Hugging Face 认证检查失败: {e}")
        return False

def check_dataset_access():
    """检查 the-stack 数据集访问权限"""
    print("检查 the-stack 数据集访问权限...")
    try:
        from datasets import load_dataset
        # 尝试访问 the-stack 的元数据
        dataset = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True)
        # 尝试获取一个样本
        sample = next(iter(dataset))
        if 'content' in sample:
            print("✓ the-stack 数据集: 可访问")
            return True
        else:
            print("✗ the-stack 数据集: 样本格式异常")
            return False
    except Exception as e:
        print(f"✗ the-stack 数据集: 访问失败 - {e}")
        return False

def main():
    print("验证 Tree-of-Evolution 环境...")
    
    # 检查认证
    auth_ok = check_huggingface_auth()
    
    # 检查文件
    files_to_check = [
        ('data/seeds/the_stack_seeds.txt', '种子数据文件'),
        ('configs/small.yaml', '小规模配置文件'),
        ('configs/standard.yaml', '标准规模配置文件'),
    ]
    
    all_files_ok = True
    for path, desc in files_to_check:
        if not check_file(path, desc):
            all_files_ok = False
    
    # 检查模型
    models_to_check = [
        ('Qwen/Qwen2.5-Coder-1.5B', '代码生成模型(1.5B)'),
        ('sentence-transformers/all-MiniLM-L6-v2', '嵌入模型'),
    ]
    
    all_models_ok = True
    for model_name, desc in models_to_check:
        if not check_model(model_name, desc):
            all_models_ok = False
    
    # 检查数据集访问
    dataset_ok = check_dataset_access()
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ GPU不可用 - 需要GPU运行")
        all_models_ok = False
    
    print(f"\n环境状态总结:")
    print(f"  Hugging Face 认证: {'✓' if auth_ok else '✗'}")
    print(f"  数据集访问: {'✓' if dataset_ok else '✗'}")
    print(f"  文件完整性: {'✓' if all_files_ok else '✗'}")
    print(f"  模型可用性: {'✓' if all_models_ok else '✗'}")
    print(f"  GPU可用性: {'✓' if torch.cuda.is_available() else '✗'}")
    
    overall_ok = auth_ok and dataset_ok and all_files_ok and all_models_ok and torch.cuda.is_available()
    print(f"\n总体状态: {'✓ 就绪' if overall_ok else '✗ 需要修复'}")
    
    if not auth_ok:
        print("\n运行以下命令设置认证:")
        print("python scripts/hf_login.py")
        print("或")
        print("huggingface-cli login")
    
    if not dataset_ok and auth_ok:
        print("\n认证成功但数据集访问失败，请确保:")
        print("1. 接受 the-stack 数据集的访问条款")
        print("2. token 有足够的权限")
    
    if not all_files_ok:
        print("\n运行以下命令下载数据:")
        print("python scripts/download_data.py --seeds-only --num-seeds 1000")

if __name__ == "__main__":
    main()