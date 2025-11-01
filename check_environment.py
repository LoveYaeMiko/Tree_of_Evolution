import importlib
import sys
import os
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def check_data_files():
    """检查数据文件"""
    required_files = [
        "data/stack_v1/sample.jsonl",
        "data/humaneval/data.jsonl", 
        "data/mbpp/data.jsonl"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} 存在")
        else:
            print(f"❌ {file} 不存在")
            all_exist = False
            
    return all_exist

def check_models():
    """检查模型文件"""
    required_models = [
        "models/Qwen2.5-Coder-1.5B",
        "models/gte-large-en-v1.5"
    ]
    
    all_exist = True
    for model in required_models:
        if Path(model).exists():
            print(f"✅ {model} 存在")
        else:
            print(f"❌ {model} 不存在")
            all_exist = False
            
    return all_exist

def check_gpu():
    """检查GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU可用: {gpu_name} (数量: {gpu_count})")
            
            # 检查显存
            free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ 显存: {free_memory:.1f} GB")
            return True
        else:
            print("❌ GPU不可用")
            return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def main():
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    # 检查基础包
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss", "faiss"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("wandb", "wandb")
    ]
    
    all_packages_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_packages_ok = False
    
    print("\n" + "=" * 50)
    print("数据检查")
    print("=" * 50)
    data_ok = check_data_files()
    
    print("\n" + "=" * 50) 
    print("模型检查")
    print("=" * 50)
    models_ok = check_models()
    
    print("\n" + "=" * 50)
    print("硬件检查")
    print("=" * 50)
    gpu_ok = check_gpu()
    
    print("\n" + "=" * 50)
    print("检查总结")
    print("=" * 50)
    
    if all_packages_ok and data_ok and models_ok and gpu_ok:
        print("🎉 所有检查通过！可以开始运行项目。")
        return True
    else:
        print("⚠️ 有些检查未通过，请先解决问题。")
        if not all_packages_ok:
            print("请运行: pip install -r requirements.txt")
        if not data_ok:
            print("请运行: python download_data.py")
        if not models_ok:
            print("请运行: python download_models.py")
        return False

if __name__ == "__main__":
    main()