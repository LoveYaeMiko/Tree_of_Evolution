import importlib
import sys
import os
from pathlib import Path

def check_environment():
    """检查环境配置"""
    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.30.0', 
        'datasets': '2.10.0',
        'accelerate': '0.20.0',
        'sentence_transformers': '2.2.0',
        'faiss': '1.7.0',
        'tqdm': '4.64.0'
    }
    
    missing_packages = []
    version_issues = []
    
    print("=== 环境检查开始 ===")
    
    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                current_version = module.__version__
                if current_version < min_version:
                    version_issues.append(f"{package}: 当前版本 {current_version}, 需要 {min_version}")
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("✗ CUDA 不可用")
    except:
        print("✗ 无法检查CUDA")
    
    # 检查数据目录
    data_dirs = ['data', 'models', 'results']
    for dir_name in data_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ 目录 {dir_name} 已准备")
    
    # 输出结果
    if missing_packages:
        print("\n✗ 缺少的包:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
    else:
        print("\n✓ 所有必需的包都已安装")
        
    if version_issues:
        print("\n⚠ 版本问题:")
        for issue in version_issues:
            print(f"  - {issue}")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    if check_environment():
        print("\n✓ 环境检查通过!")
        sys.exit(0)
    else:
        print("\n✗ 环境检查失败，请安装缺少的包")
        sys.exit(1)