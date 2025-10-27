#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"正在 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} 成功")
            return True
        else:
            print(f"✗ {description} 失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description} 异常: {e}")
        return False

def main():
    """专门修复模型加载的依赖问题"""
    print("修复 Tree-of-Evolution 模型加载依赖问题")
    print("=" * 50)
    
    # 专门针对模型加载问题的修复
    fixes = [
        # 解决 Keras 3 兼容性问题
        ("pip install tf-keras", "安装 tf-keras 解决 Keras 兼容性"),
        
        # 确保 transformers 是最新版本
        ("pip install transformers>=4.35.0 --upgrade", "更新 transformers"),
        
        # 安装必要的模型运行依赖
        ("pip install accelerate>=0.24.0", "安装 accelerate"),
        ("pip install peft>=0.7.0", "安装 PEFT"),
        
        # 安装嵌入计算依赖
        ("pip install sentence-transformers>=2.2.0", "安装 sentence-transformers"),
        ("pip install faiss-cpu>=1.7.0", "安装 FAISS"),
    ]
    
    success_count = 0
    for cmd, desc in fixes:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n修复完成: {success_count}/{len(fixes)} 个步骤成功")
    
    if success_count == len(fixes):
        print("✓ 所有模型依赖问题已修复")
    else:
        print("⚠ 部分依赖问题可能需要手动修复")

if __name__ == "__main__":
    main()