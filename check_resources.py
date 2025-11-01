#!/usr/bin/env python3
"""
资源检查脚本
用于检测数据集和模型是否下载完成
"""

import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names
import sys

class ResourceChecker:
    def __init__(self, base_dir="./resources"):
        self.base_dir = Path(base_dir)
        self.resources_status = {}
        
    def check_all_resources(self):
        """检查所有资源"""
        print("🔍 开始检查资源状态...")
        print("=" * 60)
        
        self.check_stack_v1()
        self.check_benchmarks()
        self.check_models()
        self.check_training_data()
        
        self.print_summary()
        return self.all_resources_available()
    
    def check_stack_v1(self):
        """检查Stack v1数据集"""
        print("\n📚 检查Stack v1数据集...")
        stack_dir = self.base_dir / "stack_v1"
        seed_file = stack_dir / "seed_samples.json"
        
        if seed_file.exists():
            try:
                with open(seed_file, 'r') as f:
                    data = json.load(f)
                status = f"✓ 已下载 ({len(data)} 个样本)"
                self.resources_status["stack_v1"] = True
            except Exception as e:
                status = f"✗ 文件损坏: {e}"
                self.resources_status["stack_v1"] = False
        else:
            status = "✗ 未找到"
            self.resources_status["stack_v1"] = False
        
        print(f"  Stack v1种子数据: {status}")
    
    def check_benchmarks(self):
        """检查基准数据集"""
        print("\n🏆 检查基准数据集...")
        
        benchmarks = [
            ("HumanEval", "humaneval"),
            ("MBPP", "mbpp")
        ]
        
        for name, dataset_dir in benchmarks:
            benchmark_path = self.base_dir / "benchmarks" / dataset_dir
            
            # 检查本地保存的数据集
            if benchmark_path.exists():
                try:
                    # 检查是否有数据集文件
                    dataset_files = list(benchmark_path.glob("*"))
                    if dataset_files:
                        # 尝试加载JSON测试数据
                        test_file = self.base_dir / "benchmarks" / f"{dataset_dir}_test.json"
                        if test_file.exists():
                            with open(test_file, 'r') as f:
                                test_data = json.load(f)
                            status = f"✓ 已下载 ({len(test_data)} 个样本)"
                        else:
                            status = "✓ 已下载"
                        self.resources_status[f"benchmark_{dataset_dir}"] = True
                    else:
                        status = "✗ 目录为空"
                        self.resources_status[f"benchmark_{dataset_dir}"] = False
                except Exception as e:
                    status = f"✗ 加载失败: {str(e)[:50]}..."
                    self.resources_status[f"benchmark_{dataset_dir}"] = False
            else:
                status = "✗ 未找到"
                self.resources_status[f"benchmark_{dataset_dir}"] = False
            
            print(f"  {name}: {status}")
    
    def check_models(self):
        """检查模型"""
        print("\n🤖 检查模型...")
        
        models_to_check = {
            "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
            "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
            "gte-large": "thenlper/gte-large"
        }
        
        for model_name, model_id in models_to_check.items():
            model_dir = self.base_dir / "models" / model_name
            
            if model_dir.exists():
                try:
                    # 检查模型文件
                    model_files = list(model_dir.glob("*"))
                    essential_files = [
                        "config.json",
                        "pytorch_model.bin",
                        "model.safetensors",
                        "tokenizer.json",
                        "vocab.json"
                    ]
                    
                    # 检查是否有至少一些关键文件
                    has_essential = any((model_dir / f).exists() for f in essential_files[:3])
                    
                    if has_essential and len(model_files) >= 3:
                        status = f"✓ 已下载 ({len(model_files)} 个文件)"
                        self.resources_status[f"model_{model_name}"] = True
                    else:
                        status = "✗ 文件不完整"
                        self.resources_status[f"model_{model_name}"] = False
                except Exception as e:
                    status = f"✗ 检查失败: {str(e)[:30]}..."
                    self.resources_status[f"model_{model_name}"] = False
            else:
                status = "✗ 未找到"
                self.resources_status[f"model_{model_name}"] = False
            
            print(f"  {model_name}: {status}")
    
    def check_training_data(self):
        """检查训练数据"""
        print("\n📊 检查训练数据...")
        
        training_data_paths = {
            "合成数据": self.base_dir.parent / "results" / "synthesized_data" / "training_data.json",
            "高质量数据": self.base_dir.parent / "results" / "synthesized_data" / "instruction_response_pairs_high_quality.json"
        }
        
        for name, path in training_data_paths.items():
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    status = f"✓ 已生成 ({len(data)} 个样本)"
                    self.resources_status[f"training_{name}"] = True
                except Exception as e:
                    status = f"✗ 文件损坏: {e}"
                    self.resources_status[f"training_{name}"] = False
            else:
                status = "✗ 未找到"
                self.resources_status[f"training_{name}"] = False
            
            print(f"  {name}: {status}")
    
    def check_gpu_resources(self):
        """检查GPU资源"""
        print("\n💻 检查GPU资源...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  GPU可用: ✓ ({gpu_count} 个GPU)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                print(f"    GPU {i}: {gpu_name}")
                print(f"      显存: {memory_total:.1f}GB (已分配: {memory_allocated:.1f}GB, 保留: {memory_reserved:.1f}GB)")
        else:
            print("  GPU可用: ✗ (将使用CPU，但性能会受影响)")
    
    def check_disk_space(self):
        """检查磁盘空间"""
        print("\n💾 检查磁盘空间...")
        
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            
            print(f"  总空间: {total // (2**30):,} GB")
            print(f"  已使用: {used // (2**30):,} GB") 
            print(f"  剩余空间: {free // (2**30):,} GB")
            
            if free < 50 * 1024**3:  # 50GB
                print("  ⚠ 警告: 剩余空间不足50GB，可能影响运行")
                return False
            return True
        except:
            print("  无法获取磁盘空间信息")
            return True
    
    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "=" * 60)
        print("📋 资源检查摘要")
        print("=" * 60)
        
        available = sum(1 for status in self.resources_status.values() if status)
        total = len(self.resources_status)
        
        print(f"资源完成度: {available}/{total} ({available/total*100:.1f}%)")
        
        # 按类别显示状态
        categories = {
            "数据集": [k for k in self.resources_status.keys() if k.startswith("benchmark") or k == "stack_v1"],
            "模型": [k for k in self.resources_status.keys() if k.startswith("model")],
            "训练数据": [k for k in self.resources_status.keys() if k.startswith("training")]
        }
        
        for category, keys in categories.items():
            cat_available = sum(1 for k in keys if self.resources_status[k])
            cat_total = len(keys)
            status_icon = "✓" if cat_available == cat_total else "⚠" if cat_available > 0 else "✗"
            print(f"  {category}: {status_icon} ({cat_available}/{cat_total})")
        
        # 检查GPU和磁盘
        self.check_gpu_resources()
        disk_ok = self.check_disk_space()
        
        # 提供建议
        print("\n💡 建议:")
        if available == total:
            print("  ✅ 所有资源已就绪，可以开始运行!")
        else:
            missing = [k for k, v in self.resources_status.items() if not v]
            print(f"  ⚠ 缺少资源: {', '.join(missing)}")
            print("  请运行: python download_resources.py")
        
        if not torch.cuda.is_available():
            print("  ⚠ 没有检测到GPU，训练和推理会很慢")
        
        if not disk_ok:
            print("  ⚠ 磁盘空间不足，请清理空间")
    
    def all_resources_available(self):
        """检查是否所有必需资源都可用"""
        # 必需资源：数据集和模型
        required_keys = [k for k in self.resources_status.keys() 
                        if k.startswith("benchmark") or k.startswith("model") or k == "stack_v1"]
        
        return all(self.resources_status.get(k, False) for k in required_keys)
    
    def get_missing_resources(self):
        """获取缺失的资源列表"""
        return [k for k, v in self.resources_status.items() if not v]

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="资源检查工具")
    parser.add_argument("--base_dir", type=str, default="./resources",
                       help="资源基础目录")
    parser.add_argument("--auto_download", action="store_true",
                       help="自动下载缺失的资源")
    
    args = parser.parse_args()
    
    checker = ResourceChecker(args.base_dir)
    all_available = checker.check_all_resources()
    
    if not all_available and args.auto_download:
        print("\n🔄 自动下载缺失资源...")
        try:
            from download_resources import ResourceDownloader
            downloader = ResourceDownloader(args.base_dir)
            downloader.download_all()
            
            # 重新检查
            print("\n🔄 重新检查资源状态...")
            checker = ResourceChecker(args.base_dir)
            checker.check_all_resources()
        except Exception as e:
            print(f"❌ 自动下载失败: {e}")
    
    # 返回退出代码
    sys.exit(0 if all_available else 1)

if __name__ == "__main__":
    main()