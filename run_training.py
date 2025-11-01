#!/usr/bin/env python3
"""
独立运行模型训练模块 - 修复混合精度问题
"""

import argparse
import time
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model_training.fine_tuning import CodeFineTuner

def run_advanced_debug(scale, data_path):
    """运行高级调试"""
    try:
        from advanced_debug import debug_tensor_creation, debug_model_compatibility, debug_memory_usage
        from config import get_config
        
        config = get_config(scale)
        
        print("🚀 开始高级调试...")
        
        # 1. 调试内存使用
        debug_memory_usage()
        
        # 2. 调试模型兼容性
        if not debug_model_compatibility(config):
            return False
        
        # 3. 调试张量创建
        if not debug_tensor_creation(config, data_path):
            return False
        
        print("\n🎉 所有高级调试检查通过!")
        return True
        
    except Exception as e:
        print(f"❌ 高级调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_mixed_precision():
    """修复混合精度问题"""
    try:
        from fix_mixed_precision import main as fix_main
        fix_main()
        return True
    except Exception as e:
        print(f"❌ 混合精度修复失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="独立模型训练")
    parser.add_argument("--scale", choices=["small", "standard"], default="small",
                       help="运行规模")
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="自定义输出目录")
    parser.add_argument("--fix_data", action="store_true",
                       help="自动修复数据格式")
    parser.add_argument("--debug", action="store_true",
                       help="运行高级调试")
    parser.add_argument("--fix_precision", action="store_true",
                       help="修复混合精度问题")
    
    args = parser.parse_args()
    
    # 如果指定了修复混合精度，只运行修复
    if args.fix_precision:
        print("运行混合精度修复...")
        success = fix_mixed_precision()
        sys.exit(0 if success else 1)
    
    # 获取配置
    config = get_config(args.scale)
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # 如果指定了调试模式，只运行调试
    if args.debug:
        print("运行高级调试...")
        success = run_advanced_debug(args.scale, args.data_path)
        sys.exit(0 if success else 1)
    
    print(f"开始模型训练 (规模: {args.scale})...")
    start_time = time.time()
    
    try:
        # 检查数据文件
        if not Path(args.data_path).exists():
            raise FileNotFoundError(f"训练数据文件不存在: {args.data_path}")
        
        # 如果需要，修复数据格式
        if args.fix_data:
            print("检查并修复数据格式...")
            from check_and_fix_data import check_data_format, fix_data_format
            if not check_data_format(args.data_path):
                print("数据格式有问题，尝试修复...")
                fix_data_format(args.data_path)
        
        # 模型微调
        fine_tuner = CodeFineTuner(config)
        train_dataset, val_dataset = fine_tuner.prepare_data(args.data_path)
        
        # 检查数据集中的样本
        if len(train_dataset) == 0:
            raise ValueError("训练数据集为空，无法进行训练")
        
        print(f"开始训练，训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")
        
        model_path = fine_tuner.fine_tune(train_dataset, val_dataset)
        
        # 保存训练信息
        fine_tuner.save_training_info(train_dataset, val_dataset, model_path)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 模型训练完成! 耗时: {elapsed_time/60:.2f} 分钟")
        print(f"模型保存在: {model_path}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供具体的调试建议
        print("\n🔧 具体解决方案:")
        print("1. 修复混合精度问题: python run_training.py --fix_precision")
        print("2. 运行高级调试: python run_training.py --debug --scale small --data_path YOUR_DATA_PATH")
        print("3. 检查数据格式: python check_and_fix_data.py YOUR_DATA_PATH")
        print("4. 尝试减少批量大小: 修改 config.py 中的 batch_size")
        
        raise

if __name__ == "__main__":
    main()