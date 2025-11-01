#!/bin/bash

echo "=== Tree-of-Evolution 标准规模运行脚本 ==="
echo "注意: 标准规模需要大量计算资源和时间"

# 设置环境
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 创建目录
mkdir -p resources results/standard_scale

echo "步骤1: 环境检查..."
python env_check.py
if [ $? -ne 0 ]; then
    echo "环境检查失败!"
    exit 1
fi

echo "步骤2: 下载资源..."
python download_resources.py

echo "步骤3: 数据合成..."
python run_data_synthesis.py --scale standard

echo "步骤4: 模型训练..."
DATA_PATH="./results/standard_scale/synthesized_data/training_data.json"
python run_training.py --scale standard --data_path $DATA_PATH

echo "步骤5: 模型评估..."
MODEL_PATH="./results/standard_scale/fine_tuned_model"
python run_evaluation.py --scale standard --model_path $MODEL_PATH --benchmarks all

echo "🎉 标准规模运行完成!"
echo "模型路径: $MODEL_PATH"
echo "评估结果: ./results/standard_scale/evaluation_results/"