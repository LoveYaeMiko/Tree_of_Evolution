# Tree-of-Evolution: 代码生成的树状指令进化框架

本代码库实现了论文《Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models》中提出的方法。

## 项目概述

Tree-of-Evolution (ToE) 是一个用于代码指令合成的创新框架，它通过树状结构和优化驱动的进化过程生成高质量的代码指令数据，显著提升大语言模型的代码生成能力。

## 主要特性

- **树状结构合成**: 探索多个进化路径，避免单向合成的局限性
- **优化驱动进化**: 基于束搜索和质量评估的迭代优化
- **多基准评估**: 支持 HumanEval、MBPP、EvalPlus 等主流代码基准
- **灵活配置**: 支持小规模和标准规模运行模式
- **完整流程**: 从数据准备到模型评估的端到端解决方案

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- 至少 16GB GPU 显存 (小规模版本)
- 建议使用 NVIDIA RTX 4090 或更高性能的 GPU

## 安装依赖

```bash
pip install -r requirements.txt