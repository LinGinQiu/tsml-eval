#!/bin/bash

# Iridis 专用 Conda 环境创建脚本（使用 /scratch 路径）

# 设置环境路径
env_path=/scratch/cq2u24/conda-envs/deeplearning_pytorch

# 自动清理旧环境目录
if [ -d "$env_path" ]; then
    echo "⚠️ 发现已有环境目录，正在删除：$env_path"
    rm -rf "$env_path"
fi

# 加载 Iridis Conda 模块
module purge
module load conda/python3

# 创建环境
conda create -y -p $env_path python=3.10 numpy pandas matplotlib scipy scikit-learn sympy tqdm

# 激活环境
conda activate $env_path

# 使用 pip 安装其他依赖
pip install \
  einops==0.8.0 \
  local-attention==1.9.14 \
  patool==1.12 \
  reformer-pytorch==1.4.4 \
  sktime==0.36.0 \
  PyWavelets

# 日志和提示
echo "✅ Iridis 环境创建完成，已安装全部依赖！"
echo "请使用以下命令激活环境："
echo "conda activate $env_path"
