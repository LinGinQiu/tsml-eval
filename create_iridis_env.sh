#!/bin/bash

set -e

ENV_PATH="/scratch/cq2u24/conda-envs/deeplearning_pytorch"
PYTHON_VERSION="3.10"

# 删除旧环境
if [ -d "$ENV_PATH" ]; then
  echo "⚠️ Found existing environment at $ENV_PATH, removing..."
  rm -rf "$ENV_PATH"
fi

# 创建 conda 环境
echo "🛠 Creating conda environment..."
conda create -y -p $ENV_PATH python=$PYTHON_VERSION numpy pandas scikit-learn scipy matplotlib sympy tqdm

# 初始化 conda
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

# 检测是否有 GPU
echo "🖥 Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
  echo "✅ GPU detected. Installing PyTorch with CUDA support..."
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
else
  echo "⚠ No GPU detected. Installing CPU-only PyTorch..."
  conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# 安装 Python 其他依赖
echo "📦 Installing additional Python libraries via pip..."
pip install einops==0.8.0 local-attention==1.9.14 patool==1.12 reformer-pytorch==1.4.4 sktime==0.36.0

echo "✅ Environment setup complete!"
echo "👉 You can activate the environment by running:"
echo "conda activate $ENV_PATH"