#!/bin/bash

# Iridis environment creation script with full dependencies from pyproject.toml
# Environment will be created in /scratch to avoid permission issues

set -e

ENV_PATH="/scratch/cq2u24/conda-envs/deeplearning_pytorch"
PYTHON_VERSION="3.10"

# Remove existing environment if present
if [ -d "$ENV_PATH" ]; then
  echo "⚠️ Found existing environment at $ENV_PATH, removing..."
  rm -rf "$ENV_PATH"
fi

# Load the conda module
module purge
module load conda/python3

# Create the conda environment
echo "🛠 Creating conda environment at $ENV_PATH..."
conda create -y -p $ENV_PATH python=$PYTHON_VERSION numpy pandas matplotlib scipy scikit-learn sympy tqdm

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

# Check for GPU presence
if command -v nvidia-smi &> /dev/null; then
  echo "✅ GPU detected. Installing PyTorch with CUDA support..."
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
else
  echo "⚠ No GPU detected. Installing CPU-only PyTorch..."
  conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install additional required Python packages
echo "📦 Installing additional libraries..."
pip install einops==0.8.0 local-attention==1.9.14 patool==1.12 reformer-pytorch==1.4.4 sktime==0.36.0 PyWavelets aeon tsml

# Install development dependencies from pyproject.toml
echo "📦 Installing dev tools..."
pip install pytest pytest-cov black ruff mypy pre-commit coverage sphinx sphinx_rtd_theme sphinx-autoapi twine build

# Completion message
echo "✅ Iridis environment setup complete!"
echo "👉 You can activate the environment by running: conda activate $ENV_PATH"