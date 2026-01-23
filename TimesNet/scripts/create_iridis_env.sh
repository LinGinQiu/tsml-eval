#!/bin/bash

# IridisX environment creation script
# Environment will be created at $ENV_PATH with Python $PYTHON_VERSION and full GPU support

set -e

ENV_PATH="/scratch/yx1g22/conda-envs/deeplearning_pytorch"
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

# Install PyTorch with fixed CUDA 11.8 support for IridisX
echo "✅ Installing PyTorch with CUDA 11.8 support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional required Python libraries
echo "📦 Installing additional libraries via pip..."
pip install einops==0.8.0 local-attention==1.9.14 patool==1.12 reformer-pytorch==1.4.4 sktime==0.36.0 PyWavelets aeon tsml

# Install development tools
echo "📦 Installing development tools..."
pip install pytest pytest-cov black ruff mypy pre-commit coverage sphinx sphinx_rtd_theme sphinx-autoapi twine build

# Completion message
echo "✅ IridisX environment setup complete!"
echo "👉 You can activate the environment by running: conda activate $ENV_PATH"
