#!/bin/bash

# Iridis environment creation script
# Environment will be created at $ENV_PATH with Python $PYTHON_VERSION and full GPU support

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
module load anaconda/py3.10

# Create the conda environment
echo "🛠 Creating conda environment at $ENV_PATH..."
conda create -y -p $ENV_PATH python=$PYTHON_VERSION numpy pandas matplotlib scipy scikit-learn sympy tqdm

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_PATH

# Install PyTorch (GLIBC 2.17 compatible) - will auto-select appropriate CUDA version
echo "✅ Installing PyTorch with automatic CUDA compatibility..."
conda install -y pytorch torchvision torchaudio -c pytorch

# Note: Use 'module load cuda/11.7' or 'module load cuda/11.8' on Iridis at runtime if needed.

# Install additional required Python libraries
echo "📦 Installing additional libraries via pip..."
pip install einops==0.8.0 local-attention==1.9.14 patool==1.12 reformer-pytorch==1.4.4 sktime==0.36.0 PyWavelets aeon tsml

# Install development tools
echo "📦 Installing development tools..."
pip install pytest pytest-cov black ruff mypy pre-commit coverage sphinx sphinx_rtd_theme sphinx-autoapi twine build

# Completion message
echo "✅ Iridis environment setup complete!"
echo "👉 You can activate the environment by running: conda activate $ENV_PATH"
