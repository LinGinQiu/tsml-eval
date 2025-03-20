# Iridis 环境专用环境创建脚本模板
# 保存为 create_iridis_env.sh 后运行

#!/bin/bash

# 设置环境名
env_name=deeplearning_pytorch

echo "创建 Conda 环境：$env_name..."
conda create -y -n $env_name python=3.10 numpy=1.23.5 pandas=1.5.3 matplotlib=3.7.0 scipy=1.10.1 scikit-learn=1.2.2 sympy=1.11.1 tqdm=4.64.1 pytorch=1.7.1 -c conda-forge

# 激活环境
source activate $env_name

echo "开始使用 pip 安装额外依赖..."
pip install \
  einops==0.8.0 \
  local-attention==1.9.14 \
  patool==1.12 \
  reformer-pytorch==1.4.4 \
  sktime==0.16.1 \
  PyWavelets

echo "✅ Iridis 环境创建完成！"
echo "可通过 source activate $env_name 进入环境"