#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=ecsstaff,ecsall
#SBATCH --ntasks=1
#SBATCH -A ecsstaff
#SBATCH --gres=gpu:2
#SBATCH --mem=30G
#SBATCH -c 8
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/cq2u24/test_torch_gpu.out

. /etc/profile

module purge
module load anaconda/py3.10
source activate $env_name

ENV_PATH="/scratch/cq2u24/conda-envs/deeplearning_pytorch"

echo "🔎 Activating conda environment at $ENV_PATH..."
source activate $ENV_PATH

echo "✅ Running PyTorch GPU availability test..."
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
nvidia-smi
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠ No CUDA devices found.')
"

echo "✅ Test complete."
