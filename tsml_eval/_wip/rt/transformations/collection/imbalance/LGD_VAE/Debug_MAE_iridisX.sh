#!/bin/bash
# Quick single-dataset debugging submission script for Ti-MAE UCR runs
# - Runs only one dataset and one repetition (seed=0)
# - No SLURM array jobs

# ======================== USER OPTIONS ========================
username="cq2u24"
mailto="$username@soton.ac.uk"

# Queue/partition (comma-separated ok)
queue="a100,swarm_a100,swarm_h100"

# Resources
max_time="60:00:00"     # hh:mm:ss
max_memory=30000         # MB
cpus_per_task=8
num_gpus=1

# Dataset and seed
dataset_name="ACSF1"
SEED=0

# Paths
local_path="/home/$username"
data_path="/scratch/$username"
project_dir="$local_path/ti-mae-master"
cfg_path="$project_dir/config.yaml"   # <- project-level YAML
script_file_path="$project_dir/train.py"

# Output dirs (stdout/err logs)
out_dir="$data_path/Results/output/ti-mae"

# Conda env with PyTorch etc.
env_name="/scratch/$username/conda-envs/deeplearning_pytorch"
# ====================== END USER OPTIONS ======================

set -euo pipefail

ds_out_dir="$out_dir/${dataset_name}"
mkdir -p "$ds_out_dir"

cat > generatedFile_single.sub <<SUB
#!/bin/bash
#SBATCH --partition=${queue}
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --account ecs
#SBATCH -t ${max_time}
#SBATCH --job-name=TiMAE_${dataset_name}_single
#SBATCH --mem=${max_memory}M
#SBATCH -o ${ds_out_dir}/single_run.out
#SBATCH -e ${ds_out_dir}/single_run.err
#SBATCH -c ${cpus_per_task}
#SBATCH --nodes=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=${mailto}

set -eo pipefail
set +u
. /etc/profile
set -u
module purge
module load conda
source activate ${env_name}

python -u ${script_file_path} \
  --cfg ${cfg_path} \
  --dataset_name ${dataset_name} \
  --seed ${SEED}
SUB

echo "Submitting single run: dataset=${dataset_name}, seed=${SEED}"
sbatch < generatedFile_single.sub

echo "Single submission dispatched."
