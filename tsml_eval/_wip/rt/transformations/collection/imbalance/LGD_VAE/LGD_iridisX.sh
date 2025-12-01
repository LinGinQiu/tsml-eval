#!/bin/bash
# Minimal Iridis submission script for LGD-VAE UCR runs
# - Uses project-level config: <project>/config.yaml
# - One SLURM array index = one random seed (seed = $((SLURM_ARRAY_TASK_ID - 1)))
# - Iterates datasets listed in ${datasets}

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

# Repeats: number of array tasks per dataset (seed = idx-1)
start_fold=1
max_folds=1              # change to how many repeats you want

# Paths
local_path="/home/$username"
data_path="/scratch/$username"
project_dir="$local_path/LGD_VAE"
ckpt_root="$data_path/LGD_VAE"
cfg_path="$project_dir/config.yaml"   # <- project-level YAML
script_file_path="$project_dir/train.py"

# Dataset list file (one dataset name per line)
datasets="$data_path/DataSetLists/cfam_comparison.txt"

# Output dirs (stdout/err logs)
out_dir="$data_path/Results/output/lgd_vae"

# Conda env with PyTorch etc.
env_name="/scratch/$username/conda-envs/deeplearning_pytorch"
# ====================== END USER OPTIONS ======================

set -euo pipefail

# Debug: echo commands (helps when there's "no output")
# set -x  # (debug disabled)

# Fail with line number on any error
# trap 'echo "[ERR] line=$LINENO status=$? cmd=$BASH_COMMAND" >&2' ERR  # (debug disabled)

# Ensure sbatch exists
if ! command -v sbatch >/dev/null 2>&1; then
  echo "[FATAL] sbatch not found in PATH; are you on a login node with Slurm?" >&2
  exit 1
fi

# Ensure key paths exist
[[ -f "$cfg_path" ]] || { echo "[FATAL] cfg_path not found: $cfg_path" >&2; exit 1; }
[[ -f "$script_file_path" ]] || { echo "[FATAL] train.py not found: $script_file_path" >&2; exit 1; }
mkdir -p "$out_dir"

# Validate array bounds
if [[ "$start_fold" -gt "$max_folds" ]]; then
  echo "[FATAL] start_fold ($start_fold) > max_folds ($max_folds)" >&2
  exit 1
fi

# Validate dataset list file exists and is not empty
if [[ ! -s "$datasets" ]]; then
  echo "[FATAL] Dataset list not found or empty: $datasets" >&2
  exit 1
fi

# Show context and a quick preview
# echo "[INFO] CWD=$(pwd)"
# echo "[INFO] Reading datasets from: $datasets"
# head -n 10 "$datasets" | sed -e 's/^/[INFO] LIST: /'

# Normalize: strip CR, drop blanks and comments, then load into an array
mapfile -t DATASETS_ARR < <(tr -d '\r' < "$datasets" | sed -e '/^\s*$/d' -e '/^\s*#/d')
if [[ ${#DATASETS_ARR[@]} -eq 0 ]]; then
  echo "[FATAL] No valid dataset names after filtering comments/blank lines in: $datasets" >&2
  exit 1
fi

# echo "[INFO] Found ${#DATASETS_ARR[@]} dataset(s) to submit"

count=0
for dataset in "${DATASETS_ARR[@]}"; do
  ((count+=1))
  # echo "[LOOP] #$count dataset=$dataset"  # (debug disabled)
  # Sanitize dataset for job-name (avoid spaces/slashes)
  # echo "[DBG] out_dir=$out_dir start_fold=$start_fold max_folds=$max_folds"  # (debug disabled)
  ds_name_sanitized="${dataset//[^A-Za-z0-9_.-]/_}"
  # echo "[DBG] ds_name_sanitized=$ds_name_sanitized"  # (debug disabled)

  # Prepare per-dataset log directory
  ds_out_dir="$out_dir/${dataset}"
  mkdir -p "$ds_out_dir"

  # === Skip dataset if checkpoint exists ===
  ckpt_dir="${ckpt_root}/checkpoints/${dataset}"
  if [[ -d "$ckpt_dir" ]] && ls -1 "$ckpt_dir"/*.ckpt >/dev/null 2>&1; then
    echo "[SKIP] checkpoint detected for dataset=$dataset → skipping submission"
    continue
  fi

  # Build array spec (1..max_folds)
  array_spec="${start_fold}-${max_folds}"

  # Create submission file
  cat > "generatedFile.sub" <<SUB
#!/bin/bash
#SBATCH --partition=${queue}
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --account ecs
#SBATCH -t ${max_time}
#SBATCH --job-name=LGD_VAE_${ds_name_sanitized}
#SBATCH --array=${array_spec}
#SBATCH --mem=${max_memory}M
#SBATCH -o ${ds_out_dir}/%A-%a.out
#SBATCH -e ${ds_out_dir}/%A-%a.err
#SBATCH -c ${cpus_per_task}
#SBATCH --nodes=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=${mailto}

set -eo pipefail
# avoid unbound vars from system bashrc
set +u
. /etc/profile
set -u
module purge
module load conda
source activate ${env_name}

SEED=\$((SLURM_ARRAY_TASK_ID - 1))

python -u ${script_file_path} \
  --cfg ${cfg_path} \
  --dataset_name ${dataset} \
  --seed \$SEED
SUB

  echo "[SUBMIT] ${count}: dataset=${dataset} (job-name=${ds_name_sanitized}), array=${array_spec} → logs: ${ds_out_dir}"
  # head -n 15 "generatedFile_${ds_name_sanitized}.sub" | sed -e 's/^/[DEBUG SUB]: /'  # (debug disabled)
  sbatch < "generatedFile.sub"

  sleep 0.2
done

echo "[DONE] Dispatched ${count} job array(s). Check logs under: $out_dir/<dataset>/*.out|*.err"
