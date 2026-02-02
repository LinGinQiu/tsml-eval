import json
import os
import subprocess
import sys

# ================= 🚀 用户配置区域 =================

# 1. 用户信息
USERNAME = "cq2u24"

# 2. 数据集列表文件路径
DATASET_LIST_FILE = f"/scratch/{USERNAME}/DataSetLists/Longset2-2.txt"

# 3. 分类器固定为 TimesNet
CLASSIFIER = "TimesNet"

# 4. Checkpoint 的种子后缀 (同时也是 Resample ID)
SEED_SUFFIX = "0"

# 5. 路径配置
# TimesNet 专用的运行脚本
SCRIPT_PATH = f"/home/{USERNAME}/tsml-eval/TimesNet/run.py"
# TimesNet 的 Config 文件目录
CONFIG_DIR = f"/home/{USERNAME}/tsml-eval/TimesNet/TSTConfig/"

DATA_DIR_BASE = f"/scratch/{USERNAME}/Data/imbalanced_9_1/"
BASE_RESULTS_ROOT = f"/scratch/{USERNAME}/Exp/results_TimesNet_generative/"
CKPT_ROOT = f"/scratch/{USERNAME}/models/MGD_CVAE/checkpoints/"

# ================= ⚙️ 环境与资源配置 =================

# TimesNet 必须跑在 DL 环境
ENV_DL = f"/scratch/{USERNAME}/conda-envs/deeplearning_tf"

# 队列 (TimesNet 必须用 GPU)
QUEUE_GPU = "a100,swarm_a100,swarm_h100,l4,swarm_l4"

# 资源限制
MEM = "32000M"
TIME = "10:00:00"  # TimesNet 比较慢，设长一点
CORES = 8  # TimesNet 需要更多 CPU 核心

# 是否强制重跑
FORCE_RUN = False


# ================= 🛠️ 核心逻辑 =================


def get_dataset_list(filepath):
    """读取数据集列表文件"""
    if not os.path.exists(filepath):
        print(f"[Error] Dataset list file not found: {filepath}")
        sys.exit(1)

    datasets = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                datasets.append(line)
    return datasets


def get_timesnet_args(dataset_name):
    """
    替代 Bash 脚本中的 grep/cut 逻辑。
    读取 JSON 配置文件并返回参数字符串。
    """
    config_file = os.path.join(CONFIG_DIR, f"{dataset_name}_config.json")

    if not os.path.exists(config_file):
        print(f"[Warning] Config file not found: {config_file}")
        return None

    try:
        with open(config_file) as f:
            cfg = json.load(f)

        # 提取参数，构建 argument string
        # 对应 bash: --e_layers ${e_layers} ...
        args = (
            f"--e_layers {cfg.get('e_layers')} "
            f"--batch_size {cfg.get('batch_size')} "
            f"--d_model {cfg.get('d_model')} "
            f"--d_ff {cfg.get('d_ff')} "
            f"--top_k {cfg.get('top_k')} "
            f"--des '{cfg.get('des')}' "
            f"--itr {cfg.get('itr')} "
            f"--learning_rate {cfg.get('learning_rate')} "
            f"--train_epochs {cfg.get('train_epochs')} "
            f"--patience {cfg.get('patience')}"
        )
        return args

    except Exception as e:
        print(f"[Error] Failed to parse json {config_file}: {e}")
        return None


def submit_jobs():
    # 1. 读取数据集
    datasets = get_dataset_list(DATASET_LIST_FILE)
    print(f"Loaded {len(datasets)} datasets")
    print(f"Classifier: {CLASSIFIER}")
    print(f"Seed/Resample ID: {SEED_SUFFIX}")
    print("=" * 60)

    total_submitted = 0
    total_skipped = 0

    for dataset_name in datasets:
        # 1. 获取 TimesNet 的额外参数
        extra_args = get_timesnet_args(dataset_name)
        if extra_args is None:
            print(f"[Skip] Skipping {dataset_name} due to missing config.")
            continue

        # 2. 构建 Checkpoint 目录
        ckpt_dir = os.path.join(CKPT_ROOT, f"{dataset_name}{SEED_SUFFIX}")

        if not os.path.exists(ckpt_dir):
            print(f"[Skip] No checkpoint dir for {dataset_name}")
            continue

        try:
            files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            files.sort()
        except Exception as e:
            continue

        if not files:
            continue

        print(f"\nProcessing Dataset: {dataset_name} ({len(files)} checkpoints)")

        # 创建日志目录
        log_dir = f"logs/{dataset_name}/{CLASSIFIER}"
        os.makedirs(log_dir, exist_ok=True)

        for i, ckpt_file in enumerate(files):
            ckpt_full_path = os.path.join(ckpt_dir, ckpt_file)
            ckpt_name_clean = ckpt_file.replace(".ckpt", "").replace("=", "_")

            # 变体名称 (也是结果文件夹名称)
            variant_name = f"lgd_prior_v{i}"

            # 结果路径
            this_result_dir = os.path.join(BASE_RESULTS_ROOT, variant_name) + "/"

            # === 检查结果是否存在 ===
            # TimesNet 的输出 CSV
            expected_csv = os.path.join(
                this_result_dir,
                CLASSIFIER,
                "Predictions",
                dataset_name,
                f"testResample{SEED_SUFFIX}.csv",
            )

            if os.path.exists(expected_csv) and not FORCE_RUN:
                total_skipped += 1
                continue

            # Job Name
            job_name = f"{dataset_name[:10]}_{CLASSIFIER}_{ckpt_name_clean[-5:]}"

            # SLURM 脚本内容
            job_content = f"""#!/bin/bash
#SBATCH --partition={QUEUE_GPU}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{ckpt_name_clean}.out
#SBATCH --error={log_dir}/{ckpt_name_clean}.err
#SBATCH --time={TIME}
#SBATCH --mem={MEM}
#SBATCH --cpus-per-task={CORES}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=ecs

module purge
module load conda
source activate {ENV_DL}

export PYTHONPATH=/home/{USERNAME}/tsml-eval
# === 传递 Checkpoint 路径 ===
export LGD_VAE_CHECKPOINT_PATH="{ckpt_full_path}"

# 运行 TimesNet Python 脚本
# 注意这里追加了 extra_args
python -u {SCRIPT_PATH} \
    {DATA_DIR_BASE} \
    {this_result_dir} \
    {CLASSIFIER} \
    {dataset_name} \
    {SEED_SUFFIX} \
    -dtn lgd_prior \
    -tto \
    {extra_args}
"""
            # 写入并提交
            sub_file = f"temp_{job_name}.sh"
            with open(sub_file, "w") as f:
                f.write(job_content)

            subprocess.run(["sbatch", sub_file])
            os.remove(sub_file)
            total_submitted += 1

    print("=" * 60)
    print(
        f"Summary: Submitted {total_submitted} jobs, Skipped {total_skipped} existing jobs."
    )


if __name__ == "__main__":
    submit_jobs()
