import json
import os

from aeon.datasets import load_from_ts_file

DATASET_LIST_PATH = "local/datalist/full_data_30.txt"
DATA_ROOT = "/Users/qiuchuanhang/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/Downloads/Data/30NewDatasets"
OUTPUT_CONFIG_DIR = "TSTConfig"

os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)

with open(DATASET_LIST_PATH) as f:
    datasets = [line.strip() for line in f if line.strip()]

for dataset in datasets:
    try:
        train_path = os.path.join(DATA_ROOT, dataset, f"{dataset}_TRAIN.ts")
        X, y = load_from_ts_file(train_path)
        n_samples = X.shape[0]
        n_channels = X.shape[1]
        length = X.shape[2]

        if n_samples <= 500 and length <= 100 and n_channels == 1:
            d_model, d_ff = 16, 32
            batch_size = 8
            train_epochs = 20
            patience = 5
            gpus = 1
        elif n_samples <= 2000 and length <= 300 and n_channels <= 3:
            d_model, d_ff = 32, 64
            batch_size = 16
            train_epochs = 30
            patience = 10
            gpus = 1
        else:
            d_model, d_ff = 64, 256
            batch_size = 32
            train_epochs = 50
            patience = 15
            gpus = 2

        config = {
            "dataset": dataset,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "seq_length": length,
            "d_model": d_model,
            "e_layers": 2,
            "batch_size": batch_size,
            "d_ff": d_ff,
            "top_k": 3,
            "des": "Exp",
            "itr": 1,
            "learning_rate": 0.001,
            "train_epochs": train_epochs,
            "patience": patience,
            "gpus": gpus,
        }

        config_path = os.path.join(OUTPUT_CONFIG_DIR, f"{dataset}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"✅ {dataset}: {n_samples}x{n_channels}x{length} → config written")
    except Exception as e:
        print(f"❌ Failed to process {dataset}: {e}")
