import os, warnings, socket

from keras.src.backend.config import max_epochs

from tsml_eval._wip.rt.transformations.collection.imbalance.TimeVAE.data_utils import (
    inverse_transform_data,
    scale_data,
    load_yaml_file
)
from tsml_eval._wip.rt.transformations.collection.imbalance.TimeVAE.vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)


def run_vae_pipeline(train_data, valid_data=None, vae_type: str = 'timeVAE', n_samples=100, dataset_name=None,
                     random_state=None,max_epoch=1000,):
    # ----------------------------------------------------------------------------------
    # 1. Prepare Data
    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)
    _, sequence_length, feature_dim = scaled_train_data.shape

    # ----------------------------------------------------------------------------------
    # 2. Determine Paths
    hostname = socket.gethostname()
    is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname

    # 建议加上 'timeVAE' 这样的具体子目录，或者直接用 random_state 目录
    # 这里保持你原有的逻辑，但在检查时更严谨一点
    if is_mac:
        yaml_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
        base_save_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/local/saved_models/TimeVAE/'
    else:
        yaml_path = '/home/cq2u24/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
        base_save_path = '/scratch/cq2u24/saved_models/TimeVAE/'

    # 构造具体的模型保存目录
    if dataset_name and random_state is not None:
        save_vae_model_path = os.path.join(base_save_path, f"{dataset_name}{random_state}-maxepoch{max_epoch}")
    else:
        # 如果没有传 dataset_name，就用个默认或者临时目录
        save_vae_model_path = os.path.join(base_save_path, "default_temp")

    # TimeVAE 的 save_vae_model 通常会保存两个文件：encoder_wts.h5 和 decoder_wts.h5 (或者 parameters.pkl)
    # 我们检查其中关键的参数文件是否存在
    params_file = os.path.join(save_vae_model_path, f"{vae_type}_parameters.pkl")

    # ----------------------------------------------------------------------------------
    # 3. Check for Existing Model OR Train New Model

    if os.path.exists(params_file):
        print(f"Found existing trained model at: {save_vae_model_path}")
        print("Loading model and skipping training...")

        # 你的 load_vae_model 函数应该接收模型目录
        # 注意：你需要确认 load_vae_model 内部的实现是否会自动处理 .pkl 文件路径
        vae_model = load_vae_model(vae_type=vae_type ,dir_path=save_vae_model_path)

    else:
        print(f"No existing model found at: {save_vae_model_path}")
        print("Starting training from scratch...")

        # 确保目录存在
        os.makedirs(save_vae_model_path, exist_ok=True)

        # load hyperparameters
        hyperparameters = load_yaml_file(yaml_path)[vae_type]

        # instantiate the model
        vae_model = instantiate_vae_model(
            vae_type=vae_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            **hyperparameters,
        )

        # train vae
        train_vae(
            vae=vae_model,
            train_data=scaled_train_data,
            max_epochs=max_epoch,
            verbose=1,
        )

        # save vae
        save_vae_model(vae_model, save_vae_model_path)
        print(f"Model saved to: {save_vae_model_path}")

    # ----------------------------------------------------------------------------------
    # 4. Generate Samples

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=n_samples)

    # inverse transformer samples to original scale
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    generated_samples = inverse_scaled_prior_samples.reshape(n_samples, sequence_length, feature_dim)

    return generated_samples


if __name__ == "__main__":
    # 简单测试逻辑
    # 模拟数据以便测试运行
    import numpy as np

    dummy_data = np.random.rand(100, 24, 1)  # (N, Seq, Feat)

    dataset = "test_dataset"
    model_name = "timeVAE"

    # 第一次运行会训练
    print("--- First Run ---")
    run_vae_pipeline(dummy_data, vae_type=model_name, dataset_name=dataset, random_state=42, max_epoch=10)

    # 第二次运行应该加载
    print("\n--- Second Run ---")
    run_vae_pipeline(dummy_data, vae_type=model_name, dataset_name=dataset, random_state=42,max_epoch=10)
