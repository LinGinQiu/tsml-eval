import os, warnings, socket
import gc
import numpy as np

# 新增：导入 TensorFlow 用于管理显存
import tensorflow as tf

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


def run_vae_pipeline(train_data, valid_data=None, vae_type: str = 'TimeVAE', n_samples=100, dataset_name=None,
                     random_state=None, max_epoch=1000, ):
    # ================= [关键修改 1]：防止 TensorFlow 霸占所有显存 =================
    # 这一步告诉 TF：用多少申请多少，不要一口气把 44GB 全吃掉
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 显存增长只能在程序刚启动时设置，如果已经设置过，这里会报错，忽略即可
            print(f"TF Memory growth setting ignored: {e}")
    # ========================================================================

    # ----------------------------------------------------------------------------------
    # 1. Prepare Data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)
    _, sequence_length, feature_dim = scaled_train_data.shape

    # ----------------------------------------------------------------------------------
    # 2. Determine Paths
    hostname = socket.gethostname()
    is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname

    if is_mac:
        yaml_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
        base_save_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/local/saved_models/TimeVAE/'
    else:
        yaml_path = '/home/cq2u24/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
        base_save_path = '/scratch/cq2u24/saved_models/TimeVAE/'

    if dataset_name and random_state is not None:
        save_vae_model_path = os.path.join(base_save_path, f"{dataset_name}{random_state}-maxepoch{max_epoch}")
    else:
        save_vae_model_path = os.path.join(base_save_path, "default_temp")

    params_file = os.path.join(save_vae_model_path, f"{vae_type}_parameters.pkl")

    # ----------------------------------------------------------------------------------
    # 3. Check for Existing Model OR Train New Model

    if os.path.exists(params_file):
        print(f"Found existing trained model at: {save_vae_model_path}")
        print("Loading model and skipping training...")
        vae_model = load_vae_model(vae_type=vae_type, dir_path=save_vae_model_path)

    else:
        print(f"No existing model found at: {params_file}")
        print("Starting training from scratch...")
        os.makedirs(save_vae_model_path, exist_ok=True)

        hyperparameters = load_yaml_file(yaml_path)[vae_type]

        vae_model = instantiate_vae_model(
            vae_type=vae_type,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            **hyperparameters,
        )

        train_vae(
            vae=vae_model,
            train_data=scaled_train_data,
            max_epochs=max_epoch,
            verbose=1,
        )

        save_vae_model(vae_model, save_vae_model_path)
        print(f"Model saved to: {save_vae_model_path}")

    # ----------------------------------------------------------------------------------
    # 4. Generate Samples

    prior_samples = get_prior_samples(vae_model, num_samples=n_samples)

    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    generated_samples = inverse_scaled_prior_samples.reshape(n_samples, sequence_length, feature_dim)

    # ================= [关键修改 2]：强制清理 TensorFlow 显存 =================
    # 这是最重要的一步！如果不加这几行，显存不会释放给 PyTorch
    print("Cleaning up TensorFlow session...")

    # 1. 删除模型对象
    del vae_model

    # 2. 强制 Keras/TF 清空后台 Session
    tf.keras.backend.clear_session()

    # 3. Python 垃圾回收
    gc.collect()

    # 4. 如果显存仍然不释放，可以尝试重置默认图（针对旧版 TF）
    try:
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
    except:
        pass

    print("TensorFlow memory released.")
    # ========================================================================

    return generated_samples

if __name__ == "__main__":
    # 简单测试逻辑
    # 模拟数据以便测试运行
    import numpy as np

    dummy_data = np.random.rand(100, 24, 1)  # (N, Seq, Feat)

    dataset = "test_dataset"
    model_name = "timeVAE"
    #
    # # 第一次运行会训练
    # print("--- First Run ---")
    # run_vae_pipeline(dummy_data, vae_type=model_name, dataset_name=dataset, random_state=42, max_epoch=10)

    # 第二次运行应该加载
    print("\n--- Second Run ---")
    run_vae_pipeline(dummy_data, vae_type=model_name, dataset_name=dataset, random_state=42,max_epoch=10)
