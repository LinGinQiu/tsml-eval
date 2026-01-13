import os, warnings, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.encoder = None
        self.decoder = None

    def fit_on_data(self, train_data, max_epochs=100, verbose=0):
        print('Training VAE model (Manual Loop with Early Stopping)...')

        # 1. 准备数据
        train_data = tf.cast(train_data, tf.float32)
        num_samples = train_data.shape[0]
        steps_per_epoch = num_samples // self.batch_size

        # 2. 初始化早停 (Early Stopping) 相关的变量
        best_loss = float('inf')
        patience = 50  # 容忍多少个 epoch 不下降
        wait = 0  # 当前计数器
        min_delta = 1e-2  # 最小下降幅度

        # 3. 训练循环
        for epoch in range(max_epochs):
            # --- Shuffle 数据 ---
            indices = tf.range(start=0, limit=num_samples, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            train_data_shuffled = tf.gather(train_data, shuffled_indices)

            epoch_loss_sum = 0

            # --- Batch 循环 ---
            for step in range(steps_per_epoch):
                start_idx = step * self.batch_size
                end_idx = start_idx + self.batch_size
                x_batch = train_data_shuffled[start_idx:end_idx]

                # 训练一步
                logs = self.train_step((x_batch, x_batch))
                epoch_loss_sum += logs['loss']

            # --- 计算平均 Loss ---
            avg_loss = epoch_loss_sum / steps_per_epoch

            # 打印进度 (类似 verbose=1)
            print(f"Epoch {epoch + 1}/{max_epochs} - loss: {avg_loss:.4f} "
                  f"- recon_loss: {logs['reconstruction_loss']:.4f} "
                  f"- kl_loss: {logs['kl_loss']:.4f}")

            # --- 早停策略 (Early Stopping) & 保存最佳权重 ---
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                wait = 0
                # 保存当前最好的权重到内存 (或者保存到文件)
                best_weights = self.get_weights()
                # print("  [New best loss, weights recorded]")
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    # 【重要】恢复到之前最好的权重
                    self.set_weights(best_weights)
                    break

            # (可选) 如果 Loss 变成 NaN，立即停止
            if np.isnan(avg_loss):
                print("Loss is NaN, stopping training.")
                break

        print("Training finished.")

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_num_trainable_variables(self):
        trainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
        )
        nonTrainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
        )
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams

    def get_prior_samples(self, num_samples):
        print("Getting prior samples (Direct Call Mode)...")
        # 1. 生成随机噪声
        Z = np.random.randn(num_samples, self.latent_dim)

        # 【关键修复1】强制转为 float32 (Mac 必须)
        Z = tf.cast(Z, tf.float32)

        # 【关键修复2】不要用 .predict(Z)，直接像调用函数一样调用 decoder
        # 这会直接触发前向传播，避开 Keras 的调度死锁
        samples_tensor = self.decoder(Z, training=False)

        # 转回 numpy
        samples = samples_tensor.numpy()

        print("Done. samples shape:", samples.shape)
        return samples

    def get_prior_samples_given_Z(self, Z):
        # 同样改为直接调用
        # 1. 确保是 float32
        Z = tf.cast(Z, tf.float32)

        # 2. 直接调用
        samples_tensor = self.decoder(Z, training=False)

        return samples_tensor.numpy()

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_c, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            X = data[0]
        else:
            X = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(
            encoder_wts, os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        joblib.dump(
            decoder_wts, os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

    def load_weights(self, model_dir):
        encoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        decoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == "__main__":
    pass
