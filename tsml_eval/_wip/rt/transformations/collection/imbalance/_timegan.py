import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer

from tsml_eval._wip.rt.transformations.collection.imbalance.TimeGAN.timegan import timegan



class TimeGAN(BaseCollectionTransformer):
    """
    Time-series Generative Adversarial Networks (TimeGAN) for Class Balancing.

    Acts as an oversampler: trains a TimeGAN model on the minority class data
    and generates synthetic samples to balance the class distribution.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
            self,
            module: str = 'gru',
            hidden_dim: int = 24,
            num_layer: int = 3,
            iteration: int = 1000,  # 默认值改小以便测试，原论文可能需要 50000
            batch_size: int = 128,
            random_state=None,
    ):
        """
        Args:
            module: 'gru', 'lstm', or 'lstmLN'
            hidden_dim: hidden dimensions for RNN
            num_layer: number of layers
            iteration: number of training iterations
            batch_size: the number of samples in each batch
            random_state: seed for reproducibility
        """
        self.module = module
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.iteration = iteration
        self.batch_size = batch_size
        self.random_state = random_state

        self._random_state = None
        self.sampling_strategy_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Logic to determine how many samples to generate per class.
        """
        self._random_state = check_random_state(self.random_state)

        # Validate y
        if y is None:
            raise ValueError("y is required for TimeGAN oversampling")

        # Generate sampling target by targeting all classes except the majority
        # (Makes the dataset balanced)
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)

        # Dictionary: {class_label: n_samples_to_generate}
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))

        return self

    def _transform(self, X, y=None):
        """
        Performs the TimeGAN generation logic.
        Expected Input X shape: (n_samples, n_channels, n_timepoints) or (n_samples, n_timepoints)
        """
        X_in = np.asarray(X)

        # 1. 维度调整
        # aeon/tsml 格式通常是 (N, C, L)，但 TimeGAN 需要 (N, L, C)
        if X_in.ndim == 3:
            # (N, C, L) -> (N, L, C)
            X_transposed = np.transpose(X_in, (0, 2, 1))
        else:
            # If (N, L), treat as (N, L, 1)
            X_transposed = X_in[..., np.newaxis]

        n_samples, seq_len, feat_dim = X_transposed.shape

        # 用于收集原始数据和生成的数据
        X_new_list = [X_in]
        y_new_list = [y]

        # 2. 遍历需要过采样的少数类
        for class_label, n_samples_needed in self.sampling_strategy_.items():
            if n_samples_needed <= 0:
                continue

            print(f"Oversampling class {class_label}: generating {n_samples_needed} samples using TimeGAN...")

            # 提取该少数类的数据
            minority_indices = np.where(y == class_label)[0]
            X_minority = X_transposed[minority_indices]  # Shape: (N_min, T, D)

            # 确保数据类型为 float32 (TensorFlow 兼容性)
            X_minority = X_minority.astype(np.float32)

            # 3. 配置 TimeGAN 参数
            # 这些参数对应 main_timegan.py 中的 dict()
            parameters = {
                'module': self.module,
                'hidden_dim': self.hidden_dim,
                'num_layer': self.num_layer,
                'iterations': self.iteration,
                'batch_size': min(self.batch_size, len(X_minority)),  # 防止 batch_size 大于样本数
                'n_samples_to_generate': n_samples_needed  # 这是一个传递给 timegan 的额外参数(见下文注意)
            }

            # 4. 调用 TimeGAN
            # 注意：这里调用的是 timegan.py 中的 timegan 函数
            # input: (N, L, C), output: (N_gen, L, C)
            generated_samples = timegan(X_minority, parameters)

            # 5. 后处理生成的数据
            generated_samples = np.array(generated_samples)


            # 将形状转回 aeon 格式: (N_gen, L, C) -> (N_gen, C, L)
            if X_in.ndim == 3:
                generated_samples = np.transpose(generated_samples, (0, 2, 1))
            else:
                # 如果输入是 2D，输出也转回 2D (N, L)
                generated_samples = generated_samples.squeeze(-1)

            # 生成对应的标签
            generated_labels = np.full((len(generated_samples),), class_label)

            X_new_list.append(generated_samples)
            y_new_list.append(generated_labels)

        # 6. 合并数据
        X_resampled = np.concatenate(X_new_list, axis=0)
        y_resampled = np.concatenate(y_new_list, axis=0)

        return X_resampled, y_resampled


if __name__ == "__main__":
    # 测试代码
    dataset_name = 'MedicalImages'  # 请确保你有这个数据的加载函数
    try:
        from local.load_ts_data import load_ts_data

        X_train, y_train, X_test, y_test = load_ts_data(dataset_name)

        print("Original distribution:", np.unique(y_train, return_counts=True))

        # 初始化 TimeGAN
        # 建议测试时 iteration 设小一点，否则非常慢
        smote = TimeGAN(
            iteration=10,
            batch_size=32,
            hidden_dim=24,
            random_state=42
        )

        X_resampled, y_resampled = smote.fit_transform(X_train, y_train)

        print("Resampled shape:", X_resampled.shape)
        print("Resampled distribution:", np.unique(y_resampled, return_counts=True))

    except ImportError:
        print("Required modules for data loading not found. Please check imports.")
    except Exception as e:
        print(f"An error occurred: {e}")
