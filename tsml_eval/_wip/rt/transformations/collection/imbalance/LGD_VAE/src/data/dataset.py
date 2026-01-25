# src/data/ucr_dataset.py
from os import minor

import numpy as np
import torch
from torch.utils.data import Dataset

# 你已有的工具方法（按你实际所在模块导入）
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.data.uea import load_experiment_data, stratified_resample_data
# 如果暂时没有数据增强，就先注释下面这一行或在 config 里把 augmentation_ratio 设为 0
# from your_pkg.augment import run_augmentation_single

class ZScoreNormalizer:
    """Fit on TRAIN only; apply to both TRAIN/TEST. Works with shape (N, C, T)."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):  # x: (N, C, T)
        m = x.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        s = x.std(axis=(0, 2), keepdims=True) + 1e-8
        self.mean_, self.std_ = m, s
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

class UCRDataset(Dataset):
    def __init__(self, data, labels, split, normalizer=None, augmentation_ratio=0.0,rebalance=False):
        """
        if rebalance:
        we will change the batch to make each X and y is combined both majority class and minority class samples.
        Parameters
        ----------
        data
        labels
        split
        normalizer
        augmentation_ratio
        rebalance
        """
        self.split = split.lower()
        self.labels = labels.astype(np.int64)
        self.augmentation_ratio = augmentation_ratio
        self.rebalance = rebalance
        if normalizer is not None:
            data = normalizer.transform(data)
        self.data = data.astype(np.float32)  # (N, C, T)

        self.class_freq = None
        self.class_inv_weights = None
        self.sample_weights = None
        self.majority_indices = []
        self.minority_indices = []
        if self.split == "train":
            classes, counts = np.unique(self.labels, return_counts=True)
            # === 新增逻辑：计算 Majority Indices ===
            # 1. 找到样本数最多的那个类的标签 (Majority Label)
            # argmax 返回 counts 中最大值的索引
            majority_label = classes[np.argmax(counts)]

            # 2. 找到所有等于该标签的样本索引
            # np.where 返回的是 tuple，取 [0] 获取索引数组
            self.majority_indices = np.where(self.labels == majority_label)[0].tolist()
            self.minority_indices = np.where(self.labels != majority_label)[0].tolist()
            print(f"Dataset Info: Majority Class is {majority_label}, Count: {len(self.majority_indices)}")
            # 类频率 π_c（真正的 prior）
            freq = counts.astype(np.float32) / np.sum(counts)
            self.class_freq = freq           # 顺序与 classes 一致

            # 反比权重 ~ 1/频率
            inv = 1.0 / freq
            self.class_inv_weights = inv / np.sum(inv)

            # 每个样本权重：按标签映射
            # 先做一个 label -> weight 的字典
            class_to_weight = {int(c): float(w) for c, w in zip(classes, self.class_inv_weights)}
            sample_w = np.array(
                [class_to_weight[int(y)] for y in self.labels],
                dtype=np.float32,
            )
            self.sample_weights = torch.from_numpy(sample_w)  # [N]
        else:
            self.class_freq = None
            self.class_inv_weights = None
            self.sample_weights = None

    def __len__(self):
        if self.split == "train" and self.rebalance:
            return len(2*self.majority_indices)
        return self.data.shape[0]

    def __getitem__(self, ind):
        if self.rebalance:
            if self.split == "train":
                if ind%2 ==0:
                    # majority class
                    real_ind = self.majority_indices[ind//2]
                    x = self.data[real_ind]       # (C, T)
                    y = self.labels[real_ind]
                    return (torch.from_numpy(x).float(),torch.from_numpy(x).float()), torch.tensor(y).long()
                else:
                    # minority class
                    real_ind = self.minority_indices[ind//2 % len(self.minority_indices)]
                    mix_ind = np.random.choice(self.minority_indices)
                    x_minority = self.data[mix_ind]
                    recon_x = self.data[real_ind]     # (C, T)
                    # ====== 数据增强部分，可选 ======
                    x = 0.9*recon_x+0.1*x_minority
                    y = self.labels[real_ind]
                    return (torch.from_numpy(x).float(),torch.from_numpy(recon_x).float()), torch.tensor(y).long()

        x = self.data[ind]       # (C, T)
        y = self.labels[ind]
        return (torch.from_numpy(x).float(),torch.from_numpy(x).float()), torch.tensor(y).long()

def load_ucr_splits(problem_path, dataset_name, resample_id=0, predefined_resample=False):
    """
    使用你现有的工具函数读取 UCR，输出标准化数组：
    X_train, X_test: (N, C, T)  —— 统一把原始 (N, 1, L) 转成 (N, C=1, T=L)
    """
    X_train, y_train, X_test, y_test, resample = load_experiment_data(
        problem_path, dataset_name, resample_id, predefined_resample
    )
    # 可选分层重采样
    if resample:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    # 原始往往是 (N, 1, L) or (N, L)；统一成 (N, C, T)
    if X_train.ndim == 2:  # (N, L) -> (N, 1, L)
        X_train = X_train[:, None, :]
    if X_test.ndim == 2:
        X_test = X_test[:, None, :]

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((y_train, y_test)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return X_train, y_train, X_test, y_test
