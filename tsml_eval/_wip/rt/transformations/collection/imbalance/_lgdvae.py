from collections import OrderedDict
from os import major
from typing import Optional, Union
import tqdm
import numpy as np
from numba import prange
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.lgd_pipline import LGDVAEPipeline
plt.close('all')
import glob
from tsml_eval._wip.rt.transformations.collection.imbalance._utils import _plot_series_list
from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
from tsml_eval._wip.rt.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer

__all__ = ["VOTE"]

from tsml_eval._wip.rt.utils._threading import threaded
from typing import Tuple, Dict

import os
import socket
from pathlib import Path
import torch
import sys


def get_best_ckpt_path(ckpt_dir: str | Path) -> str:
    """
    Return the path to the latest non-last checkpoint file in a directory.
    Priority: ti-mae-epoch=*.ckpt > last.ckpt
    """
    pattern = os.path.join(ckpt_dir, "ti-mae-epoch=*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    if ckpts:
        return ckpts[-1]  # 最新的那个
    # fallback
    last_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_path):
        return last_path
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


class VOTE(BaseCollectionTransformer):
    """
    Over-sampling using the Synthetic Minority Over-sampling TEchnique (SMOTE)[1]_.
    An adaptation of the imbalance-learn implementation of SMOTE in
    imblearn.over_sampling.SMOTE. sampling_strategy is sampling target by
    targeting all classes but not the majority, which is directly expressed in
    _fit.sampling_strategy.
    Parameters
    ----------
    n_neighbors : int, default=5
        The number  of nearest neighbors used to define the neighborhood of samples
        to use to generate the synthetic time series.
        `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this case.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    See Also
    --------
    ADASYN
    References
    ----------
    .. [1] Chawla et al. SMOTE: synthetic minority over-sampling technique, Journal
    of Artificial Intelligence Research 16(1): 321–357, 2002.
        https://dl.acm.org/doi/10.5555/1622407.1622416
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
            self,
            dataset_name: str = "",
            mode: str = 'mix ',
            visualize: bool = False,
            n_jobs: int = 1,
            random_state=None,
    ):
        self.random_state = random_state
        self.mode = mode
        self.n_jobs = n_jobs
        self.dataset_name = dataset_name
        self._random_state = None

        self.visualize = visualize
        self._root = None
        self._device = None
        self.n_generate_samples = None
        self._cls_maj = None
        self._cls_min = None
        self._generated_samples = None
        self._init_model()
        super().__init__()

    def _init_model(self):
        # ---- Environment detection ----
        hostname = socket.gethostname()
        is_iridis = "iridis" in hostname.lower() or "loginx" in hostname.lower()
        is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名

        if is_iridis:
            print("[ENV] Detected Iridis HPC environment")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif is_mac:
            print("[ENV] Detected local macOS environment")
            self._device = torch.device("cpu")
        else:
            print("[ENV] Unknown environment, fallback to current dir")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X, y=None):
        # Make sure inference.py can be imported
        print(f"device in oversampler is {self._device}")
        self._random_state = check_random_state(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        label_majority = classes[np.argmax(counts)]
        label_minority = classes[np.argmin(counts)]
        self._cls_maj = label_majority
        self._cls_min = label_minority
        n_generate_samples = counts[np.argmax(counts)] - counts[np.argmin(counts)]
        self.n_generate_samples = n_generate_samples
        self.pipeline = LGDVAEPipeline(dataset_name=self.dataset_name, seed=self.random_state, device=self._device)
        self.pipeline.fit(X_tr=X, y_tr=y)
        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        C, L = X.shape[1], X.shape[2]
        is_maj = (y == self._cls_maj)
        X_majority = X[is_maj]
        X_minority = X[~is_maj]
        if self.visualize:
            _plot_series_list(X_majority[:10], title="Majority Classes")
            _plot_series_list(X_minority[:10], title="Minority Classes")
        new_ts = np.zeros([self.n_generate_samples, C, L])
        new_ys = np.full([self.n_generate_samples], fill_value=self._cls_min, dtype=y.dtype)

        if self.mode == "latent":
            for i in range(self.n_generate_samples):
                index1, index2 = self._random_state.choice(X_minority.shape[0], size=2, replace=False)
                x_min_1 = torch.from_numpy(X_minority[index1][np.newaxis, :]).float().to(self._device)
                x_min_2 = torch.from_numpy(X_minority[index2][np.newaxis, :]).float().to(self._device)
                step = self._random_state.uniform()
                new_series = self.pipeline.transform(mode=self.mode, x_min1=x_min_1, x_min2=x_min_2, alpha=step)
                new_series = new_series.cpu().numpy()
                assert new_series.shape == (1, C, L), f"VAE output shape {new_series.shape} != {(1, C, L)}"
                new_ts[i] = new_series.squeeze(0)
            if self.visualize:
                _plot_series_list(new_ts[:10], title="generated from latent smote")
        elif self.mode == "pair":
            for i in range(self.n_generate_samples):
                index = self._random_state.choice(X_minority.shape[0])
                x_min = torch.from_numpy(X_minority[index][np.newaxis, :]).float().to(self._device)
                index = self._random_state.choice(X_majority.shape[0])
                x_maj = torch.from_numpy(X_majority[index][np.newaxis, :]).float().to(self._device)
                new_series = self.pipeline.transform(mode=self.mode, x_min=x_min, x_maj=x_maj, use_y=True)
                new_series = new_series.cpu().numpy()
                assert new_series.shape == (1, C, L), f"VAE output shape {new_series.shape} != {(1, C, L)}"
                new_ts[i] = new_series.squeeze(0)
            if self.visualize:
                _plot_series_list(new_ts[:10], title="generated from pair")

        elif self.mode == "lp":
            num1 = self.n_generate_samples // 2
            num2 = self.n_generate_samples - num1
            for i in range(num1):
                index = self._random_state.choice(X_minority.shape[0])
                x_min = torch.from_numpy(X_minority[index][np.newaxis, :]).float().to(self._device)
                index = self._random_state.choice(X_majority.shape[0])
                x_maj = torch.from_numpy(X_majority[index][np.newaxis, :]).float().to(self._device)
                new_series = self.pipeline.transform(mode='pair', x_min=x_min, x_maj=x_maj, use_y=True)
                new_series = new_series.cpu().numpy()
                assert new_series.shape == (1, C, L), f"VAE output shape {new_series.shape} != {(1, C, L)}"
                new_ts[i] = new_series.squeeze(0)
            if self.visualize:
                _plot_series_list(new_ts[:10], title="generated from pair")

            for i in range(num2):
                index1, index2 = self._random_state.choice(X_minority.shape[0], size=2, replace=False)
                x_min_1 = torch.from_numpy(X_minority[index1][np.newaxis, :]).float().to(self._device)
                x_min_2 = torch.from_numpy(X_minority[index2][np.newaxis, :]).float().to(self._device)
                step = self._random_state.uniform()
                new_series = self.pipeline.transform(mode='latent', x_min1=x_min_1, x_min2=x_min_2, alpha=step)

                new_series = new_series.cpu().numpy()
                assert new_series.shape == (1, C, L), f"VAE output shape {new_series.shape} != {(1, C, L)}"
                new_ts[i + num1] = new_series.squeeze(0)
            if self.visualize:
                _plot_series_list(new_ts[num1:num1 + 10], title="generated from latent smote")


        self._generated_samples = new_ts.copy()
        X_resampled.append(new_ts)
        y_resampled.append(new_ys)
        X_synthetic = np.vstack(X_resampled)
        y_synthetic = np.hstack(y_resampled)

        return X_synthetic, y_synthetic


if __name__ == "__main__":
    dataset_name = 'AllGestureWiimoteX_eq'
    smote = VOTE(mode='latent', random_state=0, visualize=False, dataset_name=dataset_name)
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    print(np.unique(y_train, return_counts=True))
    arr = X_test
    # 检查是否有 NaN
    print(np.isnan(arr).any())  # True

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))

    # from aeon.classification.deep_learning import MLPClassifier
    # hc2 = MLPClassifier()
    # from aeon.classification.hybrid import HIVECOTEV2
    #
    # hc2 = HIVECOTEV2()
    # hc2.fit(X_resampled, y_resampled)
    # y_pred = hc2.predict(X_test)
    # acc = np.mean(y_pred == y_test)
    # print(acc)
    # # stop = ""
    # # n_samples = 100  # Total number of labels
    # # majority_num = 90  # number of majority class
    # # minority_num = n_samples - majority_num  # number of minority class
    # # np.random.seed(42)
    # #
    # # X = np.random.rand(n_samples, 1, 10)
    # # y = np.array([0] * majority_num + [1] * minority_num)
    # # print(np.unique(y, return_counts=True))
    # #
    # # X_resampled, y_resampled = smote.fit_transform(X, y)
    # # print(X_resampled.shape)
    # # print(np.unique(y_resampled, return_counts=True))
