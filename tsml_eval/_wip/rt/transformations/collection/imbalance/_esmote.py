from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from numba import prange
from sklearn.utils import check_random_state

from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
from tsml_eval._wip.rt.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["chrisholder"]
__all__ = ["ESMOTE"]

from tsml_eval._wip.rt.utils._threading import threaded


class ESMOTE(BaseCollectionTransformer):
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
        n_neighbors=5,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs

        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        self.nn_ = KNeighborsTimeSeriesClassifier(
            n_neighbors=self.n_neighbors + 1,
            distance=self.distance,
            distance_params=self._distance_params,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        # if input is 3D (n, c, l) or 2D (n, l), proceed; otherwise fall back to original
        if X.ndim == 3:
            n_samples, n_channels, seq_len = X.shape
        elif X.ndim == 2:
            X = X[:, np.newaxis, :]
            n_samples, n_channels, seq_len = X.shape
        else:
            raise ValueError("Input X must be 3D or 2D: (n_samples, (n_channels), seq_len)")
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            n_samples = n_samples  # increase the number of samples to process selection
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            self.nn_.fit(X_class, y_class)
            nns = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class,
                y.dtype,
                class_sample,
                X_class,
                nns,
                n_samples,
                1.0,
                n_jobs=self.n_jobs,
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)
        X_synthetic = np.vstack(X_resampled)
        y_synthetic = np.hstack(y_resampled)
        if False:
            from warnings import warn
            from tsml_eval._wip.rt.transformations.collection.imbalance._utils import SyntheticSampleSelector
            selector = SyntheticSampleSelector(random_state=self.random_state)
            X_real = X_synthetic[:len(X)]
            y_real = y_synthetic[:len(y)]
            assert np.array_equal(X_real, X)
            assert np.array_equal(y_real, y)
            X_syn = X_synthetic[len(X):]
            y_syn = y_synthetic[len(y):]
            X_syn = np.array(list(X_syn))
            y_syn = np.array(list(y_syn))
            X_filtered, y_filtered = selector.select(X_real, y_real, X_syn, y_syn)
            X_synthetic = np.concatenate([X_real, X_filtered])
            y_synthetic = np.concatenate([y_real, y_filtered])
            if X_synthetic.ndim == 2:
                X_synthetic = X_synthetic[:, np.newaxis, :]
        return X_synthetic, y_synthetic

    @threaded
    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, n_jobs=1
    ):
        samples_indices = self._random_state.randint(
            low=0, high=nn_num.size, size=n_samples
        )

        steps = step_size * self._random_state.uniform(-1, 1, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        distance = self._random_state.choice(['msm', 'dtw', 'adtw'])

        X_new = _generate_samples(
            X,
            nn_data,
            nn_num,
            rows,
            cols,
            steps,
            random_state=self._random_state,
            distance=distance,
            **self._distance_params,
        )
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new


# @njit(cache=True, fastmath=True, parallel=True)
def _generate_samples(
    X,
    nn_data,
    nn_num,
    rows,
    cols,
    steps,
    random_state,
    distance,
    weights: Optional[np.ndarray] = None,
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 15,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
):
    X_new = np.zeros((len(rows), *X.shape[1:]), dtype=X.dtype)

    for count in prange(len(rows)):
        i = rows[count]
        j = cols[count]
        curr_ts = X[i]  # shape: (c, l)
        nn_ts = nn_data[nn_num[i, j]] # shape: (c, l)
        new_ts = curr_ts.copy()

        c = random_state.uniform(0.5, 2.0)  # Randomize MSM penalty parameter
        alignment, _ = _get_alignment_path(
            nn_ts,
            curr_ts,
            distance,
            window,
            g,
            epsilon,
            nu,
            lmbda,
            independent,
            c,
            descriptor,
            reach,
            warp_penalty,
            transformation_precomputed,
            transformed_x,
            transformed_y,
        )
        path_list = [[] for _ in range(curr_ts.shape[1])]
        for k, l in alignment:
            path_list[k].append(l)

        # num_of_alignments = np.zeros_like(curr_ts, dtype=np.int32)
        empty_of_array = np.zeros_like(curr_ts, dtype=float) # shape: (c, l)

        for k, l in enumerate(path_list):
            if len(l) == 0:
                print("No alignment found for time step")
                return new_ts

            key = random_state.choice(l)
            # Compute difference for all channels at this time step
            empty_of_array[:, k] = curr_ts[:, k] - nn_ts[:, key]

        #  apply_local_smoothing to empty_of_array
        windowsize = int(np.ceil(curr_ts.shape[-1] * 0.1))  # 10% of the length
        empty_of_array = apply_local_smoothing(empty_of_array, window_size=windowsize, mode='nearest')
        # apply_smooth_decay to empty_of_array
        empty_of_array = apply_smooth_decay(empty_of_array)

        X_new[count]  = new_ts + steps[count] * empty_of_array  #/ num_of_alignments

    return X_new


def apply_smooth_decay(arr, decay_width_ratio=0.1, decay_steepness=4):
    """
    对输入数组的每个通道应用序列边缘的平滑衰减。

    该函数使用 Sigmoid 函数来创建在序列两端（开始和结束）
    值较低、在中间接近 1 的权重，然后将这些权重应用于数组。

    Args:
        arr (np.ndarray): 输入的 NumPy 数组。
                          形状可以是 (n_channels, n_timepoints) 或 (n_timepoints,)。
                          如果是 (n_timepoints,)，函数会按其定义进行处理。
        decay_width_ratio (float, optional): 控制衰减区域的宽度。
                                             表示衰减区域占序列总长度的比例。
                                             例如，0.1 表示在序列开始和结束的 10% 区域衰减。
                                             默认值是 0.1。
        decay_steepness (float, optional): 控制衰减曲线的陡峭程度。
                                          值越大，衰减越急剧。默认值是 4。

    Returns:
        np.ndarray: 应用平滑衰减后的新数组，形状与输入数组相同。
    """
    if arr.ndim == 1:
        # 如果是 1D 数组 (n_timepoints,)
        n_timepoints = arr.shape[0]
        is_1d = True
    elif arr.ndim == 2:
        # 如果是 2D 数组 (n_channels, n_timepoints)
        n_channels, n_timepoints = arr.shape
        is_1d = False
    else:
        raise ValueError("Input array must be 1D or 2D (n_channels, n_timepoints).")

    if n_timepoints == 0:
        return arr # 空数组直接返回

    # 定义衰减的“宽度”
    decay_width = n_timepoints * decay_width_ratio

    # 创建一个从 0 到 n_timepoints-1 的时间步索引数组
    indices = np.arange(n_timepoints)

    # Sigmoid 函数参数计算
    # k_start: 衰减开始的位置（值开始上升）
    # k_end: 衰减结束的位置（值达到稳定）
    k_start = decay_width
    k_end = n_timepoints - decay_width

    # 防止 decay_width 过大导致 k_start >= k_end
    if k_start >= k_end:
        # 如果序列太短，无法形成明显的中间区域，则权重全为0或1，这里取中间值
        # 简单处理：如果衰减区域覆盖了整个序列，就让权重为0
        # 或者可以考虑返回一个警告，并使用一个更小的固定衰减宽度
        print(f"Warning: Sequence length ({n_timepoints}) is too short for the given decay_width_ratio ({decay_width_ratio}). No smooth decay applied, returning zeros.")
        return np.zeros_like(arr)


    # 计算两端的 Sigmoid 权重
    # 除以 (decay_width / decay_steepness) 控制坡度
    sigmoid_weights_start = 1 / (1 + np.exp(-(indices - k_start) / (decay_width / decay_steepness)))
    sigmoid_weights_end = 1 / (1 + np.exp((indices - k_end) / (decay_width / decay_steepness)))

    # 结合两端权重，取最小值以确保两端都衰减
    weights = np.minimum(sigmoid_weights_start, sigmoid_weights_end)

    # 将权重应用到输入数组
    if is_1d:
        weighted_arr = arr * weights
    else:
        # 使用广播机制，weights 会自动应用于所有通道
        weighted_arr = arr * weights[np.newaxis, :] # 增加一个维度以匹配 (n_channels, n_timepoints)

    return weighted_arr


from scipy.ndimage import uniform_filter1d  # 导入用于平滑的函数


def apply_local_smoothing(arr, window_size=3, mode='nearest'):
    """
    对输入数组的每个通道应用局部平滑（例如，移动平均）。

    Args:
        arr (np.ndarray): 输入的 NumPy 数组。
                          期望形状是 (n_channels, n_timepoints) 或 (n_timepoints,)。
        window_size (int, optional): 平滑窗口的大小。
                                     窗口越大，平滑效果越强，但可能丢失更多细节。
                                     必须是正整数。默认值是 3。
        mode (str, optional): 控制在数组边界如何处理。
                              'nearest': 使用最近的数据点填充边界。
                              'reflect': 通过反射数据填充边界。
                              'wrap': 通过环绕数据填充边界。
                              'constant': 用常数0填充边界。
                              默认值是 'nearest'。

    Returns:
        np.ndarray: 应用局部平滑后的新数组，形状与输入数组相同。
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if arr.ndim == 1:
        # 如果是 1D 数组 (n_timepoints,)
        n_timepoints = arr.shape[0]
        is_1d = True
        smoothed_arr = np.zeros_like(arr, dtype=arr.dtype)
    elif arr.ndim == 2:
        # 如果是 2D 数组 (n_channels, n_timepoints)
        n_channels, n_timepoints = arr.shape
        is_1d = False
        smoothed_arr = np.zeros_like(arr, dtype=arr.dtype)
    else:
        raise ValueError("Input array must be 1D or 2D (n_channels, n_timepoints).")

    if n_timepoints <= 1:
        # 对于单点或空序列，无需平滑
        return arr

    if window_size >= n_timepoints:
        # 如果窗口大小大于或等于序列长度，则整个序列会变成平均值，
        # 简单地将其设置为整个序列的均值或者不进行平滑 (取决于你希望的行为)
        # 这里选择应用一个非常大的窗口，效果类似全序列平均
        print(f"Warning: window_size ({window_size}) is >= sequence_length ({n_timepoints}). "
              f"The entire sequence will be heavily smoothed.")

    if is_1d:
        smoothed_arr = uniform_filter1d(arr, size=window_size, mode=mode)
    else:
        # 对每个通道独立进行平滑
        for c in range(n_channels):
            smoothed_arr[c, :] = uniform_filter1d(arr[c, :], size=window_size, mode=mode)

    return smoothed_arr

if __name__ == "__main__":
    # Example usage
    from local.load_ts_data import X_train, y_train, X_test, y_test
    print(np.unique(y_train, return_counts=True))
    smote = ESMOTE(n_neighbors=5, random_state=1, distance="msm")

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled,return_counts=True))
    stop = ""