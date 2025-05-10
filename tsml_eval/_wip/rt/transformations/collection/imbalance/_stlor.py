import numpy as np
from collections import OrderedDict
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer


class STLOversampler(BaseCollectionTransformer):
    """
    STL-based oversampling for imbalanced time series classification.

    For each minority class sample:
    - Optionally apply Box-Cox transformation.
    - Decompose with STL into trend, seasonal, remainder.
    - Perturb the residual (bootstrap or noise).
    - Reconstruct and generate synthetic samples.

    Parameters
    ----------
    noise_scale : float, default=0.05
        Standard deviation multiplier for Gaussian noise added to residuals.

    block_bootstrap : bool, default=True
        Whether to perform block-wise bootstrapping on residuals.

    use_boxcox : bool, default=True
        Whether to apply Box-Cox transformation before STL decomposition.

    random_state : int or RandomState, default=None
        Controls reproducibility.

    period_estimation_method : str, default="fixed"
        Method to estimate seasonality period. Options are "fixed", "acf", "fft".
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, noise_scale=0.05, block_bootstrap=True, use_boxcox=True, random_state=None, period_estimation_method="fixed"):
        self.noise_scale = noise_scale
        self.block_bootstrap = block_bootstrap
        self.use_boxcox = use_boxcox
        self.random_state = check_random_state(random_state)
        self.period_estimation_method = period_estimation_method
        super().__init__()

    def _fit(self, X, y=None):
        unique, counts = np.unique(y, return_counts=True)
        n_sample_majority = np.max(counts)
        class_majority = unique[np.argmax(counts)]
        sampling_strategy = {
            cls: n_sample_majority - count
            for cls, count in zip(unique, counts)
            if cls != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        return self

    def _transform(self, X, y=None):
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        n_samples, n_channels, seq_len = X.shape

        X_new_channels = []
        y_new_final = None

        for c in range(n_channels):
            X_c = X[:, c, :]
            X_c_new = [X_c.copy()]
            y_c_new = [y.copy()]

            for cls, n_gen in self.sampling_strategy_.items():
                if n_gen == 0:
                    continue
                X_cls = X_c[y == cls]
                synthetic = self._generate_synthetic_samples(X_cls, n_gen, seq_len)
                X_c_new.append(synthetic)
                y_c_new.append(np.full(n_gen, cls, dtype=y.dtype))

            X_c_stacked = np.vstack(X_c_new)
            y_c_stacked = np.hstack(y_c_new)
            X_new_channels.append(X_c_stacked)
            if y_new_final is None:
                y_new_final = y_c_stacked

        X_final = np.stack(X_new_channels, axis=1)
        return X_final, y_new_final

    def _stl(self, ts, period, robust=True):
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError as e:
            raise ImportError(
                "STLOversampler requires 'statsmodels'. "
                "Install it via 'pip install statsmodels' or 'conda install -c conda-forge statsmodels'."
            ) from e
        return STL(ts, period=period, robust=robust)

    def _boxcox(self, x):
        from scipy.stats import boxcox
        x_shift = x - np.min(x) + 1e-6  # ensure strictly positive
        x_bc, lambda_ = boxcox(x_shift)
        shift = np.min(x) - 1e-6
        return x_bc, lambda_, shift

    def _inv_boxcox(self, x_bc, lambda_, shift):
        from scipy.special import inv_boxcox
        return inv_boxcox(x_bc, lambda_) + shift

    def _estimate_period(self, ts, seq_len):
        if self.period_estimation_method == "acf":
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(ts, nlags=min(40, seq_len // 2), fft=True)
            peaks = np.where((acf_vals[1:-1] > acf_vals[:-2]) & (acf_vals[1:-1] > acf_vals[2:]))[0] + 1
            return peaks[0] if len(peaks) > 0 else max(2, seq_len // 2)
        elif self.period_estimation_method == "fft":
            from scipy.signal import periodogram
            freqs, power = periodogram(ts)
            freqs, power = freqs[1:], power[1:]  # skip DC
            if len(freqs) == 0 or np.max(power) == 0:
                return max(2, seq_len // 2)
            dom_freq = freqs[np.argmax(power)]
            return int(round(1 / dom_freq)) if dom_freq > 0 else max(2, seq_len // 2)
        else:  # default fixed
            return max(2, seq_len // 2)

    def _generate_synthetic_samples(self, X_cls, n_gen, seq_len):
        synthetic = []
        m = len(X_cls)
        base = n_gen // m
        remainder = n_gen % m

        for i, ts in enumerate(X_cls):
            n_i = base + (1 if i < remainder else 0)
            for _ in range(n_i):
                if self.use_boxcox:
                    ts_bc, lambda_, shift = self._boxcox(ts)
                else:
                    ts_bc, lambda_, shift = ts, None, 0

                period = self._estimate_period(ts_bc, seq_len)
                stl = self._stl(ts_bc, period=period, robust=True)
                result = stl.fit()

                trend = result.trend
                seasonal = result.seasonal
                resid = result.resid

                if self.block_bootstrap:
                    block_size = max(4, seq_len // 10)
                    n_blocks = seq_len // block_size + 1
                    blocks = []
                    for _ in range(n_blocks):
                        start = self.random_state.randint(0, seq_len - block_size + 1)
                        blocks.append(resid[start : start + block_size])
                    resid_boot = np.concatenate(blocks)[:seq_len]
                else:
                    resid_boot = resid.copy()
                    self.random_state.shuffle(resid_boot)

                noise = self.random_state.normal(0, np.std(resid) * self.noise_scale, size=seq_len)
                synthetic_ts = trend + seasonal + resid_boot + noise

                if self.use_boxcox:
                    synthetic_ts = self._inv_boxcox(synthetic_ts, lambda_, shift)

                synthetic.append(synthetic_ts)

        return np.array(synthetic)


if __name__ == "__main__":
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))

    smote = STLOversampler(use_boxcox=True, noise_scale=0.05, block_bootstrap=True, period_estimation_method="acf")

    # Fit and transform
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)
    from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
    knn = KNeighborsTimeSeriesClassifier()
    knn.fit(X_resampled, y_resampled)
    print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}")
    print(f"Class distribution after oversampling: {np.unique(y_resampled, return_counts=True)}")