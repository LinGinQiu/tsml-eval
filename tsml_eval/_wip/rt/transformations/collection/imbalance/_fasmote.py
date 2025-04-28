import numpy as np
import warnings
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer
from collections import OrderedDict
from tsml_eval._wip.rt.clustering.averaging._ba_utils import _get_alignment_path
class FrequencyAwareSMOTE(BaseCollectionTransformer):
    """
    Frequency-aware SMOTE oversampling algorithm.

    This variant selects neighbors based on dominant frequency similarity
    with a tolerance delta. Only supports univariate (single-channel) time
    series currently.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for synthetic sample generation.
    top_k : int, default=3
        Number of top dominant frequencies to consider per sample.
    freq_match_delta : int, default=2
        Maximum allowed frequency index difference to consider two samples
        as frequency neighbors.
    random_state : int, RandomState instance or None, default=None
        Random number generator seed control.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, n_neighbors=5, top_k=3, freq_match_delta=2, random_state=None):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.top_k = top_k
        self.freq_match_delta = freq_match_delta
        self._random_state = None
        super().__init__()

    def _fit(self, X, y=None):
        """Store y and build sampling strategy."""
        self._random_state = check_random_state(self.random_state)
        if X.ndim == 3:
            X = X[:, 0, :]  # flatten if (n, 1, l)

        self.y_ = y.copy()

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
        """
        Generate synthetic samples based on frequency-aware neighbor selection.

        Samples are matched if their dominant frequencies are within a tolerance
        delta (`freq_match_delta`).
        """
        if X.ndim == 3:
            X = X[:, 0, :]  # flatten if (n, 1, l)

        freq_features, freq_magnitudes = self._compute_topk_frequencies(X)

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        n_samples, seq_len = X.shape

        for class_sample, n_samples_gen in self.sampling_strategy_.items():
            if n_samples_gen == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]
            freq_class = freq_features[target_class_indices]
            mag_class = freq_magnitudes[target_class_indices]

            X_gen = np.zeros((n_samples_gen, seq_len), dtype=X.dtype)
            y_gen = np.full(n_samples_gen, fill_value=class_sample, dtype=y.dtype)

            i = 0
            max_attempts = n_samples_gen * 10
            attempts = 0

            while i < n_samples_gen and attempts < max_attempts:
                attempts += 1
                idx = self._random_state.randint(0, len(X_class))
                x_curr = X_class[idx]
                freq_curr = freq_class[idx]
                mag_curr = mag_class[idx]

                generated_components = []
                weights = mag_curr / np.sum(mag_curr)

                fallback_count = 0

                for p in range(self.top_k):
                    target_freq = freq_curr[p]
                    mag_curr_p = mag_curr[p]
                    # relaxed frequency matching within delta
                    mask = np.any(np.abs(freq_class - target_freq) <= self.freq_match_delta, axis=1)
                    candidates = np.where(mask)[0]
                    candidates = candidates[candidates != idx]

                    if len(candidates) == 0:
                        component = x_curr.copy()
                        fallback_count += 1
                    else:
                        nn_idx = self._random_state.choice(candidates)
                        mag_nn_p = mag_class[nn_idx, p]

                        x_nn = X_class[nn_idx]
                        base_alpha = mag_nn_p / (mag_curr_p + mag_nn_p + 1e-8)
                        alpha = np.clip(self._random_state.normal(loc=base_alpha, scale=0.1), 0, 1)


                        c = self._random_state.uniform(0.5, 2.0)  # Randomize MSM penalty parameter
                        alignment, _ = _get_alignment_path(
                            x_nn,
                            x_curr,
                            distance='msm',
                            c=c
                        )
                        path_list = [[] for _ in range(x_curr.shape[-1])]
                        for k, l in alignment:
                            path_list[k].append(l)

                        empty_of_array = np.zeros_like(x_curr, dtype=type(x_curr[0]))  # shape: (l)

                        for k, l in enumerate(path_list):
                            if len(l) == 0:
                                raise ValueError("No alignment found")
                            key = self._random_state.choice(l)
                            empty_of_array[k] = x_curr[k] - x_nn[key]

                        component = x_curr + alpha * empty_of_array

                    generated_components.append(weights[p] * component)

                if fallback_count == self.top_k:
                    continue  # all top-k fallback, retry

                generated_components = np.stack(generated_components, axis=0)
                x_synthetic = np.sum(generated_components, axis=0)
                X_gen[i] = x_synthetic
                i += 1

            if i < n_samples_gen:
                warnings.warn(
                    f"Only generated {i}/{n_samples_gen} samples for class {class_sample} "
                    f"due to insufficient frequency neighbors.",
                    UserWarning
                )
                X_gen = X_gen[:i]
                y_gen = y_gen[:i]

            X_resampled.append(X_gen)
            y_resampled.append(y_gen)

        X_resampled = np.concatenate(X_resampled, axis=0)
        y_resampled = np.concatenate(y_resampled, axis=0)

        return X_resampled[:, np.newaxis, :], y_resampled

    def _compute_topk_frequencies(self, X):
        """Compute top-k frequency indices and magnitudes per sample."""
        n_samples, seq_len = X.shape
        freq_features = []
        freq_magnitudes = []

        for x in X:
            fft_vals = np.fft.rfft(x)
            mag = np.abs(fft_vals)
            mag[0] = 0  # ignore DC
            topk_idx = np.argpartition(mag, -self.top_k)[-self.top_k:]
            topk_mag = mag[topk_idx]
            freq_features.append(topk_idx)
            freq_magnitudes.append(topk_mag)

        return np.array(freq_features), np.array(freq_magnitudes)


if __name__ == "__main__":
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))

    # Initialize FrequencyAwareSMOTE
    smote = FrequencyAwareSMOTE(n_neighbors=5, top_k=3, freq_match_delta=2, random_state=42)

    # Fit and transform
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)

    print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}")
    print(f"Class distribution after oversampling: {np.unique(y_resampled, return_counts=True)}")
