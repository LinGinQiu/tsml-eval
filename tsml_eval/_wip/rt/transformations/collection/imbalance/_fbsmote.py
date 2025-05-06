import numpy as np
import warnings
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer
from collections import OrderedDict
from tsml_eval._wip.rt.transformations.collection.imbalance._utils import SyntheticSampleSelector

class FrequencyBinSMOTE(BaseCollectionTransformer):
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
    normalize_energy : bool, default=False
        Whether to normalize the energy of the synthetic sample to match the original
        sample after frequency domain interpolation. This helps maintain consistent
        signal power and avoid artifacts.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, n_neighbors=3, top_k=3, freq_match_delta=2, bandwidth=1, apply_window=False, random_state=None, normalize_energy=False, enable_selection=False):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.top_k = top_k
        self.freq_match_delta = freq_match_delta
        self.bandwidth = bandwidth
        self.apply_window = apply_window
        self.normalize_energy = normalize_energy
        self.enable_selection = enable_selection
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
            if self.enable_selection:
                n_samples_gen = n_samples_gen * 2
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
                random_choice_list = self._random_state.choice(len(freq_curr), size=(self.top_k//2), replace=False)
                freq_curr = freq_curr[random_choice_list]
                mag_curr = mag_curr[random_choice_list]
                # Compute FFT of the current sample
                F_curr = np.fft.rfft(x_curr)
                incremental_idx = []
                incremental_value = []
                # Initialize synthetic FFT as a copy of current sample's FFT
                F_synthetic = F_curr.copy()

                fallback_count = 0
                scores = np.zeros(len(X_class), dtype=int)
                for p in range(len(freq_curr)):
                    target_freq = freq_curr[p]
                    # relaxed frequency matching within delta
                    score = np.any(np.abs(freq_class - target_freq) <= self.freq_match_delta, axis=1).astype(int)
                    scores += score
                topk = np.argsort(-scores)[:self.n_neighbors]
                for p in range(self.n_neighbors):
                    # Compute FFT of the neighbor sample
                    nn_idx = topk[p]
                    x_nn = X_class[nn_idx]
                    F_nn = np.fft.rfft(x_nn)
                    for f in range(len(freq_curr)):
                        target_freq = freq_curr[f]
                        mag_curr_p = mag_curr[f]

                        # Interpolation weight alpha sampled around ratio of magnitudes
                        mag_nn_p = np.abs(F_nn)[target_freq]
                        base_alpha = mag_nn_p / (mag_curr_p + mag_nn_p + 1e-8)
                        # Restrict alpha to [0.2, 0.8] to avoid extreme interpolation weights
                        # which can cause unrealistic synthetic samples or artifacts.
                        alpha = np.clip(self._random_state.normal(loc=base_alpha, scale=0.1), 0.2, 0.8)

                        # Interpolate in frequency domain over a bandwidth window around the target frequency
                        # Apply decay factor based on distance from target frequency bin
                        for idx_bin in range(target_freq - self.bandwidth, target_freq + self.bandwidth + 1):
                            if 0 <= idx_bin < F_curr.shape[0]:
                                decay = 1.0 / (1.0 + abs(idx_bin - target_freq))
                                incremental_idx.append(idx_bin)
                                incremental_value.append(decay * alpha * (F_nn[idx_bin] - F_curr[idx_bin]))

                for idx, value in zip(incremental_idx, incremental_value):
                        F_synthetic[idx] = F_synthetic[idx] + value
                # Apply optional smoothing window in frequency domain to reduce artifacts
                if self.apply_window:
                    F_synthetic *= np.hanning(F_synthetic.shape[0])

                # Normalize energy of synthetic sample to match original sample energy
                if self.normalize_energy:
                    energy_curr = np.sum(np.abs(F_curr) ** 2)
                    energy_synth = np.sum(np.abs(F_synthetic) ** 2)
                    scaling = np.sqrt(energy_curr / (energy_synth + 1e-8))
                    F_synthetic *= scaling

                # Inverse FFT to reconstruct synthetic time series
                x_synthetic = np.fft.irfft(F_synthetic, n=seq_len)

                X_gen[i] = x_synthetic
                i += 1

            X_resampled.append(X_gen)
            y_resampled.append(y_gen)

        X_resampled = np.concatenate(X_resampled, axis=0)
        y_resampled = np.concatenate(y_resampled, axis=0)

        # Optional: apply selection mechanism to filter synthetic samples
        if self.enable_selection:
            from warnings import warn
            try:
                selector = SyntheticSampleSelector(random_state=self.random_state)
                X_real = X_resampled[:len(X)]
                y_real = y_resampled[:len(y)]
                assert np.array_equal(X_real, X)
                assert np.array_equal(y_real, y)
                X_syn = X_resampled[len(X):]
                y_syn = y_resampled[len(y):]
                X_filtered, y_filtered = selector.select(X_real, y_real, X_syn, y_syn)
                X_resampled = np.concatenate([X_real, X_filtered])
                y_resampled = np.concatenate([y_real, y_filtered])
            except Exception as e:
                warn(f"Synthetic selection failed: {e}")

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

    # Initialize FrequencyBinSMOTE
    smote = FrequencyBinSMOTE(n_neighbors=3, top_k=10, freq_match_delta=2,
                              random_state=42,apply_window=True,normalize_energy=True,
                              enable_selection=True)

    # Fit and transform
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)

    print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}")
    print(f"Class distribution after oversampling: {np.unique(y_resampled, return_counts=True)}")
