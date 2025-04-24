from scipy.fft import rfft, irfft
import numpy as np
from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import ESMOTE
from typing import Optional, Union

class FrequencyESMOTE(ESMOTE):
    """
    Frequency domain SMOTE.

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

    References
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }
    def __init__(
        self, top_k=None,
        n_neighbors=5,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            distance=distance,
            distance_params=distance_params,
            weights=weights,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.top_k = top_k

    def _transform(self, X, y=None):
        # Ensure input is 3D: (n, c, l)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # Convert (n, l) → (n, 1, l)
        elif X.ndim != 3:
            raise ValueError("Input X must be 2D or 3D (n, c, l)")

        n_samples, n_channels, seq_len = X.shape

        # --- Step 1: FFT along time dimension ---
        freq = rfft(X, axis=2)  # Shape: (n, c, f)

        # --- Step 2: Optional frequency selection (top-k masking) ---
        if self.top_k is not None:
            mag = np.abs(freq)  # Magnitude spectrum
            idx = np.argsort(mag, axis=2)[:, :, :-self.top_k]  # Indices to zero
            for i in range(n_samples):
                for c in range(n_channels):
                    freq[i, c, idx[i, c]] = 0

        # --- Step 3: Convert to real + imag split as new channels ---
        real_part = freq.real  # (n, c, f)
        imag_part = freq.imag  # (n, c, f)
        X_freq = np.concatenate([real_part, imag_part], axis=1)  # (n, 2c, f)

        # --- Step 4: Use SMOTE in frequency domain ---
        X_gen, y_gen = super()._transform(X_freq, y)  # (n_new, 2c, f)

        # --- Step 5: Reconstruct complex frequency from generated real/imag parts ---
        X_gen_real = X_gen[:, :n_channels, :]  # (n_new, c, f)
        X_gen_imag = X_gen[:, n_channels:, :]  # (n_new, c, f)
        freq_gen = X_gen_real + 1j * X_gen_imag  # complex freq: (n_new, c, f)

        # --- Step 6: Inverse FFT to reconstruct time series ---
        X_time = irfft(freq_gen, n=seq_len, axis=2)  # (n_new, c, l)

        return X_time, y_gen


if __name__ == "__main__":
    X = np.random.randn(100, 3, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    smote = FrequencyESMOTE(
            n_neighbors=5,
            distance="msm",)

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print("Resampled shape:", X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""

    # === Multivariate SMOTE Verification ===
    print("\n=== Multivariate SMOTE alignment test with near-identical channels ===")
    base = np.random.randn(30, 50)
    X = np.stack([base, base + np.random.normal(0, 1e-5, size=base.shape)], axis=1)
    y = np.array([0] * 20 + [1] * 10)

    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)

    new_samples = X_resampled[len(X):]
    diffs = new_samples[:, 0, :] - new_samples[:, 1, :]
    std_dev = np.std(diffs, axis=1)

    print("Mean std deviation across channels (should be < 1e-4 if aligned):", np.mean(std_dev))