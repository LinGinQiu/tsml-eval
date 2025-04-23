from scipy.fft import rfft, irfft
import numpy as np
from tsml_eval._wip.rt.transformations.collection.imbalance._smote import SMOTE
from typing import Optional, Union

class FrequencySMOTE(SMOTE):
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

    See Also
    --------
    ADASYN

    References
    """

    _tags = {
        "capability:multivariate": False,
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
        # Squeeze to (n, length)
        X = np.squeeze(X, axis=1)
        freq = rfft(X, axis=1)

        if self.top_k is not None:
            # Keep only top_k frequencies per sample
            mag = np.abs(freq)
            idx = np.argsort(mag, axis=1)[:, :-self.top_k]
            for i in range(freq.shape[0]):
                freq[i, idx[i]] = 0

        # real + imag concat
        X_freq = np.concatenate([freq.real, freq.imag], axis=1)
        X_freq = X_freq[:, np.newaxis, :]  # shape: (n, 1, 2*f)

        # Call parent's fit_transform
        X_gen, y_gen = super()._transform(X_freq, y)

        # reconstruct complex freq
        n_freq = freq.shape[1]
        X_gen = np.squeeze(X_gen, axis=1)
        freq_gen = X_gen[:, :n_freq] + 1j * X_gen[:, n_freq:]

        # irfft to time domain
        X_time = irfft(freq_gen, n=X.shape[1], axis=1)
        X_time = X_time[:, np.newaxis, :]
        return X_time, y_gen


if __name__ == "__main__":
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    smote = FrequencySMOTE(top_k=None,
            n_neighbors=5,
            distance="euclidean",
            distance_params=None,
            weights="uniform")

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""