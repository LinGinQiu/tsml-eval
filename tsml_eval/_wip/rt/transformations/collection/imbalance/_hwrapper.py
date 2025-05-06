import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from typing import Optional, Union
from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import ESMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._fbsmote import FrequencyBinSMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._utils import SyntheticSampleSelector

from collections import OrderedDict

class HybridWrapper(BaseCollectionTransformer):

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, n_neighbors=5, top_k=3, freq_match_delta=2, bandwidth=1, apply_window=False, random_state=None,
                 normalize_energy=True,
                 enable_selection=False,
                 distance: Union[str, callable] = "msm",
                 distance_params: Optional[dict] = None,
                 weights: Union[str, callable] = "uniform",
                 n_jobs: int = 1,
                 ):
        self.n_neighbors = n_neighbors
        self.top_k = top_k
        self.freq_match_delta = freq_match_delta
        self.bandwidth = bandwidth
        self.apply_window = apply_window
        self.random_state = random_state
        self.enable_selection = enable_selection
        self.normalize_energy = normalize_energy
        self.distance = distance
        self.distance_params = distance_params or {}
        self.weights = weights
        self.n_jobs = n_jobs

        esmote = ESMOTE(
            n_neighbors=5,
            distance=self.distance,
            distance_params=self.distance_params,
            weights=self.weights,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        fbsmote = FrequencyBinSMOTE(
            n_neighbors=self.n_neighbors,
            top_k=self.top_k,
            freq_match_delta=self.freq_match_delta,
            bandwidth=self.bandwidth,
            apply_window=self.apply_window,
            random_state=self.random_state,
            normalize_energy=self.normalize_energy,
            enable_selection=False
        )
        self.transformers = [esmote, fbsmote]
        super().__init__()


    def _fit(self, X, y=None):

        self.fitted_transformers = []
        for i, transformer in enumerate(self.transformers):
            transformer.fit(X, y)
            self.fitted_transformers.append(transformer)
        return self

    def _transform(self, X, y=None):
        # Apply transform from each transformer, keep only synthetic samples, and concatenate results
        synthetic_X_parts = []
        synthetic_y_parts = []
        synthetic_X_parts.append(X.copy())
        synthetic_y_parts.append(y.copy())

        for transformer in self.fitted_transformers:
            X_resampled, y_resampled = transformer.transform(X, y)

            # Assume the original samples are at the beginning
            synthetic_X = X_resampled[len(X):]
            synthetic_y = y_resampled[len(y):]

            synthetic_X_parts.append(synthetic_X)
            synthetic_y_parts.append(synthetic_y)

        X_synthetic = np.concatenate(synthetic_X_parts, axis=0)
        y_synthetic = np.concatenate(synthetic_y_parts, axis=0)
        # Optional: apply selection mechanism to filter synthetic samples
        if self.enable_selection:
            from warnings import warn
            selector = SyntheticSampleSelector(random_state=self.random_state)
            X_real = X_synthetic[:len(X)]
            y_real = y_synthetic[:len(y)]
            assert np.array_equal(X_real, X)
            assert np.array_equal(y_real, y)
            X_syn = X_synthetic[len(X):]
            y_syn = y_synthetic[len(y):]
            X_filtered, y_filtered = selector.select(X_real, y_real, X_syn, y_syn)
            X_synthetic = np.concatenate([X_real, X_filtered])
            y_synthetic = np.concatenate([y_real, y_filtered])
            print("X_synthetic shape:", X_synthetic.shape)
            if X_synthetic.ndim == 2:
                X_synthetic = X_synthetic[:, np.newaxis, :]


        return X_synthetic, y_synthetic

if __name__ == "__main__":
    def has_duplicate_samples(X):
        # Flatten each sample to 1D for comparison
        flattened = X.reshape((X.shape[0], -1))
        # Use numpy's unique with axis=0
        _, idx = np.unique(flattened, axis=0, return_index=True)
        return len(idx) != len(X)

    from sklearn.utils import shuffle
    np.random.seed(42)

    # Create imbalanced dummy dataset
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    if has_duplicate_samples(X):
        print("Warning: Duplicate samples detected in X!")
    else:
        print("No duplicate samples in X.")
    wrapper = HybridWrapper(
            n_neighbors=3,
            top_k=6,
            freq_match_delta=2,
            bandwidth=1,
            apply_window=True,
            random_state=1,
            enable_selection=True,
            normalize_energy=True,
            distance="msm",
            distance_params=None,
            weights="uniform",
            n_jobs=1,
        )
    wrapper.fit(X, y)
    X_resampled, y_resampled = wrapper.transform(X, y)
    if has_duplicate_samples(X_resampled):
        print("Warning: Duplicate samples detected in X_resampled!")
    else:
        print("No duplicate samples in X_resampled.")
    print("Original shape:", X.shape)
    print("Resampled shape:", X_resampled.shape)
    print(np.unique(y, return_counts=True))
    print(np.unique(y_resampled, return_counts=True))

    # Checks
    assert X_resampled.shape[0] > X.shape[0]
    assert np.array_equal(X_resampled[:len(X)], X)
    assert np.array_equal(y_resampled[:len(y)], y)
    print("Test passed.")

