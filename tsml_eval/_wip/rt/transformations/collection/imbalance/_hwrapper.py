import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from typing import Optional, Union
from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import ESMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._fbsmote import FrequencyBinSMOTE


class HybridWrapper(BaseCollectionTransformer):

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, n_neighbors=5, top_k=3, freq_match_delta=2, bandwidth=1, apply_window=False, random_state=None, normalize_energy=False,
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
        self.normalize_energy = normalize_energy
        self.distance = distance
        self.distance_params = distance_params or {}
        self.weights = weights
        self.n_jobs = n_jobs

        esmote = ESMOTE(
            n_neighbors=self.n_neighbors,
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
        )
        self.transformers = [esmote, fbsmote]
        super().__init__()


    def _fit(self, X, y=None):
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        class_majority = max(target_stats, key=target_stats.get)
        majority_indices = np.flatnonzero(y == class_majority)
        minority_indices = np.flatnonzero(y != class_majority)

        # Split majority indices into equal parts
        n_transformers = len(self.transformers)
        majority_split = np.array_split(majority_indices, n_transformers)

        self.fitted_transformers = []
        self.datasets = []
        for i, transformer in enumerate(self.transformers):
            # Create the subset: one part of majority + all minority
            part_indices = np.concatenate([majority_split[i], minority_indices])
            X_sub = X[part_indices]
            y_sub = y[part_indices]
            transformer.fit(X_sub, y_sub)
            self.fitted_transformers.append(transformer)
            self.datasets.append([X_sub, y_sub])
        return self

    def _transform(self, X, y=None):
        # Apply transform from each transformer, keep only synthetic samples, and concatenate results
        synthetic_X_parts = []
        synthetic_y_parts = []
        synthetic_X_parts.append(X.copy())
        synthetic_y_parts.append(y.copy())

        for transformer, (X_sub, y_sub) in zip(self.fitted_transformers,self.datasets):
            X_resampled, y_resampled = transformer.transform(X_sub, y_sub)
            # Assume the original samples are at the beginning
            synthetic_X = X_resampled[len(X_sub):]
            synthetic_y = y_resampled[len(y_sub):]
            synthetic_X_parts.append(synthetic_X)
            synthetic_y_parts.append(synthetic_y)

        X_synthetic = np.concatenate(synthetic_X_parts, axis=0)
        y_synthetic = np.concatenate(synthetic_y_parts, axis=0)
        return X_synthetic, y_synthetic

if __name__ == "__main__":

    from sklearn.utils import shuffle
    np.random.seed(42)

    # Create imbalanced dummy dataset
    X_majority = np.random.rand(68, 1, 100)
    y_majority = np.zeros(68, dtype=int)
    X_minority = np.random.rand(32, 1, 100)
    y_minority = np.ones(32, dtype=int)
    X = np.concatenate([X_majority, X_minority], axis=0)
    y = np.concatenate([y_majority, y_minority], axis=0)
    X, y = shuffle(X, y, random_state=42)

    wrapper = HybridWrapper()
    wrapper.fit(X, y)
    X_resampled, y_resampled = wrapper.transform(X, y)

    print("Original shape:", X.shape)
    print("Resampled shape:", X_resampled.shape)
    print("Original distribution:", dict(zip(*np.unique(y, return_counts=True))))
    print("Resampled distribution:", dict(zip(*np.unique(y_resampled, return_counts=True))))

    # Checks
    assert X_resampled.shape[0] > X.shape[0]
    assert np.array_equal(X_resampled[:len(X)], X)
    assert np.array_equal(y_resampled[:len(y)], y)
    print("Test passed.")

