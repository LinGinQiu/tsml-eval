import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from typing import Optional, Union
from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import ESMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._fbsmote import FrequencyBinSMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._stlor import STLOversampler
from tsml_eval._wip.rt.transformations.collection.imbalance._utils import SyntheticSampleSelector

from collections import OrderedDict

class HybridWrapper(BaseCollectionTransformer):

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, random_state=None,
                 n_jobs: int = 1,
                 enable_selection: bool = True,
                 ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.enable_selection = enable_selection
        self._set_transformers()
        super().__init__()

    def _set_transformers(self):
        transformers = [
                "esmote",
                "fbsmote",
            ]
        self._transformers = []
        for transformer in transformers:
            if transformer == "esmote":
                esmote = ESMOTE(
                    n_neighbors=5,
                    distance="msm",
                    distance_params=None,
                    weights="uniform",
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
                self._transformers.append(esmote)
            elif transformer == "fbsmote":
                fbsmote = FrequencyBinSMOTE(
                    n_neighbors=3,
                    top_k=6,
                    freq_match_delta=2,
                    bandwidth=1,
                    random_state=self.random_state,
                    normalize_energy=True,
                    enable_selection=False,
                )
                self._transformers.append(fbsmote)
            elif transformer == "stl":
                stl = STLOversampler(
                    noise_scale=0.05,
                    block_bootstrap=True,
                    use_boxcox=True,
                    random_state=self.random_state,
                    period_estimation_method="acf"
                )
                self._transformers.append(stl)



    def _fit(self, X, y=None):

        self.fitted_transformers = []
        for i, transformer in enumerate(self._transformers):
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
            X_syn = np.array(list(X_syn))
            y_syn = np.array(list(y_syn))
            X_filtered, y_filtered = selector.select(X_real, y_real, X_syn, y_syn)
            X_synthetic = np.concatenate([X_real, X_filtered])
            y_synthetic = np.concatenate([y_real, y_filtered])
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
    from local.load_ts_data import X_train, y_train, X_test, y_test

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'train data count: {np.unique(y_train, return_counts=True)}')

    if has_duplicate_samples(X_train):
        print("Warning: Duplicate samples detected in X!")
    else:
        print("No duplicate samples in X.")
    wrapper = HybridWrapper(
            random_state=1,
            enable_selection=True,
            n_jobs=1,
        )
    wrapper.fit(X_train, y_train)
    X_resampled, y_resampled = wrapper.transform(X_train, y_train)
    if has_duplicate_samples(X_resampled):
        print("Warning: Duplicate samples detected in X_resampled!")
    else:
        print("No duplicate samples in X_resampled.")
    print("Resampled shape:", X_resampled.shape)
    print("Resampled y shape:", y_resampled.shape)
    print("Unique classes in resampled y:", np.unique(y_resampled, return_counts=True))

