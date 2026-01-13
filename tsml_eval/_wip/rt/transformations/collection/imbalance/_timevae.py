import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist, pdist, squareform
from matplotlib.path import Path
from collections import OrderedDict
from typing import Union, Optional
from aeon.transformations.collection import BaseCollectionTransformer


class TimeVAE(BaseCollectionTransformer):
    """

    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
            self,
            random_state=None,
    ):
        self.random_state = random_state

        self._random_state = None
        self.sampling_strategy_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Logic to determine how many samples to generate per class.
        """
        self._random_state = check_random_state(self.random_state)

        # Validate y
        if y is None:
            raise ValueError("y is required for HS-SMOTE")

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)

        # Dictionary: {class_label: n_samples_to_generate}
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))

        return self

    def _transform(self, X, y=None):
        """
        Performs the TimeVAE generation logic.
         [num_samples, window_len, feature_dim].
        """
        X_in = np.asarray(X)
        if X_in.ndim == 3:
            # TimeVAE expects (N, T, D) i.e., (Samples, Seq_Len, Channels)
            # We must transpose: (N, C, L) -> (N, L, C)
            X_transposed = np.transpose(X_in, (0, 2, 1))
        else:
            # If (N, L), treat as (N, L, 1)
            X_transposed = X_in[..., np.newaxis]

        n_samples, seq_len, feat_dim = X_transposed.shape

        X_new_list = [X_in]
        y_new_list = [y]

        # Iterate through each minority class that needs oversampling
        for class_label, n_samples_needed in self.sampling_strategy_.items():
            if n_samples_needed <= 0:
                continue

            print(f"Oversampling class {class_label}: generating {n_samples_needed} samples using TimeVAE...")

            # 1. Filter data for the specific minority class
            minority_indices = np.where(y == class_label)[0]
            X_minority = X_transposed[minority_indices]  # Shape: (N_min, T, D)
            # 2. Initialize and train TimeVAE model
            # transform to float 32
            X_minority = X_minority.astype(np.float32)
            from tsml_eval._wip.rt.transformations.collection.imbalance.TimeVAE.vae_pipeline import run_vae_pipeline
            generated_samples = run_vae_pipeline(X_minority, vae_type='timeVAE', n_samples=n_samples_needed)
            generated_samples_final = np.transpose(generated_samples, (0, 2, 1))

            # Append to lists
            X_new_list.append(generated_samples_final)
            y_new_list.append(np.full(n_samples_needed, class_label))

        # Concatenate all data
        X_resampled = np.vstack(X_new_list)
        y_resampled = np.hstack(y_new_list)

        return X_resampled, y_resampled

if __name__ == "__main__":
    global leng
    dataset_name = 'MedicalImages'
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    print(np.unique(y_train, return_counts=True))
    # _plot_series_list([X_majority[0][0][:leng], X_majority[1][0][:leng]], title="Majority class example")
    # _plot_series_list([X_majority[2][0][:leng], X_majority[3][0][:leng]], title="Majority class example")
    smote = TimeVAE(
        random_state=42,
            )

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""
