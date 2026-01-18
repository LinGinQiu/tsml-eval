"""SMOTE over sampling algorithm.

See more in imblearn.over_sampling.SMOTE
original authors:
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT
"""

from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer

__all__ = ["ShapeSMOTE"]


class ShapeSMOTE(BaseCollectionTransformer):
    """
    Over-sampling
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._random_state = None
        super().__init__()

    def _fit(self, X, y=None):
        # set the additional_neihbor required by SMOTE
        self._random_state = check_random_state(self.random_state)
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
        """SMOTE oversampling. Supports (n, c, l) or (n, l) shaped input."""
        # if input is 3D (n, c, l) or 2D (n, l), proceed; otherwise fall back to original
        if X.ndim == 3:
            n_samples, n_channels, seq_len = X.shape
        elif X.ndim == 2:
            X = X[:, np.newaxis, :]
            n_samples, n_channels, seq_len = X.shape
        else:
            raise ValueError("Input X must be 3D or 2D: (n_samples, (n_channels), seq_len)")

        X_resampled_channels = []
        y_resampled_final = None

        for c in range(n_channels):
            X_c = X[:, c, :]  # shape: (n, seq_len)
            X_resampled = [X_c.copy()]
            y_resampled = [y.copy()]

            for class_sample, n_samples_gen in self.sampling_strategy_.items():
                if n_samples_gen == 0:
                    continue
                target_class_indices = np.flatnonzero(y == class_sample)
                X_class = X_c[target_class_indices]
                y_class = y[target_class_indices]

                X_new, y_new = self._make_samples(
                    X_class, y.dtype, class_sample,n_samples)
                X_resampled.append(X_new)
                y_resampled.append(y_new)

            X_c_final = np.vstack(X_resampled)  # shape (n_new, seq_len)
            y_c_final = np.hstack(y_resampled)
            X_resampled_channels.append(X_c_final)

            if y_resampled_final is None:
                y_resampled_final = y_c_final  # all channels use same y

        # reshape back to (n_new, c, l)
        X_stacked = np.stack(X_resampled_channels, axis=1)
        self.new_generated_samples_pair = None  # reset for next call
        return X_stacked, y_resampled_final

    def _make_samples(
        self, X, y_dtype, y_type, n_samples):
        """Make artificial samples constructed based on nearest neighbours.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_cases, n_timepoints), time series from which the new series will
            be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.
        """
        samples_indices = self._random_state.randint(
            low=0, high=X.size, size=n_samples
        )
        # find the mean of the samples
        mean = np.mean(X, axis=0)

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(
        self, X, nn_data, nn_num, rows, cols, steps, y_type=None, y=None
    ):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        y_type : str, int or None, default=None
            Class label of the current target classes for which we want to generate
            samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        diffs = nn_data[nn_num[rows, cols]] - X[rows]
        if y is not None:  # only entering for BorderlineSMOTE-2
            mask_pair_samples = y[nn_num[rows, cols]] != y_type
            diffs[mask_pair_samples] *= self._random_state.uniform(
                low=0.0, high=0.5, size=(mask_pair_samples.sum(), 1)
            )
        X_new = X[rows] + steps * diffs
        return X_new.astype(X.dtype)

if __name__ == "__main__":
    X = np.random.randn(100, 3, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    smote = SMOTE()

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""

    # === Multivariate SMOTE Verification ===
    print("\n=== Multivariate SMOTE alignment test with near-identical channels ===")
    base = np.random.randn(30, 50)
    X = np.stack([base, base + np.random.normal(0, 1e-5, size=base.shape)], axis=1)
    y = np.array([0] * 20 + [1] * 10)

    smote = SMOTE(n_neighbors=3, random_state=42, distance="euclidean")
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)

    new_samples = X_resampled[len(X):]
    diffs = new_samples[:, 0, :] - new_samples[:, 1, :]
    std_dev = np.std(diffs, axis=1)

    print("Mean std deviation across channels (should be < 1e-4 if aligned):", np.mean(std_dev))
