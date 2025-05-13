from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state

from sklearn.neighbors import NearestNeighbors
from tsml_eval._wip.rt.transformations.collection.imbalance._smote import SMOTE
from aeon.transformations.collection import BaseCollectionTransformer

class SMOTENaNDE(SMOTE):
    """
    SMOTE oversampling with Natural Neighbor Deletion and Differential Evolution optimization.

    This class extends the basic SMOTE oversampling by identifying suspicious
    synthetic samples via natural neighbor mismatches and optimizing their positions
    iteratively using differential evolution instead of deleting them.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors for synthetic sample generation.
    distance : str or callable, default="euclidean"
        Distance metric to use.
    distance_params : dict or None, default=None
        Additional parameters for the distance metric.
    weights : str or callable, default="uniform"
        Weighting function for neighbors.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int, RandomState instance or None, default=None
        Random seed or random number generator.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_neighbors=5,
        distance: Union[str, callable] = "euclidean",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs

        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        self.new_generated_samples_pair = None
        self.X_original_ = None
        self.y_original_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self.X_original_ = X
        self.y_original_ = y
        super()._fit(X, y)
        return self

    def _transform(self, X, y=None):
        """
        Oversample using SMOTE, then identify suspicious samples via natural neighbor
        mismatches, and optimize their positions using differential evolution.
        """
        # Step 1: Use SMOTE logic to oversample minority classes and generate synthetic samples
        X_resampled, y_resampled = super()._transform(X, y)

        # Step 2: Identify suspicious samples based on natural neighbor mismatches
        suspicious_indices, NANE = self._natural_neighbor_filtering(X_resampled, y_resampled)

        # Step 3: Optimize suspicious sample positions using differential evolution
        X_optimized = self._differential_evolution_optimize(X_resampled, y_resampled, suspicious_indices)

        # Step 4: Return recombined dataset (optimized + normal samples)
        return X_optimized, y_resampled

    def _natural_neighbor_filtering(self, X, y, max_r=20):
        """
        Implementation of parameter-free natural neighbor (NaN) filtering based on the SMOTE-NaN-DE paper.

        This function finds the Natural Neighbor Eigenvalue (NaNE) λ and identifies suspicious examples:
        if any natural neighbor has a different label, mark the sample as suspicious.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, seq_len)
            Time series input.
        y : ndarray of shape (n_samples,)
            Class labels.
        max_r : int
            Maximum number of nearest neighbors to search for forming NSS.

        Returns
        -------
        suspicious_indices : ndarray
            Indices of suspicious samples (those with NaNs from different class).
        NaNE : int
            The final natural neighbor eigenvalue λ.
        """
        n_samples, n_channels, seq_len = X.shape
        X_flat = X.reshape(n_samples, -1)

        for r in range(1, max_r + 1):
            # Find r-nearest neighbors
            nn = NearestNeighbors(n_neighbors=r + 1, algorithm='kd_tree').fit(X_flat)
            knn = nn.kneighbors(X_flat, return_distance=False)[:, 1:]  # Exclude self
            reverse_neighbors = [[] for _ in range(n_samples)]

            for i in range(n_samples):
                for neighbor in knn[i]:
                    reverse_neighbors[neighbor].append(i)

            # Find mutual neighbors: NaN(xi) = xj ∈ NN(xi) and xi ∈ NN(xj)
            natural_neighbors = [set() for _ in range(n_samples)]
            for i in range(n_samples):
                for j in knn[i]:
                    if i in knn[j]:
                        natural_neighbors[i].add(j)

            # Check if all points have at least one natural neighbor
            all_have_nan = all(len(nns) > 0 for nns in natural_neighbors)
            if all_have_nan:
                NaNE = r
                break
        else:
            NaNE = max_r

        # Identify suspicious samples
        suspicious_indices = []
        for i, neighbors in enumerate(natural_neighbors):
            for j in neighbors:
                if y[i] != y[j]:
                    suspicious_indices.append(i)
                    break

        return np.array(suspicious_indices), NaNE

    def _differential_evolution_optimize(self, X, y, suspicious_indices, max_iter=100):
        """
        Iteratively optimize suspicious sample positions using differential evolution.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, seq_len)
            The oversampled dataset.
        y : ndarray of shape (n_samples,)
            The corresponding labels.
        suspicious_indices : ndarray
            Indices of suspicious samples to optimize.
        max_iter : int
            Maximum number of iterations for DE.

        Returns
        -------
        X_optimized : ndarray
            The dataset with suspicious samples optimized.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.base import clone

        X = X.copy()
        n_samples, n_channels, seq_len = X.shape
        D = n_channels * seq_len
        flat_X = X.reshape(n_samples, -1)

        SE_idx = np.array(suspicious_indices)
        NE_idx = np.setdiff1d(np.arange(n_samples), SE_idx)
        X_NE, y_NE = flat_X[NE_idx], y[NE_idx]
        X_SE, y_SE = flat_X[SE_idx], y[SE_idx]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_NE, y_NE)

        for _ in range(max_iter):
            predictions = clf.predict(X_SE)
            misclassified = predictions != y_SE

            if not np.any(misclassified):
                break

            for i in np.where(misclassified)[0]:
                xi = X_SE[i]
                same_class = X_SE[y_SE == y_SE[i]]

                if len(same_class) < 3:
                    noise = np.random.uniform(-0.1, 0.1, size=D)
                    mutant = xi + noise
                else:
                    r1, r2, r3 = same_class[np.random.choice(len(same_class), 3, replace=False)]
                    Fi = np.random.choice([8, 20, 0.1 + np.random.rand() * 0.8])
                    mutant = r1 + Fi * (r2 - r3)

                K = np.random.rand()
                trial = xi + K * (mutant - xi)
                trial = np.clip(trial, xi.min(), xi.max())

                if clf.predict(trial.reshape(1, -1))[0] == y_SE[i]:
                    X_SE[i] = trial

        flat_X[SE_idx] = X_SE
        return flat_X.reshape(n_samples, n_channels, seq_len)


if __name__ == "__main__":
    X = np.random.randn(100, 3, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))
    smote = SMOTE(
            n_neighbors=5,
            distance="euclidean",
            distance_params=None,
            weights="uniform")

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""

    # === Multivariate SMOTE Verification ===
    print("\n=== Multivariate SMOTE alignment test with near-identical channels ===")
    base = np.random.randn(30, 50)
    X = np.stack([base, base + np.random.normal(0, 1e-5, size=base.shape)], axis=1)
    y = np.array([0] * 20 + [1] * 10)

    smote = SMOTENaNDE(n_neighbors=3, random_state=42, distance="euclidean")
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform(X, y)

    new_samples = X_resampled[len(X):]
    diffs = new_samples[:, 0, :] - new_samples[:, 1, :]
    std_dev = np.std(diffs, axis=1)

    print("Mean std deviation across channels (should be < 1e-4 if aligned):", np.mean(std_dev))