import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from collections import Counter
from aeon.transformations.collection import BaseCollectionTransformer


class CDSMOTE(BaseCollectionTransformer):
    """
    Class Decomposition + SMOTE oversampling for imbalanced classification.

    This transformer first applies k-means clustering on the majority class
    to decompose it into subclasses (class decomposition). Then, if the minority
    class size is below the average size of majority subclasses, it applies SMOTE
    to generate synthetic minority samples.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to split the majority class into.
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(self, n_clusters=2, k_neighbors=5, random_state=42):
        super().__init__()
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _fit(self, X, y=None):
        """Nothing to fit, since CD+SMOTE works directly on data."""
        return self

    def _transform(self, X, y=None):
        """
        Apply class decomposition on the majority class and SMOTE oversampling
        on the minority class if needed.

        Parameters
        ----------
        X : np.ndarray, shape (n_instances, n_channels, series_length)
            Input time series data.
        y : np.ndarray, shape (n_instances,)
            Class labels.

        Returns
        -------
        X_res : np.ndarray
            Resampled feature matrix.
        y_res : np.ndarray
            Resampled labels.
        """
        # Flatten time series for clustering and SMOTE
        n, c, l = X.shape
        X_flat = X.reshape(n, c * l)
        # Identify majority and minority
        class_counts = Counter(y)
        maj_class = max(class_counts, key=class_counts.get)
        min_class = min(class_counts, key=class_counts.get)
        if maj_class == min_class:
            return X, y  # No imbalance

        X_majority = X_flat[y == maj_class]
        X_minority = X_flat[y == min_class]

        # Step 1: Class decomposition with k-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_majority)

        # Assign new subclass labels
        y_majority_decomposed = np.array(
            [f"{maj_class}_Clt{c}" for c in cluster_labels]
        )

        # Combine majority (decomposed) and minority
        X_new = np.vstack([X_majority, X_minority])
        y_new = np.concatenate([y_majority_decomposed, np.array([min_class] * len(X_minority))])

        # Step 2: Oversample minority if needed
        avg_majority_size = len(X_majority) / self.n_clusters
        min_cluster_size = min(Counter(y_new), key=Counter(y_new).get)
        if len(X_minority) < avg_majority_size and min_cluster_size > 1:
            if len(X_minority) < self.k_neighbors:
                self.k_neighbors = len(X_minority) - 1
            if min_cluster_size < self.k_neighbors:
                self.k_neighbors = min_cluster_size - 1
            smote = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_new, y_new)
        else:
            X_res, y_res = X_new, y_new

        # Reshape back to (n_instances, n_channels, series_length)
        X_res = X_res.reshape(-1, c, l)

        return X_res, y_res


if __name__ == "__main__":
    ## Example usage
    from local.load_ts_data import X_train, y_train, X_test, y_test

    print(np.unique(y_train, return_counts=True))
    smote = CDSMOTE(n_clusters=2, k_neighbors=5, random_state=1)

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)

    print(np.unique(y_resampled, return_counts=True))
    stop = ""
    n_samples = 100  # Total number of labels
    majority_num = 90  # number of majority class
    minority_num = n_samples - majority_num  # number of minority class
    np.random.seed(42)

    X = np.random.rand(n_samples, 1, 10)
    y = np.array([0] * majority_num + [1] * minority_num)
    print(np.unique(y, return_counts=True))
    smote = CDSMOTE(n_clusters=2, k_neighbors=5, random_state=1)

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
