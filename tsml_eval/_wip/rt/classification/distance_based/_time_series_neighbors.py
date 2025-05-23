"""KNN time series classification.

A KNN classifier which supports time series distance measures.
The class can take callables or uses string references to utilise the numba based
distances in aeon.distances.
"""

import numbers
from typing import Optional

__maintainer__ = []
__all__ = ["KNeighborsTimeSeriesClassifier"]

from typing import Callable, Union

import numpy as np

from aeon.classification.base import BaseClassifier
from tsml_eval._wip.rt.distances import get_distance_function, pairwise_distance

WEIGHTS_SUPPORTED = ["uniform", "distance"]


class KNeighborsTimeSeriesClassifier(BaseClassifier):
    """
    K-Nearest Neighbour Time Series Classifier.

    A KNN classifier which supports time series distance measures.
    It determines distance function through string references to numba
    based distances in aeon.distances, and can also be used with callables.

    Parameters
    ----------
    n_neighbors : int, default = 1
        Set k for knn.
    weights : str or callable, default = 'uniform'
        Mechanism for weighting a vote one of: 'uniform', 'distance', or a callable
        function.
    distance : str or callable, default ='dtw'
        Distance measure between time series.
        Distance metric to compute similarity between time series. A list of valid
        strings for metrics can be found in the documentation for
        :func:`aeon.distances.get_distance_function` or through calling
        :func:`aeon.distances.get_distance_function_names`. If a
        callable is passed it must be
        a function that takes two 2d numpy arrays of shape ``(n_channels,
        n_timepoints)`` as input and returns a float.
    distance_params : dict, default = None
        Dictionary for metric parameters for the case that distance is a str.
    n_jobs : int, default = None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details. Parameter for compatibility purposes, still unimplemented.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
    >>> classifier.fit(X_train, y_train)
    KNeighborsTimeSeriesClassifier(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "distance",
    }

    def __init__(
        self,
        distance: Union[str, Callable] = "dtw",
        distance_params: Optional[dict] = None,
        n_neighbors: int = 1,
        weights: Union[str, Callable] = "uniform",
        n_jobs: int = 1,
    ) -> None:
        self.distance = distance
        self.distance_params = distance_params
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self._distance_params = distance_params or {}

        if weights not in WEIGHTS_SUPPORTED:
            raise ValueError(
                f"Unrecognised kNN weights: {weights}. "
                f"Allowed values are: {WEIGHTS_SUPPORTED}. "
            )
        self.weights = weights

        super().__init__()

    def _fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal,
        a list of 2D numpy arrays is passed, which may have different lengths.
        y : array-like, shape = (n_cases)
            The class labels.
        """
        self.metric_ = get_distance_function(method=self.distance)
        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def _predict_proba(self, X):
        """
        Return probability estimates for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
                If the series are all equal length, a numpy3D will be passed. If
                unequal, a list of 2D numpy arrays is passed, which may have
                different lengths.

        Returns
        -------
        p : array of shape = (n_cases, n_classes_)
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        preds = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            idx, weights = self._kneighbors(X[i])
            for id, w in zip(idx, weights):
                predicted_class = self.y_[id]
                preds[i, predicted_class] += w

            preds[i] = preds[i] / np.sum(preds[i])

        return preds

    def _predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
        If the series are all equal length, a numpy3D will be passed. If unequal, a list
        of 2D numpy arrays is passed, which may have different lengths.

        Returns
        -------
        y : array of shape (n_cases)
            Class labels for each data sample.
        """
        self._check_is_fitted()

        preds = np.empty(len(X), dtype=self.classes_.dtype)
        for i in range(len(X)):
            scores = np.zeros(len(self.classes_))
            idx, weights = self._kneighbors(X[i])
            for id, w in zip(idx, weights):
                predicted_class = self.y_[id]
                scores[predicted_class] += w

            preds[i] = self.classes_[np.argmax(scores)]

        return preds

    def _kneighbors(self, X):
        """
        Find the K-neighbors of a point.

        Returns indices and weights of each point.

        Parameters
        ----------
        X : np.ndarray
            A single time series instance if shape = (n_channels, n_timepoints)

        Returns
        -------
        ind : array
            Indices of the nearest points in the population matrix.
        ws : array
            Array representing the weights of each neighbor.
        """
        distances = np.array(
            [
                self.metric_(X, self.X_[j], **self._distance_params)
                for j in range(len(self.X_))
            ]
        )

        # Find indices of k nearest neighbors using partitioning:
        # [0..k-1], [k], [k+1..n-1]
        # They might not be ordered within themselves,
        # but it is not necessary and partitioning is
        # O(n) while sorting is O(nlogn)
        closest_idx = np.argpartition(distances, self.n_neighbors)
        closest_idx = closest_idx[: self.n_neighbors]

        if self.weights == "distance":
            ws = distances[closest_idx]
            # Using epsilon ~= 0 to avoid division by zero
            ws = 1 / (ws + np.finfo(float).eps)
        elif self.weights == "uniform":
            ws = np.repeat(1.0, self.n_neighbors)
        else:
            raise Exception(f"Invalid kNN weights: {self.weights}")

        return closest_idx, ws

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints) or list of
        shape [n_cases] of 2D arrays shape (n_channels,n_timepoints_i)
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the value
            passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distances to points, only present if
            return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        self._check_is_fitted()

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )

        query_is_train = X is None
        if query_is_train:
            X = self.X_
            n_neighbors += 1
        else:
            X = self._preprocess_collection(X, store_metadata=False)
            self._check_shape(X)

        # Compute pairwise distances between X and fit data
        distances = pairwise_distance(
            X,
            self.X_ if not query_is_train else None,
            method=self.distance,
            **self._distance_params,
        )

        sample_range = np.arange(distances.shape[0])[:, None]
        neigh_ind = np.argpartition(distances, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        neigh_ind = neigh_ind[
            sample_range, np.argsort(distances[sample_range, neigh_ind])
        ]

        if query_is_train:
            neigh_ind = neigh_ind[:, 1:]

        if return_distance:
            if query_is_train:
                neigh_dist = distances[sample_range, neigh_ind]
                return neigh_dist, neigh_ind
            return distances[sample_range, neigh_ind], neigh_ind

        return neigh_ind

    def _fit_setup(self, X, y):
        # KNN can support if all labels are the same so always return False for single
        # class problem so the fit will always run
        X, y, _ = super()._fit_setup(X, y)
        return X, y, False

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> Union[dict, list[dict]]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        # non-default distance and algorithm
        params1 = {"distance": "euclidean"}

        return [params1]
