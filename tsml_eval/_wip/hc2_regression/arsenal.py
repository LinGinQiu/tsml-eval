"""Arsenal classifier.

kernel based ensemble of ROCKET classifiers.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["Arsenal"]

import time

import numpy as np
from aeon.base._base import _clone_estimator
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


class Arsenal(BaseRegressor):
    """Arsenal ensemble.

    Overview: an ensemble of ROCKET transformers using RidgeClassifierCV base
    classifier. Weights each classifier using the accuracy from the ridge
    cross-validation. Allows for generation of probability estimates at the
    expense of scalability compared to RocketClassifier.

    Parameters
    ----------
    num_kernels : int, default=2,000
        Number of kernels for each ROCKET transform.
    n_estimators : int, default=25
        Number of estimators to build for the ensemble.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket","minirocket","multirocket"]
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=100
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    weights_ : list of shape (n_estimators) of float
        Weight of each estimator in the ensemble.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances,total_intervals * att_subsample_size)
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    RocketClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/kernel_based/Arsenal.java>`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "classifier_type": "kernel",
    }

    def __init__(
        self,
        num_kernels=2000,
        n_estimators=25,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=100,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_estimators = n_estimators
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.estimators_ = []
        self.weights_ = []
        self.transformed_data_ = []

        self._weight_sum = 0
        self._label_average = 0

        super().__init__()

    def _fit(self, X, y):
        """Fit Arsenal to training data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        self._label_average = np.mean(y)

        if self.rocket_transform == "rocket":
            base_rocket = Rocket(num_kernels=self.num_kernels)
        elif self.rocket_transform == "minirocket":
            if self.n_dims_ > 1:
                base_rocket = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                )
            else:
                base_rocket = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                )
        elif self.rocket_transform == "multirocket":
            if self.n_dims_ > 1:
                base_rocket = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                )
            else:
                base_rocket = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        if time_limit > 0:
            self.n_estimators = 0
            self.estimators_ = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self.n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._threads_to_use)(
                    delayed(self._fit_estimator)(
                        _clone_estimator(
                            base_rocket,
                            None
                            if self.random_state is None
                            else (255 if self.random_state == 0 else self.random_state)
                            * 37
                            * (i + 1),
                        ),
                        X,
                        y,
                    )
                    for i in range(self._threads_to_use)
                )

                estimators, transformed_data = zip(*fit)

                self.estimators_ += estimators
                self.transformed_data_ += transformed_data

                self.n_estimators += self._threads_to_use
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._threads_to_use)(
                delayed(self._fit_estimator)(
                    _clone_estimator(
                        base_rocket,
                        None
                        if self.random_state is None
                        else (255 if self.random_state == 0 else self.random_state)
                        * 37
                        * (i + 1),
                    ),
                    X,
                    y,
                )
                for i in range(self.n_estimators)
            )

            self.estimators_, self.transformed_data_ = zip(*fit)

        self.weights_ = []
        self._weight_sum = 0
        for rocket_pipeline in self.estimators_:
            weight = rocket_pipeline.steps[2][1].best_score_
            self.weights_.append(weight)
            self._weight_sum += weight

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        y_preds = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_for_estimator)(
                X,
                self.estimators_[i],
                i,
            )
            for i in range(self.n_estimators)
        )

        return np.around(np.sum(y_preds, axis=0) / self._weight_sum, 8)

    def _get_train_preds(self, X, y) -> np.ndarray:
        self.check_is_fitted()
        # X, y = check_X_y(X, y, coerce_to_numpy=True)

        n_instances, n_dims, series_length = X.shape

        if (
            n_instances != self.n_instances_
            or n_dims != self.n_dims_
            or series_length != self.series_length_
        ):
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        p = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._train_preds_for_estimator)(
                y,
                i,
            )
            for i in range(self.n_estimators)
        )
        y_probas, weights, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(n_instances)
        for n, oob in enumerate(oobs):
            for inst in oob:
                divisors[inst] += weights[n]

        for i in range(n_instances):
            results[i] = (
                self._label_average if divisors[i] == 0 else results[i] / divisors[i]
            )

        return results

    def _fit_estimator(self, rocket, X, y):
        transformed_x = rocket.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(transformed_x, y)
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        ridge.fit(scaler.transform(transformed_x), y)
        return [
            make_pipeline(rocket, scaler, ridge),
            transformed_x if self.save_transformed_data else None,
        ]

    def _predict_for_estimator(self, X, classifier, idx):
        return classifier.predict(X) * self.weights_[idx]

    def _train_preds_for_estimator(self, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        indices = range(self.n_instances_)
        subsample = rng.choice(self.n_instances_, size=self.n_instances_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros(self.n_instances_)
        if len(oob) == 0:
            return results, 1, oob

        clf = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 10)),
        )
        clf.fit(self.transformed_data_[idx].iloc[subsample], y[subsample])
        preds = clf.predict(self.transformed_data_[idx].iloc[oob])

        weight = clf.steps[1][1].best_score_

        for n, pred in enumerate(preds):
            results[oob[n]] = pred * weight

        return results, weight, oob

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            params = {"num_kernels": 20, "n_estimators": 5}
        else:
            params = {
                "num_kernels": 10,
                "n_estimators": 2,
                "save_transformed_data": True,
            }

        return params
