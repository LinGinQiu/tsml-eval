# -*- coding: utf-8 -*-
"""A tsml wrapper for sklearn regressors."""

__author__ = ["DGuijoRubio", "MatthewMiddlehurst"]
__all__ = ["SklearnToTsmlRegressor"]

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class SklearnToTsmlRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators."""

    def __init__(self, regressor=None, concatenate_channels=False, random_state=None):
        self.regressor = regressor
        self.concatenate_channels = concatenate_channels
        self.random_state = random_state

        super(SklearnToTsmlRegressor, self).__init__()

    def fit(self, X, y):
        if self.regressor is None:
            raise ValueError("Regressor not set")

        X, y = self._validate_data(X=X, y=y)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        self._regressor = _clone_estimator(self.regressor, self.random_state)
        self._regressor.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X, concatenate_channels=self.concatenate_channels)

        return self._regressor.predict(X)

    def _more_tags(self):
        return {"X_types": ["2darray"]}
