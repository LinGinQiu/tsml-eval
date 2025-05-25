from aeon.classification.hybrid import HIVECOTEV2
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.interval_based._drcif import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


class HIVECOTEV2_Custom(HIVECOTEV2):

    def __init__(self, *args, disable_modules=None, **kwargs):
        """
        Custom HIVECOTEV2 to disable specified modules.

        Parameters
        ----------
        disable_modules : list of str or None
            List of module names to disable. Possible values: "STC", "DrCIF", "Arsenal", "TDE".
        """
        super().__init__(*args, **kwargs)
        self.disable_modules = disable_modules or []

    def _fit(self, X, y):
        if self.stc_params is None:
            self._stc_params = {"n_shapelet_samples": HIVECOTEV2._DEFAULT_N_SHAPELETS}
        if self.drcif_params is None:
            self._drcif_params = {"n_estimators": HIVECOTEV2._DEFAULT_N_TREES}
        if self.arsenal_params is None:
            self._arsenal_params = {
                "n_kernels": HIVECOTEV2._DEFAULT_N_KERNELS,
                "n_estimators": HIVECOTEV2._DEFAULT_N_ESTIMATORS,
            }
        if self.tde_params is None:
            self._tde_params = {
                "n_parameter_samples": HIVECOTEV2._DEFAULT_N_PARA_SAMPLES,
                "max_ensemble_size": HIVECOTEV2._DEFAULT_MAX_ENSEMBLE_SIZE,
                "randomly_selected_params": HIVECOTEV2._DEFAULT_RAND_PARAMS,
            }

        ct = self.time_limit_in_minutes / 6 if self.time_limit_in_minutes > 0 else 0
        if ct > 0:
            self._stc_params["time_limit_in_minutes"] = ct
            self._drcif_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

        # Fit STC
        if "STC" not in self.disable_modules:
            self._stc = ShapeletTransformClassifier(
                **self._stc_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            train_preds = self._stc.fit_predict(X, y)
            self.stc_weight_ = accuracy_score(y, train_preds) ** 4
        else:
            self._stc = None
            self.stc_weight_ = 0
            if self.verbose > 0:
                print("STC disabled")

        # Fit DrCIF
        if "DrCIF" not in self.disable_modules:
            self._drcif = DrCIFClassifier(
                **self._drcif_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            train_preds = self._drcif.fit_predict(X, y)
            self.drcif_weight_ = accuracy_score(y, train_preds) ** 4
        else:
            self._drcif = None
            self.drcif_weight_ = 0
            if self.verbose > 0:
                print("DrCIF disabled")

        # Fit Arsenal
        if "Arsenal" not in self.disable_modules:
            self._arsenal = Arsenal(
                **self._arsenal_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            train_preds = self._arsenal.fit_predict(X, y)
            self.arsenal_weight_ = accuracy_score(y, train_preds) ** 4
        else:
            self._arsenal = None
            self.arsenal_weight_ = 0
            if self.verbose > 0:
                print("Arsenal disabled")

        # Fit TDE
        if "TDE" not in self.disable_modules:
            self._tde = TemporalDictionaryEnsemble(
                **self._tde_params,
                random_state=self.random_state,
                n_jobs=self._n_jobs,
            )
            train_preds = self._tde.fit_predict(X, y)
            self.tde_weight_ = accuracy_score(y, train_preds) ** 4
        else:
            self._tde = None
            self.tde_weight_ = 0
            if self.verbose > 0:
                print("TDE disabled")

        return self