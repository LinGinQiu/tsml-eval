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
    _tags = {
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "hybrid",
    }

    def __init__(self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        save_component_probas=False,
        verbose=0,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
        disable_modules=None,):
        """
        Custom HIVECOTEV2 to disable specified modules.

        Parameters
        ----------
        disable_modules : list of str or None
            List of module names to disable. Possible values: "STC", "DrCIF", "Arsenal", "TDE".
        """

        super().__init__(
            stc_params=stc_params,
            drcif_params=drcif_params,
            arsenal_params=arsenal_params,
            tde_params=tde_params,
            time_limit_in_minutes=time_limit_in_minutes,
            save_component_probas=save_component_probas,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )
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

    def _predict_proba(self, X, return_component_probas=False) -> np.ndarray:
        """Predicts label probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))

        # Only add probas if the classifier has been fitted (i.e., not disabled)
        if self._stc is not None:
            stc_probas = self._stc.predict_proba(X)
            dists += stc_probas * (np.ones(self.n_classes_) * self.stc_weight_)
        else:
            stc_probas = None

        if self._drcif is not None:
            drcif_probas = self._drcif.predict_proba(X)
            dists += drcif_probas * (np.ones(self.n_classes_) * self.drcif_weight_)
        else:
            drcif_probas = None

        if self._arsenal is not None:
            arsenal_probas = self._arsenal.predict_proba(X)
            dists += arsenal_probas * (np.ones(self.n_classes_) * self.arsenal_weight_)
        else:
            arsenal_probas = None

        if self._tde is not None:
            tde_probas = self._tde.predict_proba(X)
            dists += tde_probas * (np.ones(self.n_classes_) * self.tde_weight_)
        else:
            tde_probas = None

        # Save component probas if requested
        if self.save_component_probas:
            self.component_probas = {
                "STC": stc_probas,
                "DrCIF": drcif_probas,
                "Arsenal": arsenal_probas,
                "TDE": tde_probas,
            }

        # If all components are disabled (unlikely), avoid division by zero
        sum_dists = dists.sum(axis=1, keepdims=True)
        sum_dists[sum_dists == 0] = 1e-8

        return dists / sum_dists

if __name__ == "__main__":
    X = np.random.randn(100, 1, 100)
    y = np.random.choice([0, 0, 1], size=100)
    print(np.unique(y, return_counts=True))

    hc2 = HIVECOTEV2_Custom(
        disable_modules=["STC"],)
    # Fit and transform
    hc2.fit(X, y)
    hc2.predict(X)
