from tsml_eval._wip.rt.transformations.collection.imbalance._ibgan import IBGANAugmenter
from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import VOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._cfam import CFAM
from tsml_eval._wip.rt.transformations.collection.imbalance._smote import SMOTE
from tsml_eval._wip.rt.transformations.collection.imbalance._ohit import OHIT
from tsml_eval._wip.rt.transformations.collection.imbalance._tsmote import TSMOTE

from tsml_eval._wip.rt.transformations.collection.imbalance._utils import evaluate_data

from aeon.transformations.collection import BaseCollectionTransformer
import numpy as np

class rebalance_eval(BaseCollectionTransformer):
    _tags = {
        "requires_y": True,
        "capability:multivariate": True,
    }

    def __init__(self,
                 dataset_name,
                 n_jobs=1,
                 random_state=0,):
        self.dataset_name = dataset_name
        self.n_jobs = n_jobs
        self.random_state = random_state
        ibgan = IBGANAugmenter(random_state=self.random_state, n_jobs=self.n_jobs)
        cfam = CFAM(random_state=self.random_state)
        ldg_lp = VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="lp",
            random_state=random_state,
        )
        ldg_l = VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="latent",
            random_state=random_state,
        )
        ldg_p = VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="pair",
            random_state=random_state,
        )
        ohit = OHIT(distance="euclidean", random_state=random_state)
        smot = SMOTE(
            n_neighbors=5,
            distance="euclidean",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        tsmot = TSMOTE(
            random_state=random_state,
            spy_size=0.15,
            window_size=None,
            distance="euclidean",
            distance_params=None,
        )

        self.augmenters = {
            "CFAM": cfam,
            "VOTE_LP": ldg_lp,
            "VOTE_L": ldg_l,
            "VOTE_P": ldg_p,
            "OHIT": ohit,
            "SMOTE": smot,
            "TSMOTE": tsmot,
            "IBGAN": ibgan,
        }
        super().__init__()

    def _fit(self, X, y=None):
        for name, augmenter in self.augmenters.items():
            print(f"Fitting augmenter: {name}")
            augmenter.fit(X, y)
        return self

    def _transform(self, X, y=None):
        results = {}
        ori_data = X.copy()
        for name, augmenter in self.augmenters.items():
            print(f"Transforming data using augmenter: {name}")
            X_resampled, y_resampled = augmenter.transform(X, y)
            gen_data = augmenter._generated_samples
            print(f"Generated samples: {gen_data.shape}")
            result = evaluate_data(ori_data, gen_data, dataset_name=self.dataset_name, model_name=name)
            results[name] = result
            print(f"Evaluation results for {name}: {result}")
        import sys
        print("Rebalance evaluation completed.")
        sys.exit(0)


if __name__ == "__main__":
    dataset_name = 'AllGestureWiimoteX_eq'
    smote = rebalance_eval(dataset_name=dataset_name)
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    print(np.unique(y_train, return_counts=True))
    arr = X_test
    # 检查是否有 NaN
    print(np.isnan(arr).any())  # True

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
