from collections import OrderedDict
from typing import Optional, Union
import tqdm
import numpy as np
from numba import prange
from sklearn.utils import check_random_state

from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
from tsml_eval._wip.rt.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer


__maintainer__ = ["chrisholder"]
__all__ = ["ESMOTE"]

from tsml_eval._wip.rt.utils._threading import threaded

import numpy as np
from typing import Tuple, Dict

def _neighbor_consensus_mask(
        diffs: np.ndarray,
        eps: float = 1e-12,
        sign_tau: float = 0.6,
        spread_tau: float = 0.6,
        q_mag: float = 0.2,
        combine: str = "geom",  # "geom": sign*(1-spread); "lin": a*sign + b*(1-spread)
        a: float = 0.5,  # weight for sign in linear combine
        b: float = 0.5,  # weight for (1-spread) in linear combine
        min_effective: int = 2,  # require at least this many effective neighbors
        smooth_window: int = 0,  # optional temporal smoothing window for r (0=off)
) -> tuple[np.ndarray, dict]:
    """
    Compute per-timestep neighbor consensus (mask + soft score) from deformations.
    diffs: (m, T) with d_i(t) = proj_i(t) - anchor(t)

    Returns:
      mask: (T,) bool   -- hard consensus (direction agreement & low dispersion)
      info: dict with:
            'sign_consistency': (T,) in [0,1]
            'spread': (T,) in [0,1]
            'r': (T,) soft consensus score in [0,1]
            'mag_ref': (T,)
            'effective_count': (T,) int
    """
    m, T = diffs.shape

    # Robust magnitude reference per timestep
    abs_d = np.abs(diffs)
    mag_ref = np.median(abs_d, axis=0) + eps  # (T,)

    # Per-timestep negligible-magnitude gating (avoid random sign near zero)
    # eps_t depends on the distribution at each t rather than a global threshold.
    eps_t = np.quantile(abs_d, q_mag, axis=0)  # (T,)
    effective = abs_d >= eps_t[None, :]  # (m, T)
    eff_count = effective.sum(axis=0)  # (T,)

    # Direction (sign) consensus in [0,1]
    signs = np.sign(diffs) * effective.astype(diffs.dtype)
    denom = np.maximum(eff_count, 1)  # avoid /0; if 0 -> returns 0
    sign_cons = np.abs(signs.sum(axis=0) / denom)

    # Magnitude dispersion in [0,1]: robust MAD normalized by mag_ref
    mad = np.median(np.abs(abs_d - np.median(abs_d, axis=0)), axis=0)  # (T,)
    # Fallback if MAD is degenerate
    mad0 = (mad <= eps)
    if np.any(mad0):
        std = abs_d.std(axis=0)
        mad[mad0] = std[mad0]
    spread = np.clip(mad / mag_ref, 0.0, 1.0)

    # Soft consensus r(t) in [0,1]
    if combine == "lin":
        r = np.clip(a * sign_cons + b * (1.0 - spread), 0.0, 1.0)
    else:  # geometric-like combine is stricter when any component is weak
        r = np.clip(sign_cons * (1.0 - spread), 0.0, 1.0)

    # Optional temporal smoothing on r
    if smooth_window and smooth_window > 1:
        w = smooth_window
        # simple box filter; keep ends unchanged if you prefer
        r_sm = np.copy(r)
        half = w // 2
        for t in range(T):
            t0 = max(0, t - half)
            t1 = min(T, t + half + 1)
            r_sm[t] = r[t0:t1].mean()
        r = r_sm

    # Hard consensus mask
    # Also require a minimum effective voter count to avoid spurious agreement.
    mask = (sign_cons >= sign_tau) & (spread <= spread_tau) & (eff_count >= min_effective)

    info = {
        "sign_consistency": sign_cons,
        "spread": spread,
        "r": r,
        "mag_ref": mag_ref,
        "effective_count": eff_count,
        "eps_t": eps_t,
    }
    return mask, info

class ESMOTE(BaseCollectionTransformer):
    """
    Over-sampling using the Synthetic Minority Over-sampling TEchnique (SMOTE)[1]_.
    An adaptation of the imbalance-learn implementation of SMOTE in
    imblearn.over_sampling.SMOTE. sampling_strategy is sampling target by
    targeting all classes but not the majority, which is directly expressed in
    _fit.sampling_strategy.
    Parameters
    ----------
    n_neighbors : int, default=5
        The number  of nearest neighbors used to define the neighborhood of samples
        to use to generate the synthetic time series.
        `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this case.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    See Also
    --------
    ADASYN
    References
    ----------
    .. [1] Chawla et al. SMOTE: synthetic minority over-sampling technique, Journal
    of Artificial Intelligence Research 16(1): 321–357, 2002.
        https://dl.acm.org/doi/10.5555/1622407.1622416
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_neighbors=5,
            distance: Union[str, callable] = "twe",
        distance_params: Optional[dict] = None,
        weights: Union[str, callable] = "uniform",
            set_dangerous: bool = False,
            set_barycentre_averaging: bool = False,
            set_inner_add: bool = False,
            two_part_strategy: bool = False,
            iteration_generate: bool = False,
            use_interpolated_path: bool = False,
            use_soft_distance: bool = False,
            use_multi_soft_distance: bool = False,
        n_jobs: int = 1,
        random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs
        self.set_dangerous = set_dangerous
        self.set_barycentre_averaging = set_barycentre_averaging
        self.set_inner_add = set_inner_add
        self.two_part_strategy = two_part_strategy
        self.iteration_generate = iteration_generate
        self.use_interpolated_path = use_interpolated_path
        self.use_soft_distance = use_soft_distance
        self.use_multi_soft_distance = use_multi_soft_distance
        self._random_state = None
        self._distance_params = distance_params or {}

        self.nn_ = None
        super().__init__()

    def _fit(self, X, y=None):

        self._random_state = check_random_state(self.random_state)

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        num_minority = min(counts)
        suggested_n_neighbors = int(min(2+ 0.1*num_minority, self.n_neighbors))
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))
        self.suggested_n_neighbors_ = suggested_n_neighbors

        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        self.nn_ = KNeighborsTimeSeriesClassifier(
            n_neighbors=self.suggested_n_neighbors_ + 1,
            distance=self.distance,
            distance_params=self._distance_params,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )

        # got the minority class label and the number needs to be generated
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue

            # n_samples = 2*n_samples  # increase the number of samples to process selection
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            y_class = y[target_class_indices]

            if self.set_barycentre_averaging:
                X_new = np.zeros((n_samples, *X.shape[1:]), dtype=X.dtype)
                majority_class_indices = np.flatnonzero(y != class_sample)
                X_majority = X[majority_class_indices]
                y_majority = y[majority_class_indices]
                self.nn_.fit(X_majority, y_majority)
                X_class_size = len(X_class)
                for n in range(n_samples):
                    # randomly select subset samples to generate a new sample
                    subset = max(4, X_class_size // 10 + 1)
                    index_subset_series = self._random_state.choice(len(X_class), size=subset, replace=False)
                    X_sub = X_class[index_subset_series]
                    # random_one = self._random_state.choice(len(X_majority))
                    # X_class = np.concatenate([X_class, X_majority[random_one][None, ...]], axis=0)
                    step = self._random_state.uniform(low=0, high=1)
                    X_new_one = self._generate_sample_use_elastic_distance(X_sub[0], X_sub[1:],
                                                                          distance=self.distance,
                                                                          step=step,
                                                                          )

                    # calculate the l2 distance between X_sub[0] and X_sub[1:],
                    distances = np.linalg.norm(X_sub[1:].squeeze() - X_sub[0].squeeze(), axis=1)
                    distance_sum = np.sum(distances)
                    nearest_one = X_majority[self.nn_.kneighbors(X=X_new_one, return_distance=False)[0, 0]]
                    distance_vary = np.sqrt(np.sum((X_new_one.squeeze() - nearest_one.squeeze()) ** 2))
                    if distance_vary < distance_sum:
                        X_new[n] = X_new_one
                    else:
                        X_sub = np.concatenate([X_sub, nearest_one[None, ...]], axis=0)
                        X_new[n] = self._generate_sample_use_elastic_distance(X_sub[0], X_sub[1:],
                                                                              distance=self.distance,
                                                                              step=step,
                                                                              )


                y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
                X_resampled.append(X_new)
                y_resampled.append(y_new)
                X_synthetic = np.vstack(X_resampled)
                y_synthetic = np.hstack(y_resampled)
                return X_synthetic, y_synthetic

            if self.two_part_strategy:
                self.nn_.fit(X, y)
                global_nn = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
                X_class_replaced = []
                X_class_dangerous = []
                X_dangerous_nns = []

                for i in range(len(X_class)):
                    # for each minority class sample, if its global nearest neighbors are not include the minority class, skip it
                    triger = np.isin(target_class_indices, global_nn[i]).any()
                    if triger:
                        X_class_replaced.append(X_class[i])
                    else:
                        X_class_dangerous.append(X_class[i])
                        X_dangerous_nns.append(global_nn[i])

                X_class_new = np.array(X_class_replaced)
                n_samples_new = int(n_samples * (len(X_class_new) / len(X_class)))
                if len(X_class_new) > 1:
                    if len(X_class_new) <= self.suggested_n_neighbors_:
                        self.suggested_n_neighbors_ = len(X_class_new) - 1
                    self.nn_temp_ = KNeighborsTimeSeriesClassifier(
                        n_neighbors=self.suggested_n_neighbors_ + 1,
                        distance=self.distance,
                        distance_params=self._distance_params,
                        weights=self.weights,
                        n_jobs=self.n_jobs,
                    )
                    self.nn_temp_.fit(X_class_new, y_class[:len(X_class_new)])
                    nns = self.nn_temp_.kneighbors(X=X_class_new, return_distance=False)[:, 1:]

                    X_new, y_new = self._make_samples(
                        X_class_new,
                        y.dtype,
                        class_sample,
                        X_class_new,
                        nns,
                        n_samples_new,
                        1.0,
                        n_jobs=self.n_jobs,
                    )
                    X_resampled.append(X_new)
                    y_resampled.append(y_new)
                if len(X_class_dangerous) > 0 and self.set_dangerous:
                    self.nn_temp_ = KNeighborsTimeSeriesClassifier(
                        n_neighbors=1,
                        distance=self.distance,
                        distance_params=self._distance_params,
                        weights=self.weights,
                        n_jobs=self.n_jobs,
                    )
                    self.nn_temp_.fit(X_class_new, y_class[:len(X_class_new)])
                    nns = self.nn_temp_.kneighbors(X=X_class_dangerous, return_distance=False)

                    n_samples_dangerous = n_samples - n_samples_new
                    X_new_dangerous, y_new_dangerous = self._make_samples_for_dangerous(
                        X_class_dangerous,
                        y.dtype,
                        class_sample,
                        X,
                        X_dangerous_nns,
                        X_class_new,
                        nns,
                        n_samples_dangerous,
                        n_jobs=self.n_jobs)
                    X_resampled.append(X_new_dangerous)
                    y_resampled.append(y_new_dangerous)
            elif self.iteration_generate:
                # --- Build a discriminator list based on a priority list ---
                distance_funcs = ["adtw", "twe", "msm"]
                discriminators = []
                discriminator_names = []
                # Priority: MRHydra, RocketClassifier, RandomForestClassifier
                # Try to import and instantiate each, fit on (X, y), add to discriminators if successful
                # 1. MRHydra
                try:
                    from aeon.classification.convolution_based import MultiRocketHydraClassifier as MRHydra
                    disc = MRHydra(n_jobs=self.n_jobs, random_state=self.random_state)
                    disc.fit(X, y)
                    discriminators.append(disc)
                    discriminator_names.append("MRHydra")
                except Exception:
                    pass
                # 2. Rocket+RidgeClassifierCV
                try:
                    from aeon.classification.convolution_based import RocketClassifier
                    disc2 = RocketClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                    disc2.fit(X, y)
                    # patch classes_ attribute for compatibility
                    discriminators.append(disc2)
                    discriminator_names.append("RocketClassifier")
                except Exception:
                    pass
                # 3. RandomForestClassifier
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    # flatten for tabular
                    X_flat = X.reshape((X.shape[0], -1))
                    disc3 = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs)
                    disc3.fit(X_flat, y)
                    # patch for compatibility (predict/predict_proba expects flat)
                    disc3._esmote_flatten = True
                    disc3.classes_ = disc3.classes_
                    discriminators.append(disc3)
                    discriminator_names.append("RandomForestClassifier")
                except Exception:
                    pass
                # Fallback KNN-DTW will be appended later only if needed
                knn_fallback_added = False
                knn_fallback_idx = None

                accept_count = {d: 0 for d in distance_funcs}
                X_new_parts = []
                y_new_parts = []
                X_iter = X_class.copy()
                y_iter = y_class.copy()
                if len(X_iter) == 1:
                    X_iter = np.concatenate([X_iter, X_iter], axis=0)
                    y_iter = np.concatenate([y_iter, y_iter], axis=0)
                num_iters = 5
                remaining = n_samples
                max_pool_ratio = 1.5
                thr_start, thr_end = 0.65, 0.85

                for it in range(num_iters):
                    if remaining <= 0:
                        break
                    if it < 2:
                        distance_func = self._random_state.choice(distance_funcs)
                    else:
                        weights = np.array([accept_count[d] + 1 for d in distance_funcs], dtype=float)
                        weights = weights / weights.sum()
                        distance_func = self._random_state.choice(distance_funcs, p=weights)
                    self.distance = distance_func
                    trial = int(np.ceil(remaining / max(1, (num_iters - it))))
                    trial = max(1, trial)
                    k = min(self.suggested_n_neighbors_ + 1, len(X_iter))
                    k = max(2, k)
                    nn_temp_ = KNeighborsTimeSeriesClassifier(
                        n_neighbors=k,
                        distance=self.distance,
                        distance_params=self._distance_params,
                        weights=self.weights,
                        n_jobs=self.n_jobs,
                    )
                    nn_temp_.fit(X_iter, y_iter)
                    nns = nn_temp_.kneighbors(X=X_iter, return_distance=False)[:, 1:]
                    X_try, y_try = self._make_samples(
                        X_iter,
                        y.dtype,
                        class_sample,
                        X_iter,
                        nns,
                        trial,
                        1.0,
                        n_jobs=self.n_jobs,
                    )
                    prob_list = []
                    for di, disc in enumerate(discriminators):
                        try:
                            # flatten if required
                            if hasattr(disc, "_esmote_flatten") and disc._esmote_flatten:
                                X_try_pred = X_try.reshape((X_try.shape[0], -1))
                            else:
                                X_try_pred = X_try
                            pred_try = disc.predict(X_try_pred)
                            # Only skip if all predictions are for a single class
                            if np.unique(pred_try).size == 1:
                                continue
                            if hasattr(disc, "_esmote_flatten") and disc._esmote_flatten:
                                prob = disc.predict_proba(X_try_pred)
                            else:
                                prob = disc.predict_proba(X_try_pred)
                        except Exception:
                            continue
                        # get correct class index
                        class_indices = [i for i, c in enumerate(disc.classes_) if c == class_sample]
                        if not class_indices:
                            continue
                        min_idx = class_indices[0]
                        p = prob[:, min_idx]
                        if np.std(p) < 1e-3:
                            continue
                        prob_list.append(p)
                    # If no valid discriminators, try to add KNN-DTW fallback if not already added
                    if not prob_list:
                        if not knn_fallback_added:
                            try:
                                knn_fallback = KNeighborsTimeSeriesClassifier(
                                    n_neighbors=min(3, len(X)),
                                    distance="dtw",
                                    n_jobs=1,
                                    weights="uniform",
                                )
                                knn_fallback.fit(X, y)
                                discriminators.append(knn_fallback)
                                discriminator_names.append("KNN-DTW")
                                knn_fallback_added = True
                                knn_fallback_idx = len(discriminators) - 1
                            except Exception:
                                knn_fallback_added = False
                        # Try using the KNN-DTW fallback if it was added
                        if knn_fallback_added and knn_fallback_idx is not None:
                            disc = discriminators[knn_fallback_idx]
                            try:
                                prob = disc.predict_proba(X_try)
                                class_indices = [i for i, c in enumerate(disc.classes_) if c == class_sample]
                                if class_indices:
                                    min_idx = class_indices[0]
                                    p = prob[:, min_idx]
                                    if np.std(p) >= 1e-3:
                                        prob_list.append(p)
                            except Exception:
                                pass
                    # After fallback, if still empty, accept all
                    if not prob_list:
                        idx_keep = np.arange(len(X_try))
                    else:
                        p_min = np.mean(prob_list, axis=0)
                        thr = thr_start + (thr_end - thr_start) * (it / max(1, num_iters - 1))
                        idx_keep = np.where(p_min >= thr)[0]
                        if idx_keep.size == 0:
                            topk = max(1, min(len(p_min), trial // 4 if trial >= 4 else 1))
                            idx_keep = np.argsort(-p_min)[:topk]
                    X_kept = X_try[idx_keep]
                    y_kept = y_try[idx_keep]
                    accept_count[distance_func] += len(idx_keep)
                    if len(X_kept) > 0:
                        X_new_parts.append(X_kept)
                        y_new_parts.append(y_kept)
                        remaining -= len(X_kept)
                        if (len(X_iter) / max(1, len(X_class))) < max_pool_ratio:
                            X_iter = np.concatenate([X_iter, X_kept], axis=0)
                            y_iter = np.concatenate([y_iter, y_kept], axis=0)
                # If we still need more, backfill using the best-performing distance so far
                if remaining > 0:
                    best_distance = max(distance_funcs, key=lambda d: accept_count[d])
                    self.distance = best_distance
                    k = min(self.suggested_n_neighbors_ + 1, len(X_iter))
                    k = max(2, k)
                    nn_temp_ = KNeighborsTimeSeriesClassifier(
                        n_neighbors=k,
                        distance=self.distance,
                        distance_params=self._distance_params,
                        weights=self.weights,
                        n_jobs=self.n_jobs,
                    )
                    nn_temp_.fit(X_iter, y_iter)
                    nns = nn_temp_.kneighbors(X=X_iter, return_distance=False)[:, 1:]
                    X_back, y_back = self._make_samples(
                        X_iter,
                        y.dtype,
                        class_sample,
                        X_iter,
                        nns,
                        remaining,
                        1.0,
                        n_jobs=self.n_jobs,
                    )
                    X_new_parts.append(X_back)
                    y_new_parts.append(y_back)
                if len(X_new_parts) > 0:
                    X_new = np.vstack(X_new_parts)[:n_samples]
                    y_new = np.hstack(y_new_parts)[:n_samples]
                    X_resampled.append(X_new)
                    y_resampled.append(y_new)
            else:
                self.nn_temp_ = KNeighborsTimeSeriesClassifier(
                    n_neighbors=self.suggested_n_neighbors_ + 1,
                    distance=self.distance,
                    distance_params=self._distance_params,
                    weights=self.weights,
                    n_jobs=self.n_jobs,
                )
                self.nn_temp_.fit(X_class, y_class[:len(X_class)])
                nns = self.nn_temp_.kneighbors(X=X_class, return_distance=False)[:, 1:]

                X_new, y_new = self._make_samples(
                    X_class,
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    n_samples,
                    1.0,
                    n_jobs=self.n_jobs,
                )
                X_resampled.append(X_new)
                y_resampled.append(y_new)

        X_synthetic = np.vstack(X_resampled)
        y_synthetic = np.hstack(y_resampled)
        return X_synthetic, y_synthetic

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, n_jobs=1
    ):
        if self.use_multi_soft_distance:
            X_new = np.zeros((n_samples, *X.shape[1:]), dtype=X.dtype)
            for i in range(n_samples):
                sample_index = self._random_state.randint(low=0, high=len(X))
                nn_indices = nn_num[sample_index]
                nn_indices_minus1 = self._random_state.choice(nn_indices, size=3, replace=False)  # select 3 neighbors
                nn_ts = nn_data[nn_indices_minus1]
                step = step_size * self._random_state.uniform(low=0, high=1.4)
                X_new[i] = self._generate_sample_use_elastic_distance(X[sample_index], nn_ts,
                                                                      distance=self.distance,
                                                                      step=step,
                                                                      )
        else:
            samples_indices = self._random_state.randint(
                low=0, high=nn_num.size, size=n_samples
            )
            steps = step_size * self._random_state.uniform(low=0, high=1, size=n_samples)[:, np.newaxis]
            rows = np.floor_divide(samples_indices, nn_num.shape[1])
            cols = np.mod(samples_indices, nn_num.shape[1])
            X_new = np.zeros((len(rows), *X.shape[1:]), dtype=X.dtype)
            for count in range(len(rows)):
                i = rows[count]
                j = cols[count]
                nn_ts = nn_data[nn_num[i, j]]
                X_new[count] = self._generate_sample_use_elastic_distance(X[i], nn_ts, distance=self.distance,
                                                                          step=steps[count],
                                                                          )

        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _make_samples_for_dangerous(self, X_class_dangerous, y_dtype, y_type, X, X_dangerous_nns, X_class_new, nns,
                                    n_samples, n_jobs=1):
        """
        Generate samples for dangerous class samples that are not in the neighborhood of the majority class.
        Parameters
        ----------
        X_class_dangerous
        y_dtype
        X
        X_dangerous_nns : np.ndarray
            The nearest neighbors of the dangerous class samples.
        n_samples
        n_jobs

        Returns
        X_new : np.ndarray
            The generated samples for the dangerous class.
        y_new : np.ndarray
            The labels for the generated samples, all set to the class type of the
        -------

        """
        samples_indices = self._random_state.randint(
            low=0, high=len(X_class_dangerous), size=n_samples
        )
        steps = 0.5 * self._random_state.uniform(low=0, high=1, size=n_samples)[:, np.newaxis]
        X_new = np.zeros((n_samples, *X.shape[1:]), dtype=X.dtype)
        for new_index, sample_index in enumerate(samples_indices):
            nn_ts_index = self._random_state.choice(X_dangerous_nns[sample_index])
            nn_ts_m_index = self._random_state.choice(nns[sample_index])
            # for each dangerous sample, generate a bias using its nearest neighbor
            nn_ts = X[nn_ts_index]
            nn_ts_m = X_class_new[nn_ts_m_index]
            bias_add = self._generate_sample_use_elastic_distance(X_class_dangerous[sample_index], nn_ts,
                                                           distance=self.distance,
                                                           step=steps[sample_index], return_bias=True, )
            bias_minus = self._generate_sample_use_elastic_distance(X_class_dangerous[sample_index],
                                                                    distance=self.distance, nn_ts=nn_ts_m,
                                                                    step=(1.0 - steps[sample_index]),
                                                                    return_bias=True, )
            X_new[new_index] = X_class_dangerous[sample_index] + bias_add - bias_minus

        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_sample_use_elastic_distance(self, curr_ts, nn_ts, distance, step,
                                           window: Union[float, None] = None,
                                           g: float = 0.0,
                                           epsilon: Union[float, None] = None,
                                           nu: float = 0.001,
                                           lmbda: float = 1.0,
                                           independent: bool = True,
                                           c: float = 1.0,
                                           descriptor: str = "identity",
                                           reach: int = 15,
                                           warp_penalty: float = 1.0,
                                           transformation_precomputed: bool = False,
                                           transformed_x: Optional[np.ndarray] = None,
                                           transformed_y: Optional[np.ndarray] = None,
                                           return_bias=False,
                                           ):

        """
        Generate a single synthetic sample using soft distance.
        .
        """
        # shape: (c, l)
        # shape: (c, l)
        new_ts = curr_ts.copy()
        if self.set_barycentre_averaging or self.use_multi_soft_distance:
            reshape_ts = False
            if new_ts.ndim == 2:
                new_ts = new_ts.squeeze()
                curr_ts = curr_ts.squeeze()
                nn_ts = nn_ts.squeeze()
                reshape_ts = True
            if self.set_barycentre_averaging:
                distance = 'msm'  # Barycentre averaging is only applicable with MSM distance
                max_iter = 5
                centre = new_ts  # Initial centre is the current time series
                n_time_points = new_ts.shape[0]
                alignment = np.zeros(n_time_points)  # Stores the sum of values warped to each point
                num_warps_to = np.zeros(n_time_points)  # Tracks how many times each point is warped to
                if nn_ts.ndim == 1:
                    nn_ts = [nn_ts]
                for i in range(max_iter):

                    for Xi in [curr_ts, *nn_ts]:
                        # Assume msm_alignment_path computes the alignment path.
                        # It's important that this function provides the full path, not just the distance.

                        curr_alignment, _ = _get_alignment_path(
                            centre,
                            Xi,
                            distance,
                            window,
                            g,
                            epsilon,
                            nu,
                            lmbda,
                            independent,
                            c,
                            descriptor,
                            reach,
                            warp_penalty,
                            transformation_precomputed,
                            transformed_x,
                            transformed_y,
                        )

                        for j, k in curr_alignment:
                            alignment[k] += Xi[j]
                            num_warps_to[k] += 1

                    # Avoid division by zero for points that were never warped to
                    # If a point was never warped to, we can set it to 1 to avoid
                    num_warps_to[num_warps_to == 0] = 1
                    new_centre = alignment / num_warps_to

                    # Check for convergence. If the new centre is not significantly different, stop.
                    # This is a simplified check. A more robust check would involve the sum of squared distances.
                    if np.array_equal(new_centre, centre):
                        break

                    centre = new_centre
                new_ts = centre
            elif self.use_multi_soft_distance:
                T = len(curr_ts)
                projs, confs = [], []

                for nb in nn_ts:
                    # 1) Soft alignment weights (row-stochastic W)
                    W = self._generate_sample_use_soft_distance(
                        curr_ts,
                        nb,
                        distance=distance,
                        step=step,
                        return_weights=True,
                    )
                    # Ensure neighbor is 1D (T_neighbor,)
                    nb_1d = nb.squeeze()
                    # Project neighbor onto current (anchor) timeline
                    proj = W @ nb_1d  # (T,)

                    # 2) Confidence from row entropy: conf = 1 - H_row
                    P = np.clip(W, 1e-12, None)
                    ent = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
                    conf = 1.0 - ent  # (T,) in [0,1]

                    projs.append(proj)
                    confs.append(conf)

                projs = np.stack(projs, axis=0)  # (m, T)
                confs = np.stack(confs, axis=0)  # (m, T)

                # 3) Robust deformation direction on anchor timeline
                direction = np.median(projs, axis=0) - curr_ts  # (T,)

                # 4) Aggregate confidence and compute soft neighbor consensus
                conf_mean = confs.mean(axis=0)  # (T,) in [0,1]
                diffs = projs - curr_ts[None, :]  # (m, T)
                cons_mask, cons_info = _neighbor_consensus_mask(
                    diffs,
                    sign_tau=0.6,
                    spread_tau=0.6,
                )
                # Soft consensus score r(t) in [0,1]
                r = cons_info["sign_consistency"] * (1.0 - cons_info["spread"])  # (T,)
                r = np.clip(r, 0.0, 1.0)

                # 5) Build per-timestep gamma caps via soft gating by conf & consensus (smoothed)
                #    s(t) = w_conf * conf_mean(t) + w_cons * r(t), optionally sharpened
                w_conf, w_cons, sharp = 0.5, 0.5, 1.5
                s = np.clip(w_conf * conf_mean + w_cons * r, 0.0, 1.0)
                if sharp != 1.0:
                    s = np.power(s, sharp)

                # ---- Smooth s over time to avoid jagged pointwise decisions ----
                def _smooth1d(x: np.ndarray, win: int = 5) -> np.ndarray:
                    win = max(1, int(win))
                    if win == 1:
                        return x
                    k = np.ones(win, dtype=float) / float(win)
                    # reflect padding to preserve edges better
                    pad = win // 2
                    xp = np.pad(x, (pad, pad), mode="reflect")
                    xs = np.convolve(xp, k, mode="valid")
                    return xs[: x.shape[0]]

                s_smooth = _smooth1d(s, win=5)  # small window for gentle continuity

                # ---- Soft gate: sigmoid around a center threshold to avoid hard cuts ----
                gate_center, gate_sharp = 0.5, 8.0  # center in [0,1], larger sharp -> crisper transition
                gate = 1.0 / (1.0 + np.exp(-gate_sharp * (s_smooth - gate_center)))  # in (0,1)

                # ---- Build coherent gamma curve ----
                # If step <= 1: interpolation only. Use smoothed score and soft gate to attenuate.
                # If step  > 1: blend interpolation (where gate low) and extrapolation (where gate high)
                T = s.shape[0]
                gamma = np.zeros(T, dtype=float)

                if step <= 1.0:
                    # Pure interpolation, capped by step, smoothly attenuated by gate
                    gamma = step * s_smooth * gate
                else:
                    # Mixed interp/extrap in a *continuous* manner
                    # Interp component (near off regions)
                    interp_comp = min(1.0, step) * s_smooth * (1.0 - gate)
                    # Extrap component (near on regions): 1 + alpha*s, where alpha = step-1
                    alpha = max(0.0, step - 1.0)
                    extrap_comp = gate * (1.0 + alpha * s_smooth)
                    gamma = interp_comp + extrap_comp

                # Final gentle smoothing on gamma to ensure segment consistency
                gamma = _smooth1d(gamma, win=5)

                x_new = curr_ts + gamma * direction
                return x_new

            if reshape_ts:
                new_ts = new_ts.reshape(1, -1)
            return new_ts

        elif self.use_interpolated_path:
            # 1) Compute alignment path. With _get_alignment_path(center=nn_ts, ts=curr_ts)
            # each returned tuple is (i_curr, j_nn). We will anchor time on i_curr to stay on curr_ts timeline.
            alignment, _ = _get_alignment_path(
                nn_ts,
                curr_ts,
                distance,
                window,
                g,
                epsilon,
                nu,
                lmbda,
                independent,
                c,
                descriptor,
                reach,
                warp_penalty,
                transformation_precomputed,
                transformed_x,
                transformed_y,
            )

            # --- Helpers ---
            def _safe_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_query: float) -> np.ndarray:
                """Predict y at x_query via robust linear fit.
                x_train: shape (M,)
                y_train: shape (M, C) where C = n_channels
                Returns shape (C,).
                """
                # Guard: need at least 2 distinct x to fit a line
                if x_train.size < 2 or np.unique(x_train).size < 2:
                    # Fallback to nearest neighbor in x
                    idx = int(np.argmin(np.abs(x_train - x_query))) if x_train.size > 0 else 0
                    return y_train[idx] if x_train.size > 0 else curr_ts[:, int(min(max(round(x_query), 0),
                                                                                    curr_ts.shape[-1] - 1))]

                # Center & scale x for numerical stability
                x_mean = x_train.mean()
                x_std = x_train.std()
                if x_std == 0:
                    x_std = 1.0
                x_norm = (x_train - x_mean) / x_std
                xq = (x_query - x_mean) / x_std

                # Build design matrix [x, 1]
                Xdm = np.column_stack([x_norm, np.ones_like(x_norm)])  # (M, 2)
                # Solve for each channel using least squares (more stable than polyfit here)
                # y_train is (M, C)
                # Solution: beta = argmin ||Xdm beta - y||. We'll do batched via lstsq per channel.
                C = y_train.shape[1]
                y_pred = np.empty((C,), dtype=float)
                # Use pinv based solve for stability
                pinv = np.linalg.pinv(Xdm)
                coef = pinv @ y_train  # (2, C)
                a, b = coef[0], coef[1]  # each (C,)
                y_pred = a * xq + b
                return y_pred

            # 2) Build candidate points (time, value) anchored on curr timeline (i_curr index)
            # For each alignment pair (i_curr, j_nn), we create a fused value at time t=i_curr.
            # Optionally, also create midpoints to densify local fitting.
            pts_t = []  # list of float times on curr timeline
            pts_v = []  # list of vectors, shape (C,)
            C, L = curr_ts.shape

            # Collect original aligned points (alignment yields (i_curr, j_nn))
            for i_curr, j_nn in alignment:
                t = float(i_curr)
                v = step * curr_ts[:, i_curr] + (1.0 - step) * nn_ts[:, j_nn]
                pts_t.append(t)
                pts_v.append(v)

            # Add midpoints between consecutive alignment entries (operate on curr timeline)
            for (i1, j1), (i2, j2) in zip(alignment[:-1], alignment[1:]):
                if i1 == i2:
                    continue
                t_mid = 0.5 * (i1 + i2)
                # Fuse curr at the left endpoint and nn at the right endpoint
                v_mid = step * curr_ts[:, i1] + (1.0 - step) * nn_ts[:, j2]
                pts_t.append(float(t_mid))
                pts_v.append(v_mid)

            # Convert to arrays and sort by time
            if len(pts_t) == 0:
                return new_ts  # nothing to do
            pts_t = np.asarray(pts_t, dtype=float)
            pts_v = np.stack(pts_v, axis=0)  # (M, C)
            order = np.argsort(pts_t)
            pts_t = pts_t[order]
            pts_v = pts_v[order]

            # 3) For each integer time t in [0, L-1], perform a local linear regression using a sliding window
            # Choose a small window radius in time units. Radius 1.5 ~ uses up to ~3 units around t.
            radius = 1.5
            new_vals = np.empty((C, L), dtype=float)
            for t_int in range(L):
                # Select neighborhood
                mask = np.abs(pts_t - t_int) <= radius
                if not np.any(mask):
                    # Fallback to nearest two points (or nearest one)
                    nearest_idx = np.argsort(np.abs(pts_t - t_int))[:min(3, len(pts_t))]
                    x_train = pts_t[nearest_idx]
                    y_train = pts_v[nearest_idx]  # (m, C)
                else:
                    x_train = pts_t[mask]
                    y_train = pts_v[mask]
                # Ensure right shape: (M,) and (M, C)
                y_train = np.asarray(y_train, dtype=float)
                if y_train.ndim == 1:
                    y_train = y_train[None, :]
                pred = _safe_linear_predict(x_train, y_train, float(t_int))  # (C,)
                new_vals[:, t_int] = pred

            new_ts = new_vals
            return new_ts
        if self.use_soft_distance:
            new_ts = self._generate_sample_use_soft_distance(
                curr_ts,
                nn_ts,
                distance,
                step,
                return_bias=return_bias,
            )
            return new_ts
        else:
            alignment, _ = _get_alignment_path(
                nn_ts,
                curr_ts,
                distance,
                window,
                g,
                epsilon,
                nu,
                lmbda,
                independent,
                c,
                descriptor,
                reach,
                warp_penalty,
                transformation_precomputed,
                transformed_x,
                transformed_y,
            )

            path_list = [[] for _ in range(curr_ts.shape[1])]
            for k, l in alignment:
                path_list[k].append(l)

            # num_of_alignments = np.zeros_like(curr_ts, dtype=np.int32)
            empty_of_array = np.zeros_like(curr_ts, dtype=float)  # shape: (c, l)

            for k, l in enumerate(path_list):
                if len(l) == 0:
                    import logging
                    logging.getLogger("aeon").setLevel(logging.WARNING)
                    logging.warning(
                        f"Alignment path for channel {k} is empty. "
                        "Returning the original time series.")
                    return new_ts

                key = self._random_state.choice(l)
                # Compute difference for all channels at this time step
                empty_of_array[:, k] = curr_ts[:, k] - nn_ts[:, key]

        #  apply_local_smoothing to empty_of_array
        # windowsize = int(np.ceil(curr_ts.shape[-1] * 0.1))  # 10% of the length
        # empty_of_array = apply_local_smoothing(empty_of_array, window_size=windowsize, mode='nearest')
        # apply_smooth_decay to empty_of_array
        # empty_of_array = apply_smooth_decay(empty_of_array)
        bias = step * empty_of_array
        if return_bias:
            return bias
        if (self.
                set_inner_add):
            # If set_inner_add is True, we add the bias to the current time series
            new_ts = new_ts + bias
        else:
            new_ts = new_ts - bias  # / num_of_alignments
        return new_ts

    def _generate_sample_use_soft_distance(self, curr_ts, nn_ts, distance, step,
                                           return_bias=False, return_weights=False,
                                           ):

        """
        Generate a single synthetic sample using soft distance.
        .
        """
        # shape: (c, l)
        # shape: (c, l)
        new_ts = curr_ts.copy()
        T = curr_ts.shape[-1]
        if distance == 'msm':
            from tsml_eval._wip.rt.distances.elastic import soft_msm_gradient
            A, _ = soft_msm_gradient(curr_ts, nn_ts)
        elif distance == 'dtw':
            from tsml_eval._wip.rt.distances.elastic import soft_dtw_gradient
            A, _ = soft_dtw_gradient(curr_ts, nn_ts)
        elif distance == 'twe':
            from tsml_eval._wip.rt.distances.elastic import soft_twe_gradient
            A, _ = soft_twe_gradient(curr_ts, nn_ts)
        elif distance == 'adtw':
            from tsml_eval._wip.rt.distances.elastic import soft_adtw_gradient
            A, _ = soft_adtw_gradient(curr_ts, nn_ts)
        else:
            raise ValueError(f"Soft distance not implemented for distance: {distance}")
        A = np.maximum(A, 0.0)
        row_sum = A.sum(axis=1, keepdims=True) + 1e-12
        W = A / row_sum
        if return_weights:
            return W
        proj = W @ nn_ts.squeeze()
        P = np.clip(W, 1e-12, None)
        ent = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
        conf = 1.0 - ent

        direction = proj - curr_ts
        if return_bias:
            return direction

        conf_tau = 0.4
        conf[conf < conf_tau] = 0.0
        s = np.clip(conf, 0.0, 1.0)
        s = np.power(s, 2.0)
        gamma = 0.0 + step * s
        new_ts = curr_ts + gamma * direction
        return new_ts

if __name__ == "__main__":
    smote = ESMOTE(n_neighbors=5, random_state=1, distance="twe", use_soft_distance=True, use_multi_soft_distance=True)
    # Example usage
    from local.load_ts_data import X_train, y_train, X_test, y_test

    print(np.unique(y_train, return_counts=True))

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

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
