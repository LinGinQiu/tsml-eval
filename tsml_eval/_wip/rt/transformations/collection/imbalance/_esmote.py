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
            iteration_generate: bool = True,
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
                for n in range(n_samples):
                    # randomly select 2 samples to generate a new sample
                    index_two_series = self._random_state.choice(len(X_class), size=2, replace=False)
                    X_class = X_class[index_two_series]
                    y_class = y_class[index_two_series]
                    step = self._random_state.uniform(low=0, high=1)
                    X_new[n] = self._generate_sample_use_elastic_distance(X_class[0], X_class[1],
                                                                          distance=self.distance,
                                                                          step=step,
                                                                          use_barycentre_averaging=True, )
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
                from aeon.classification.convolution_based import MultiRocketHydraClassifier as MRHydra
                discriminator = MRHydra(n_jobs=self.n_jobs, random_state=self.random_state)
                discriminator.fit(X, y)

                # --- Iterative generation with adaptive filtering ---
                # Use multiple elastic distances and track acceptance per distance
                distance_funcs = ["adtw", "twe", "msm"]
                accept_count = {d: 0 for d in distance_funcs}

                # Containers for accepted synthetic samples
                X_new_parts = []
                y_new_parts = []

                # Working pool that we may augment with accepted synthetics
                X_iter = X_class.copy()
                y_iter = y_class.copy()

                # Guard: if pool has a single sample, duplicate it to allow k-NN (k>=2 with self-exclusion)
                if len(X_iter) == 1:
                    X_iter = np.concatenate([X_iter, X_iter], axis=0)
                    y_iter = np.concatenate([y_iter, y_iter], axis=0)

                # Scheduling controls
                num_iters = 5
                remaining = n_samples  # how many we still need overall
                max_pool_ratio = 1.5  # cap size of (real+synthetic) pool vs. original minority size
                thr_start, thr_end = 0.65, 0.85  # dynamic probability threshold schedule

                for it in range(num_iters):
                    if remaining <= 0:
                        break

                    # Choose distance: uniform in early iters, then weighted by past acceptance
                    if it < 2:
                        distance_func = self._random_state.choice(distance_funcs)
                    else:
                        weights = np.array([accept_count[d] + 1 for d in distance_funcs], dtype=float)
                        weights = weights / weights.sum()
                        distance_func = self._random_state.choice(distance_funcs, p=weights)
                    self.distance = distance_func

                    # Size to try this round; ceil so we don't starve the tail
                    trial = int(np.ceil(remaining / max(1, (num_iters - it))))
                    trial = max(1, trial)

                    # Safe k for k-NN (we query k including self, then drop self via [:,1:])
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

                    # Propose synthetic samples
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

                    # Score by discriminator; keep high-probability minority
                    prob = discriminator.predict_proba(X_try)
                    cls_idx_arr = np.where(discriminator.classes_ == class_sample)[0]
                    if cls_idx_arr.size == 0:
                        # Edge case: discriminator has not seen this class label
                        idx_keep = np.arange(len(X_try))
                    else:
                        cls_idx = cls_idx_arr[0]
                        p_min = prob[:, cls_idx]
                        thr = thr_start + (thr_end - thr_start) * (it / max(1, num_iters - 1))
                        idx_keep = np.where(p_min >= thr)[0]
                        if idx_keep.size == 0:
                            # Fallback: take the top few if none pass the threshold
                            topk = max(1, min(len(p_min), trial // 4 if trial >= 4 else 1))
                            idx_keep = np.argsort(-p_min)[:topk]

                    X_kept = X_try[idx_keep]
                    y_kept = y_try[idx_keep]
                    accept_count[distance_func] += len(idx_keep)

                    if len(X_kept) > 0:
                        X_new_parts.append(X_kept)
                        y_new_parts.append(y_kept)
                        remaining -= len(X_kept)

                        # Optionally grow the working pool to diversify future proposals,
                        # but cap the pool size relative to the original minority size.
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
                                                                   use_barycentre_averaging=self.set_barycentre_averaging, )

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
                                           use_barycentre_averaging=False
                                           ):

        """
        Generate a single synthetic sample using soft distance.
        .
        """
        # shape: (c, l)
        # shape: (c, l)
        new_ts = curr_ts.copy()
        if use_barycentre_averaging:
            if new_ts.ndim == 2:
                new_ts = new_ts.squeeze()
                curr_ts = curr_ts.squeeze()
                nn_ts = nn_ts.squeeze()
                reshape_ts = True
            distance = 'msm'  # Barycentre averaging is only applicable with MSM distance
            max_iter = 5
            centre = new_ts  # Initial centre is the current time series
            n_time_points = new_ts.shape[0]
            alignment = np.zeros(n_time_points)  # Stores the sum of values warped to each point
            num_warps_to = np.zeros(n_time_points)  # Tracks how many times each point is warped to
            for i in range(max_iter):
                for Xi in [curr_ts, nn_ts]:
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
            if reshape_ts:
                new_ts = new_ts.reshape(1, -1)
            return new_ts

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

if __name__ == "__main__":
    ## Example usage
    from local.load_ts_data import X_train, y_train, X_test, y_test

    print(np.unique(y_train, return_counts=True))
    smote = ESMOTE(n_neighbors=5, random_state=1, distance="msm")

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
    smote = ESMOTE(n_neighbors=5, random_state=1, distance="msm")

    X_resampled, y_resampled = smote.fit_transform(X, y)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
