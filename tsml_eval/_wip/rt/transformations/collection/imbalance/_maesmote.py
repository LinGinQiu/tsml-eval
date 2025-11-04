from collections import OrderedDict
from typing import Optional, Union
import tqdm
import numpy as np
from numba import prange
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import glob
from tsml_eval._wip.rt.classification.distance_based import KNeighborsTimeSeriesClassifier
from tsml_eval._wip.rt.clustering.averaging._ba_utils import _get_alignment_path
from aeon.transformations.collection import BaseCollectionTransformer

__maintainer__ = ["chrisholder"]
__all__ = ["ESMOTE"]

from tsml_eval._wip.rt.utils._threading import threaded
from typing import Tuple, Dict

import os
import socket
from pathlib import Path
import torch
import sys


def get_best_ckpt_path(ckpt_dir: str | Path) -> str:
    """
    Return the path to the latest non-last checkpoint file in a directory.
    Priority: ti-mae-epoch=*.ckpt > last.ckpt
    """
    pattern = os.path.join(ckpt_dir, "ti-mae-epoch=*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    if ckpts:
        return ckpts[-1]  # 最新的那个
    # fallback
    last_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_path):
        return last_path
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _plot_series_list(series_list, title):
    try:
        fig, ax = plt.subplots()
    except Exception:
        return  # matplotlib may be unavailable in some environments
    for s in series_list:
        s_arr = np.asarray(s)
        if s_arr.ndim == 1:
            ax.plot(s_arr)
        elif s_arr.ndim == 2:
            # shape assumed (C, L); draw each channel
            for c in range(s_arr.shape[0]):
                ax.plot(s_arr[c])
        else:
            ax.plot(s_arr.reshape(-1))
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    fig.tight_layout()
    try:
        plt.show(block=False)
    except Exception:
        plt.show()


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


class MAESMOTE(BaseCollectionTransformer):
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
            use_soft_distance: bool = True,
            dataset_name: str = "",
            use_multi_soft_distance: bool = False,
            visualize: bool = False,
            use_project: bool = True,
            n_jobs: int = 1,
            random_state=None,
            mae_imputation: bool = False,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params
        self.weights = weights
        self.n_jobs = n_jobs
        self.use_soft_distance = use_soft_distance
        self.use_multi_soft_distance = use_multi_soft_distance
        self.use_project = use_project
        self.dataset_name = dataset_name
        self._random_state = None
        self._distance_params = distance_params or {}

        self.mae_imputation = mae_imputation
        self._root = None
        self._ckpt = None
        self._stats = None
        self._stats_path = None
        self._device = None
        self._inference = None
        self._init_model()

        self.visualize = visualize
        self.nn_ = None
        super().__init__()

    def _init_model(self):
        # ---- Environment detection ----
        hostname = socket.gethostname()
        is_iridis = "iridis" in hostname.lower() or "loginx" in hostname.lower()
        is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名

        if is_iridis:
            print("[ENV] Detected Iridis HPC environment")
            self._root = Path("/home/cq2u24/ti-mae-master")
            self._ckpt = Path("/scratch/cq2u24/ti-mae/checkpoints")
            self._stats = Path("/scratch/cq2u24/ti-mae/stats")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif is_mac:
            print("[ENV] Detected local macOS environment")
            self._root = Path("/Users/qiuchuanhang/PycharmProjects/ti-mae-master")
            self._ckpt = self._root / "checkpoints"
            self._stats = self._root / "stats"
            self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            print("[ENV] Unknown environment, fallback to current dir")
            self._root = Path.cwd()
            self._ckpt = self._root / "checkpoints"
            self._stats = self._root / "stats"
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit(self, X, y=None):

        self._random_state = check_random_state(self.random_state)

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        num_minority = min(counts)
        suggested_n_neighbors = int(min(2 + 0.1 * num_minority, self.n_neighbors))
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

        ckpt_dir = self._ckpt / self.dataset_name
        ckpt = get_best_ckpt_path(ckpt_dir)
        self._stats_path = self._stats / f"{self.dataset_name}_zscore.npz"
        # Make sure inference.py can be imported
        sys.path.append(str(self._root))

        from inference import Inference
        from src.nn.pl_model import LitAutoEncoder

        infer = Inference.from_checkpoint(
            ckpt,
            LitAutoEncoder,
            device=self._device,
        )
        self._inference = infer

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
                step = step_size * self._random_state.uniform(low=0, high=1.1)
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
        if self.use_multi_soft_distance:
            reshape_ts = False
            if new_ts.ndim == 2:
                new_ts = new_ts.squeeze()
                curr_ts = curr_ts.squeeze()
                nn_ts = nn_ts.squeeze()
                reshape_ts = True
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

            new_ts = curr_ts + gamma * direction
            if reshape_ts:
                new_ts = new_ts.reshape(1, -1)
            return new_ts

        if self.use_soft_distance:
            new_ts = self._generate_sample_use_soft_distance(
                curr_ts,
                nn_ts,
                distance,
                step,
                return_bias=return_bias,
                visualize=self.visualize,
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

        new_ts = new_ts - bias  # / num_of_alignments
        return new_ts

    def _generate_sample_use_soft_distance(self, curr_ts, nn_ts, distance, step,
                                           return_bias=False, return_weights=False,
                                           visualize: bool = False,
                                           ):

        """
        Generate a single synthetic sample using soft distance.
        .
        """
        # shape: (c, l)
        # shape: (c, l)
        new_ts = curr_ts.copy()
        T = curr_ts.shape[-1]

        # draw curr.ts
        # --- lightweight plotting helpers ---

        if visualize:
            _plot_series_list([curr_ts, nn_ts], title="Anchor (curr_ts) and Neighbor (nn_ts)")
        # reduce the magnitute between curr_ts and nn_ts
        # Match per-channel mean absolute amplitude of nn_ts to curr_ts before soft alignment
        # This keeps the DC offset intact and only rescales magnitudes.
        eps_amp = 1e-12
        if nn_ts.ndim == 2 and curr_ts.ndim == 2:
            # shapes (C, L)
            m_curr = np.mean(np.abs(curr_ts), axis=1, keepdims=True)  # (C, 1)
            m_nn = np.mean(np.abs(nn_ts), axis=1, keepdims=True) + eps_amp  # (C, 1)
            scale = m_curr / m_nn  # (C, 1)
            nn_ts_cp = nn_ts * scale
        else:
            # Fallback: flatten channels jointly if unexpected shape appears
            m_curr = np.mean(np.abs(curr_ts))
            m_nn = np.mean(np.abs(nn_ts)) + eps_amp
            scale = m_curr / m_nn
            nn_ts_cp = nn_ts * scale

        if distance == 'msm':
            from tsml_eval._wip.rt.distances.elastic import soft_msm_gradient
            A, _ = soft_msm_gradient(curr_ts, nn_ts_cp)
        elif distance == 'dtw':
            from tsml_eval._wip.rt.distances.elastic import soft_dtw_gradient
            A, _ = soft_dtw_gradient(curr_ts, nn_ts_cp)
        elif distance == 'twe':
            from tsml_eval._wip.rt.distances.elastic import soft_twe_gradient
            A, _ = soft_twe_gradient(curr_ts, nn_ts_cp)
        elif distance == 'adtw':
            from tsml_eval._wip.rt.distances.elastic import soft_adtw_gradient
            A, _ = soft_adtw_gradient(curr_ts, nn_ts_cp)
        else:
            raise ValueError(f"Soft distance not implemented for distance: {distance}")
        A = np.maximum(A, 0.0)
        row_sum = A.sum(axis=1, keepdims=True) + 1e-12
        W = A / row_sum
        if return_weights:
            return W
        proj = W @ nn_ts.squeeze()
        if self.use_project:
            return proj.reshape(new_ts.shape)
        if visualize:
            # Show projected neighbor on the anchor timeline (compare with curr_ts)
            _plot_series_list([curr_ts[:, :100], proj[:100]], title="Projected neighbor onto anchor timeline")
        # draw projected point
        P = np.clip(W, 1e-12, None)
        ent = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
        conf = 1.0 - ent

        direction = proj - curr_ts
        if return_bias:
            return direction

        # conf_tau = 0.4
        # conf[conf < conf_tau] = 0.0
        # s = np.clip(conf, 0.0, 1.0)
        # gamma = 0.0 + step*s
        new_ts = curr_ts + step * direction
        if visualize and not return_bias and not return_weights:
            _plot_series_list([curr_ts[:, :100], new_ts[:, :100]], title="Soft-ESMOTE synthetic vs anchor")
        if self.mae_imputation:
            mask = torch.zeros(1, T)
            # 前 20% & 后 20%
            num_mask = int(0.2 * T)
            mask[:, :num_mask] = 1  # 前20%
            mask[:, -num_mask:] = 1  # 后20%
            if self._device == torch.device("mps"):
                new_ts = torch.from_numpy(new_ts[np.newaxis, :].astype(np.float32))
            else:
                new_ts = torch.from_numpy(new_ts[np.newaxis, :])
            new_ts = self._inference.impute_from_stats_path(
                x_raw=new_ts,
                mask=mask,
                stats_path=self._stats_path,
            )
            new_ts = new_ts.squeeze(0).cpu().numpy()
        if visualize and not return_bias and not return_weights:
            _plot_series_list([curr_ts[:, :100], new_ts[:, :100]],
                              title="maesmote synthetic vs anchor after mae imputation")
        return new_ts


if __name__ == "__main__":
    smote = MAESMOTE(n_neighbors=5, random_state=1, distance="dtw", use_soft_distance=True, use_project=False,
                     use_multi_soft_distance=False, visualize=True, mae_imputation=True, dataset_name='ACSF1')
    # Example usage
    from local.load_ts_data import X_train, y_train, X_test, y_test

    print(np.unique(y_train, return_counts=True))

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)

    print(np.unique(y_resampled, return_counts=True))
    # stop = ""
    # n_samples = 100  # Total number of labels
    # majority_num = 90  # number of majority class
    # minority_num = n_samples - majority_num  # number of minority class
    # np.random.seed(42)
    #
    # X = np.random.rand(n_samples, 1, 10)
    # y = np.array([0] * majority_num + [1] * minority_num)
    # print(np.unique(y, return_counts=True))
    #
    # X_resampled, y_resampled = smote.fit_transform(X, y)
    # print(X_resampled.shape)
    # print(np.unique(y_resampled, return_counts=True))
