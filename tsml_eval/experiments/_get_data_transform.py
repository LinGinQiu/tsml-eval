"""get data transformer function."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.transformations.collection import Normalizer

from tsml_eval.utils.functions import str_in_nested_list

scaling_transformers = [
    ["normalizer", "normaliser"],
]
unbalanced_transformers = [
    "smote",
    "adasyn",
    "tsmote",
    "ohit",
    "esmote",
    "fbsmote",
    "hssmote",
    "hw",
    "state",
    "dtw",
    "soft_dtw_proj",
    "soft_dtw",
    "msm",
    "lgd_l",
    "lgd_p",
    "lgd_m",
    "lgd_lp",
    "lgd_mlp",
    "mgvae",
    "ibgan",
    "timevae",
    "timegan",
    "soft_msm",
    "adtw",
    "soft_adtw",
    "twe",
    "soft_twe",
    "cfam",
    "cdsmote",
    "eval",
]
unequal_transformers = [
    ["padder", "zero-padder"],
    "mean-padder",
    "zero-noise-padder",
    "zero-noise-padder-min",
    "mean-noise-padder",
    ["truncator", "truncate"],
    "truncate-max",
    "resizer",
]


def get_data_transform_by_name(
    transformer_names,
    dataset_name=None,
    row_normalise=False,
    random_state=None,
    n_jobs=1,
):
    """Return a transformers matching a given input name(s).

    Parameters
    ----------
    transformer_names : str or list of str
        String or list of strings indicating the transformer(s) to be returned.
    row_normalise : bool, default=False
        Adds a Normalizer to the front of the transformer list.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the classifier if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both classifier ``fit`` and
        ``predict`` if available. `-1` means using all processors.

    Return
    ------
    transformers : A transformer or list of transformers.
        The transformer(s) matching the input transformer name(s). Returns a list if
        more than one transformer is requested.
    """
    if transformer_names is None and not row_normalise:
        return None

    t_list = []
    if row_normalise:
        t_list.append(Normalizer())

    if transformer_names is not None:
        if not isinstance(transformer_names, list):
            transformer_names = [transformer_names]

        for transformer_name in transformer_names:
            t = transformer_name.casefold()
            if str_in_nested_list(scaling_transformers, t):
                t_list.append(_set_scaling_transformer(t, random_state, n_jobs))
            elif str_in_nested_list(unbalanced_transformers, t):
                t_list.append(
                    _set_unbalanced_transformer(t, dataset_name, random_state, n_jobs)
                )
            elif str_in_nested_list(unequal_transformers, t):
                t_list.append(_set_unequal_transformer(t, random_state, n_jobs))
            else:
                raise ValueError(
                    f"UNKNOWN TRANSFORMER: {t} in get_data_transform_by_name"
                )

    return t_list if len(t_list) > 1 else t_list[0]


def _set_scaling_transformer(t, random_state, n_jobs):
    if t == "normalizer" or t == "normaliser":
        return Normalizer()


def _set_unequal_transformer(t, random_state, n_jobs):
    if t == "padder" or t == "zero-padder":
        from aeon.transformations.collection import Padder

        return Padder()
    elif t == "mean-padder":
        from tsml_eval._wip.unequal_length._pad import Padder

        return Padder(fill_value="mean", random_state=random_state)
    elif t == "zero-noise-padder":
        from tsml_eval._wip.unequal_length._pad import Padder

        return Padder(add_noise=0.001, random_state=random_state)
    elif t == "zero-noise-padder-min":
        from tsml_eval._wip.unequal_length._pad import Padder

        return Padder(
            pad_length="min",
            add_noise=0.001,
            error_on_long=False,
            random_state=random_state,
        )
    elif t == "mean-noise-padder":
        from tsml_eval._wip.unequal_length._pad import Padder

        return Padder(fill_value="mean", add_noise=0.001, random_state=random_state)
    elif t == "truncator" or t == "truncate":
        from tsml_eval._wip.unequal_length._truncate import Truncator

        return Truncator()
    elif t == "truncate-max":
        from tsml_eval._wip.unequal_length._truncate import Truncator

        return Truncator(truncated_length="max", error_on_short=False)
    elif t == "resizer":
        from tsml_eval._wip.unequal_length._resize import Resizer

        return Resizer()


def _set_unbalanced_transformer(t, dataset_name, random_state, n_jobs):
    if t == "smote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._smote import (
            SMOTE,
        )

        return SMOTE(
            n_neighbors=5,
            distance="euclidean",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    elif t == "adasyn":
        from tsml_eval._wip.rt.transformations.collection.imbalance._adasyn import (
            ADASYN,
        )

        return ADASYN(
            n_neighbors=5,
            distance="euclidean",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    elif t == "tsmote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._tsmote import (
            TSMOTE,
        )

        return TSMOTE(
            random_state=random_state,
            spy_size=0.15,
            window_size=None,
            distance="euclidean",
            distance_params=None,
        )
    elif t == "ohit":
        from tsml_eval._wip.rt.transformations.collection.imbalance._ohit import (
            OHIT,
        )

        return OHIT(distance="euclidean", random_state=random_state)
    elif t == "lgd_l":
        from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import (
            VOTE,
        )

        return VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="latent",
            random_state=random_state,
        )
    elif t == "lgd_p":
        from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import (
            VOTE,
        )

        return VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="pair",
            random_state=random_state,
        )
    elif t == "lgd_m":
        from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import (
            VOTE,
        )

        return VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="mjp",
            random_state=random_state,
        )
    elif t == "lgd_lp":
        from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import (
            VOTE,
        )

        return VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="lp",
            random_state=random_state,
        )
    elif t == "lgd_mlp":
        from tsml_eval._wip.rt.transformations.collection.imbalance._lgdvae import (
            VOTE,
        )

        return VOTE(
            n_jobs=n_jobs,
            dataset_name=dataset_name,
            mode="mlp",
            random_state=random_state,
        )
    elif t == "mgvae":
        from tsml_eval._wip.rt.transformations.collection.imbalance._mgvae import (
            MGVAE,
        )

        return MGVAE(
            epochs_pre=200,
            epochs_fine=100,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif t == "timevae":
        from tsml_eval._wip.rt.transformations.collection.imbalance._timevae import (
            TimeVAE,
        )

        return TimeVAE(
            dataset_name=dataset_name,
            random_state=random_state,
        )
    elif t == "timegan":
        from tsml_eval._wip.rt.transformations.collection.imbalance._timegan import (
            TimeGAN,
        )

        return TimeGAN(
            dataset_name=dataset_name,
            module="gru",
            hidden_dim=24,
            num_layer=3,
            iteration=5000,
            batch_size=32,
            random_state=random_state,
        )
    elif t == "hssmote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._hssmote import (
            HS_SMOTE,
        )

        return HS_SMOTE(
            n_neighbors=5,
            random_state=random_state,
        )
    elif t == "esmote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="msm",
            distance_params=None,
            set_barycentre_averaging=True,
            use_interpolated_path=False,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif t == "dtw":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="dtw",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif t == "soft_dtw_proj":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="dtw",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
            use_soft_distance=True,
            use_project=True,
            use_multi_soft_distance=False,
        )
    elif t == "soft_dtw":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="dtw",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
            use_soft_distance=True,
            use_multi_soft_distance=False,
        )
    elif t == "msm":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="msm",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif t == "soft_msm":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="msm",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
            use_soft_distance=True,
            use_multi_soft_distance=True,
        )
    elif t == "adtw":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="adtw",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif t == "soft_adtw":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="adtw",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
            use_soft_distance=True,
            use_multi_soft_distance=True,
        )
    elif t == "twe":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="twe",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif t == "ibgan":
        from tsml_eval._wip.rt.transformations.collection.imbalance._ibgan import (
            IBGANAugmenter,
        )

        return IBGANAugmenter(
            n_jobs=n_jobs,
            random_state=random_state,
        )

    elif t == "soft_twe":
        from tsml_eval._wip.rt.transformations.collection.imbalance._esmote import (
            ESMOTE,
        )

        return ESMOTE(
            n_neighbors=5,
            distance="twe",
            distance_params=None,
            weights="uniform",
            n_jobs=n_jobs,
            random_state=random_state,
            use_soft_distance=True,
            use_multi_soft_distance=True,
        )
    elif t == "state":
        from tsml_eval._wip.rt.transformations.collection.imbalance._stlor import (
            STLOversampler,
        )

        return STLOversampler(
            noise_scale=0.05,
            block_bootstrap=True,
            use_boxcox=True,
            random_state=random_state,
            period_estimation_method="acf",
        )

    elif t == "fbsmote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._fbsmote import (
            FrequencyBinSMOTE,
        )

        return FrequencyBinSMOTE(
            n_neighbors=3,
            top_k=6,
            freq_match_delta=2,
            bandwidth=1,
            random_state=random_state,
            normalize_energy=True,
            enable_selection=False,
        )
    elif t == "hw" or t == "hybrid-wrapper":
        from tsml_eval._wip.rt.transformations.collection.imbalance._hwrapper import (
            HybridWrapper,
        )

        return HybridWrapper(
            random_state=random_state,
            enable_selection=True,
            n_jobs=n_jobs,
        )
    elif t == "cfam":
        from tsml_eval._wip.rt.transformations.collection.imbalance._cfam import CFAM

        return CFAM(
            random_state=random_state,
        )
    elif t == "cdsmote":
        from tsml_eval._wip.rt.transformations.collection.imbalance._cdsmote import (
            CDSMOTE,
        )

        return CDSMOTE(
            n_clusters=2,
            k_neighbors=5,
            random_state=random_state,
        )

    elif t == "eval":
        from tsml_eval._wip.rt.transformations.collection.imbalance._rebalance_eval import (
            rebalance_eval,
        )

        return rebalance_eval(
            dataset_name=dataset_name,
            n_jobs=n_jobs,
            random_state=random_state,
        )
