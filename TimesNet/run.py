import argparse
import os
import random

import numpy as np
import torch
import torch.backends

from TimesNet.exp.exp_classification import Exp_Classification
from TimesNet.utils.print_args import print_args

if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    import sys

    if len(sys.argv) == 1:
        sys.argv += [
            "/Users/qiuchuanhang/Desktop/Exp_tsml/AALTD2025imbalance-main/UCR_Imbalanced_9_1",
            "/Users/qiuchuanhang/PycharmProjects/Time-Series-Library-main/local/results",
            "TimesNet",
            "DistalPhalanxOutlineAgeGroup",
            "0",
            "-dtn",
            "lgd_lp",
            "--transform_train_only",
            "--task_name",
            "classification",
            "--is_training",
            "1",
            "--e_layers",
            "2",
            "--batch_size",
            "8",
            "--d_model",
            "16",
            "--d_ff",
            "32",
            "--top_k",
            "3",
            "--des",
            "Exp",
            "--itr",
            "1",
            "--learning_rate",
            "0.001",
            "--train_epochs",
            "20",
            "--patience",
            "5",
            "--gpu_type",
            "mps",
        ]
    parser = argparse.ArgumentParser(description="classification")

    # basic config
    parser.add_argument(
        "problem_path", type=str, help="Location of problem files, full path."
    )
    parser.add_argument(
        "results_path", type=str, help="Location of where to write results."
    )
    parser.add_argument(
        "classifier_name",
        type=str,
        help="Name of classifier used in writing results. This value will also be assigned to args.model",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help='Name of problem. Files must follow <problem_path>/<dataset>/<dataset>+"_TRAIN.ts" structure.',
    )
    parser.add_argument(
        "resample_id",
        type=int,
        help="Seed for resampling. If set to 0, use default train/test split.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="classification",
        help="task name, options:[classification]",
    )
    parser.add_argument("--is_training", type=int, default=1, help="status")

    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing result files.",
    )
    parser.add_argument(
        "-tr",
        "--build_train_file",
        action="store_true",
        help="Generate train files with cross-validation results.",
    )
    parser.add_argument(
        "-bt",
        "--benchmark_time",
        action="store_true",
        default=True,
        help="Benchmark hardware and record timing.",
    )
    parser.add_argument(
        "-pr",
        "--predefined_resample",
        action="store_true",
        help="Use predefined resample from file.",
    )
    parser.add_argument(
        "-dtn",
        "--data_transform_name",
        action="append",
        default=None,
        help="str to pass to get_data_transform_by_name to apply a transformation "
        "to the data prior to running the experiment. By default no transform "
        "is applied. Can be used multiple times (default: %(default)s).",
    )
    parser.add_argument(
        "-tto",
        "--transform_train_only",
        action="store_true",
        help="if set, transformations will be applied only to the training dataset, "
        "leaving the test dataset unchanged (default: %(default)s).",
    )
    parser.add_argument(
        "-rn",
        "--row_normalise",
        action="store_true",
        help="normalise the data rows prior to fitting and predicting. "
        "effectively the same as passing Normalizer to --data_transform_name "
        "(default: %(default)s).",
    )
    # data loader
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%%)"
    )

    # model define
    parser.add_argument(
        "--expand", type=int, default=2, help="expansion factor for Mamba"
    )
    parser.add_argument(
        "--d_conv", type=int, default=4, help="conv kernel size for Mamba"
    )
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=1, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument(
        "--use_norm",
        type=int,
        default=1,
        help="whether to use normalize; True 1 False 0",
    )
    parser.add_argument(
        "--down_sampling_layers",
        type=int,
        default=0,
        help="num of down sampling layers",
    )
    parser.add_argument(
        "--down_sampling_window", type=int, default=1, help="down sampling window size"
    )
    parser.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        help="down sampling method, only support avg, max, conv",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=96,
        help="the length of segmen-wise iteration of SegRNN",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--gpu_type", type=str, default="cuda", help="gpu type"
    )  # cuda or mps
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    # metrics (dtw)
    parser.add_argument(
        "--use_dtw",
        type=bool,
        default=False,
        help="the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)",
    )

    # Augmentation
    parser.add_argument(
        "--augmentation_ratio", type=int, default=0, help="How many times to augment"
    )
    parser.add_argument("--seed", type=int, default=2, help="Randomization seed")
    parser.add_argument(
        "--jitter",
        default=False,
        action="store_true",
        help="Jitter preset augmentation",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
        help="Scaling preset augmentation",
    )
    parser.add_argument(
        "--permutation",
        default=False,
        action="store_true",
        help="Equal Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--randompermutation",
        default=False,
        action="store_true",
        help="Random Length Permutation preset augmentation",
    )
    parser.add_argument(
        "--magwarp",
        default=False,
        action="store_true",
        help="Magnitude warp preset augmentation",
    )
    parser.add_argument(
        "--timewarp",
        default=False,
        action="store_true",
        help="Time warp preset augmentation",
    )
    parser.add_argument(
        "--windowslice",
        default=False,
        action="store_true",
        help="Window slice preset augmentation",
    )
    parser.add_argument(
        "--windowwarp",
        default=False,
        action="store_true",
        help="Window warp preset augmentation",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
        help="Rotation preset augmentation",
    )
    parser.add_argument(
        "--spawner",
        default=False,
        action="store_true",
        help="SPAWNER preset augmentation",
    )
    parser.add_argument(
        "--dtwwarp",
        default=False,
        action="store_true",
        help="DTW warp preset augmentation",
    )
    parser.add_argument(
        "--shapedtwwarp",
        default=False,
        action="store_true",
        help="Shape DTW warp preset augmentation",
    )
    parser.add_argument(
        "--wdba",
        default=False,
        action="store_true",
        help="Weighted DBA preset augmentation",
    )
    parser.add_argument(
        "--discdtw",
        default=False,
        action="store_true",
        help="Discrimitive DTW warp preset augmentation",
    )
    parser.add_argument(
        "--discsdtw",
        default=False,
        action="store_true",
        help="Discrimitive shapeDTW warp preset augmentation",
    )
    parser.add_argument("--extra_tag", type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")

    args = parser.parse_args()
    # Set additional argument relationships:
    # args.data is assigned from args.dataset
    # args.model_id is also assigned from args.resample_id
    # args.model is assigned from args.classifier_name for convenience in downstream use
    args.data = args.dataset
    args.model_id = args.resample_id
    args.model = args.classifier_name

    print("torch device setting")
    if args.use_gpu and torch.cuda.is_available() and args.gpu_type == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            args.use_multi_gpu = True
            args.device_ids = list(range(available_gpus))
            args.devices = ",".join(str(i) for i in args.device_ids)
            args.gpu = args.device_ids[0]
            args.device = torch.device(f"cuda:{args.gpu}")
            print(f"Detected {available_gpus} GPUs; using multi-GPU mode.")
        else:
            args.use_multi_gpu = False
            args.device_ids = [args.gpu]
            args.devices = str(args.gpu)
            args.device = torch.device(f"cuda:{args.gpu}")
            print("Single GPU detected; using single-GPU mode.")
    elif (
        args.use_gpu
        and args.gpu_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        args.device = torch.device("mps")
        print("Using MPS device")
    else:
        args.use_gpu = False
        args.device = torch.device("cpu")
        print("Using CPU device")

    print("Args in experiment:")
    print_args(args)

    if args.task_name == "classification":
        Exp = Exp_Classification
    else:
        raise NotImplementedError
    if args.data_transform_name:
        print("Data transform name:", args.data_transform_name)
        if args.transform_train_only:
            print("Transform train only:", args.transform_train_only)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        if args.gpu_type == "mps":
            pass
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()
