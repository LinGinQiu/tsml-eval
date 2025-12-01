from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.pl_model import LitAutoEncoder  # 现在这个就是你新的 LGD-VAE 的 pl 封装
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.data.dataset import (
    UCRDataset, ZScoreNormalizer, load_ucr_splits
)

import os
import inspect
import numpy as np
from datetime import datetime
import argparse
import yaml

import torch
torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------
# 一个支持点操作的 dict，方便 cfg.xxx 写法
# --------------------------------------------------------
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._convert(v)

    @staticmethod
    def _convert(v):
        if isinstance(v, dict):
            return DotDict(v)
        if isinstance(v, list):
            return [DotDict._convert(x) for x in v]
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

class PrintLossCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        epoch = trainer.current_epoch

        # 取出并打印常用 loss 项
        recon = metrics.get("train/recon_loss")
        kl_g = metrics.get("train/kl_g")
        kl_c = metrics.get("train/kl_c")
        align = metrics.get("train/align_loss")
        cls = metrics.get("train/cls_loss")
        center = metrics.get("train/loss_center")
        disentangle = metrics.get("train/disentangle_loss")
        total = metrics.get("train/loss")

        msg = f"[Epoch {epoch}] Train Loss={float(total):.4f}"
        if recon is not None: msg += f", Recon={float(recon):.4f}"
        if kl_g is not None:  msg += f", KL_g={float(kl_g):.4f}"
        if kl_c is not None:  msg += f", KL_c={float(kl_c):.4f}"
        if align is not None: msg += f", Align={float(align):.4f}"
        if cls is not None:   msg += f", Cls={float(cls):.4f}"
        if center is not None:msg += f", Center={float(center):.4f}"
        if disentangle is not None:msg += f", Disentangle={disentangle:.4f}"

        print(msg)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        epoch = trainer.current_epoch
        val_loss = metrics.get("eval/loss")

        print(f"[Epoch {epoch}] Val Loss={float(val_loss):.4f}")

# ========================================================
# main
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./local_test.yaml", help="Path to YAML config")
    parser.add_argument("--dataset_name", type=str, default="FiftyWords", help="UCR dataset name to override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides YAML)")
    args = parser.parse_args()

    # 1) 读 YAML
    with open(args.cfg) as f:
        raw_cfg = yaml.safe_load(f)

    # 2) CLI 覆盖
    if args.dataset_name is not None:
        raw_cfg.setdefault("data", {})
        raw_cfg["data"]["dataset_name"] = args.dataset_name
    if args.seed is not None:
        raw_cfg.setdefault("experiment", {})
        raw_cfg["experiment"]["seed"] = args.seed

    # 3) 基础路径：每个数据集一个 ckpt 目录
    if "paths" in raw_cfg:
        ckpt_root = raw_cfg["paths"].get("ckpt_dir", "./checkpoints")
        raw_cfg["paths"]["ckpt_dir"] = os.path.join(ckpt_root, raw_cfg["data"]["dataset_name"])
        # callback 那里也要跟着改
        raw_cfg.setdefault("callbacks", {}).setdefault("checkpointing", {})
        raw_cfg["callbacks"]["checkpointing"]["dirpath"] = raw_cfg["paths"]["ckpt_dir"]

    cfg = DotDict(raw_cfg)

    # 4) 设置随机种子
    seed_val = cfg.experiment.seed if "experiment" in cfg and "seed" in cfg.experiment else 42
    pl.seed_everything(seed_val, workers=True)

    # ====================================================
    # 读取 UCR + minority 过滤 + zscore
    # ====================================================
    if getattr(cfg.data, "format", "ucr") != "ucr":
        raise NotImplementedError("Only UCR format is implemented yet.")

    dataset_name = cfg.data.dataset_name
    problem_path = cfg.paths.data_root
    resample_id = getattr(cfg, "seed", 0)
    predefined_resample = getattr(cfg.data, "predefined_resample", False)

    # 4.1 读原始 UCR 划分
    X_tr, y_tr, X_te, y_te = load_ucr_splits(
        problem_path=problem_path,
        dataset_name=dataset_name,
        resample_id=resample_id,
        predefined_resample=predefined_resample,
    )

    # 4.3 z-score（fit on train，然后落盘 npz）
    normalizer = ZScoreNormalizer().fit(X_tr)
    stats_dir = os.path.join(cfg.paths.work_root, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    np.savez(os.path.join(stats_dir, f"{dataset_name}_zscore.npz"),
             mean=normalizer.mean_, std=normalizer.std_)

    X_tr = normalizer.transform(X_tr)
    X_te = normalizer.transform(X_te)

    # 4.4 构建 Dataset
    train_dataset = UCRDataset(
        data=X_tr,
        labels=y_tr,
        split="train",
        normalizer=None,
        augmentation_ratio=0.0,  # 现在先关掉额外增强
    )

    eval_dataset = UCRDataset(
        data=X_te,
        labels=y_te,
        split="test",
        normalizer=None,
        augmentation_ratio=0.0,
    )
    print(f"{datetime.now()} : Train size: {len(train_dataset)}; Eval size: {len(eval_dataset)}.")

    # ====================================================
    # 动态调模型参数 —— 但现在是 VAE，不是 MAE
    # ====================================================
    train_size = len(train_dataset)
    seq_len = int(train_dataset.data.shape[-1])
    cfg_model = dict(cfg.model)  # 复制一份，避免后面乱改原始的

    # 按样本数做简单分档
    if train_size <= 120:
        cfg_model.setdefault("embed_dim", 64)
        cfg_model.setdefault("enc_depth", 2)
        cfg_model.setdefault("dec_depth", 2)
        cfg_model.setdefault("n_heads", 4)
        cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 150)
    elif train_size <= 520:
        cfg_model.setdefault("embed_dim", 96)
        cfg_model.setdefault("enc_depth", 4)
        cfg_model.setdefault("dec_depth", 4)
        cfg_model.setdefault("n_heads", 6)
        cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 200)
    else:
        cfg_model.setdefault("embed_dim", 128)
        cfg_model.setdefault("enc_depth", 6)
        cfg_model.setdefault("dec_depth", 4)
        cfg_model.setdefault("n_heads", 8)
        cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 300)

    # Lightning pl 的 __init__ 参数过滤
    sig = inspect.signature(LitAutoEncoder.__init__)
    valid_keys = {
        p.name for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    # 补 in_chans / seq_len
    if "in_chans" in valid_keys and "in_chans" not in cfg_model:
        # 我们的 UCRDataset 是 (N, C, T)
        cfg_model["in_chans"] = int(train_dataset.data.shape[1])
    if "seq_len" in valid_keys and "seq_len" not in cfg_model:
        cfg_model["seq_len"] = seq_len

    # 只保留 pl_model 需要的字段
    model_kwargs = {k: v for k, v in cfg_model.items() if k in valid_keys}

    # 构建 pl_model
    autoencoder = LitAutoEncoder(**model_kwargs, weights=train_dataset.class_freq)
    # 挂上完整 cfg，方便 pl_model 里读 optim/trainer 配置
    autoencoder.cfg = cfg

    # ====================================================
    # DataLoader
    # ====================================================
    if train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset.sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.train_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.data.loader_workers,
        )
    else:
        # 退化情况（比如你不小心 split 不是 "train"）
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.train_batch_size,
            shuffle=True,
            num_workers=cfg.data.loader_workers,
        )

    # def check_loader_distribution(loader, max_batches=200):
    #     from collections import Counter
    #     label_counter = Counter()
    #     n = 0
    #     for b_idx, batch in enumerate(loader):
    #         if b_idx >= max_batches:
    #             break
    #         x, y = batch
    #         y = y.view(-1).cpu().tolist()
    #         label_counter.update(y)
    #         n += len(y)
    #     print("Total samples:", n)
    #     print("Label counts:", label_counter)
    #     for label, cnt in label_counter.items():
    #         print(f"label {label}: {cnt / n:.4f}")
    #
    # check_loader_distribution(train_loader, max_batches=200)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.loader_workers,
    )

    # ====================================================
    # Callbacks & Trainer
    # ====================================================
    callbacks = []
    if "callbacks" in cfg and "checkpointing" in cfg.callbacks:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.checkpointing))
    # if "callbacks" in cfg and "early_stopping" in cfg.callbacks:
    #     callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))



    callbacks.append(PrintLossCallback())

    logger = CSVLogger(
        save_dir=cfg.paths.logs_dir,  # 最外层目录
        name=dataset_name,  # 子目录名称
        version=cfg.experiment.seed  # 可选，版本号
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        **(cfg.trainer if "trainer" in cfg else {}),
    )



    trainer.fit(
        autoencoder,
        train_loader,
        eval_loader,
        ckpt_path=(cfg.ckpt_path if hasattr(cfg, "ckpt_path") else None),
    )


if __name__ == "__main__":
    main()
