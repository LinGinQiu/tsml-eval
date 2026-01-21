# src/pipeline/lgd_vae_pipeline.py

from __future__ import annotations

import os
import inspect
from datetime import datetime
from typing import Optional, Tuple

import yaml
import numpy as np

import torch
torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.pl_model import LitAutoEncoder
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.data.dataset import (
    UCRDataset,
    ZScoreNormalizer,
    load_ucr_splits,
)
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.inference import Inference  # 如果不在同一级，根据你的项目结构调整路径


# --------------------------------------------------------
# DotDict: 支持 cfg.xxx 点操作
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


import torch
from torch.utils.data import Sampler
import random


class SwitchableWeightedSampler(Sampler):
    def __init__(self, full_weights, majority_indices, switch_epoch, num_samples=None, replacement=True):
        """
        Args:
            full_weights: 对应 dataset 中每个样本的权重 (用于阶段二)
            majority_indices: majority 样本的 index 列表 (用于阶段一)
            switch_epoch: 第几个 epoch 开始切换到全样本加权采样
            num_samples: 加权采样时的采样数量 (通常等于 dataset 长度)
            replacement: 加权采样时是否放回 (通常为 True)
        """
        super().__init__()
        self.full_weights = torch.as_tensor(full_weights, dtype=torch.double)
        self.majority_indices = majority_indices
        self.switch_epoch = switch_epoch
        self.replacement = replacement

        # 如果没有指定 num_samples，默认等于全量数据长度
        self.num_samples = len(self.full_weights) if num_samples is None else num_samples

        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        if self.current_epoch < self.switch_epoch:
            # === 阶段一：只训练 Majority ===
            # 这里通常不需要加权，只需要随机打乱 Majority 的数据
            # 如果你希望 Majority 内部也加权，逻辑会更复杂，但通常不需要
            indices = self.majority_indices[:]
            random.shuffle(indices)

            # 返回迭代器
            return iter(indices)

        else:
            # === 阶段二：全样本加权采样 (模拟 WeightedRandomSampler) ===
            # PyTorch WeightedRandomSampler 的底层核心就是 torch.multinomial
            rand_tensor = torch.multinomial(
                self.full_weights,
                self.num_samples,
                self.replacement
            )
            return iter(rand_tensor.tolist())

    def __len__(self):
        # 动态长度：根据阶段返回不同的长度
        if self.current_epoch < self.switch_epoch:
            return len(self.majority_indices)
        else:
            return self.num_samples

# --------------------------------------------------------
# 回调：打印各项 loss
# --------------------------------------------------------
class PrintLossCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        recon = metrics.get("train/recon_loss")
        kl_g = metrics.get("train/kl_g")
        kl_c = metrics.get("train/kl_c")
        align = metrics.get("train/align_loss")
        cls = metrics.get("train/cls_loss")
        center = metrics.get("train/loss_center")
        disentangle = metrics.get("train/disentangle_loss")
        total = metrics.get("train/loss")

        if total is None:
            return

        msg = f"[Epoch {epoch}] Train Loss={float(total):.4f}"
        if recon is not None:       msg += f", Recon={float(recon):.4f}"
        if kl_g is not None:        msg += f", KL_g={float(kl_g):.4f}"
        if kl_c is not None:        msg += f", KL_c={float(kl_c):.4f}"
        if align is not None:       msg += f", Align={float(align):.4f}"
        if cls is not None:         msg += f", Cls={float(cls):.4f}"
        if center is not None:      msg += f", Center={float(center):.4f}"
        if disentangle is not None: msg += f", Disentangle={float(disentangle):.4f}"

        print(msg)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get("eval/loss")
        recon_loss = metrics.get("eval/recon_loss")
        if val_loss is not None:
            print(f"[Epoch {epoch}] Val Loss={float(val_loss):.4f}"
                  f", Recon={float(recon_loss):.4f}")


# --------------------------------------------------------
# EarlyStopping with warmup: DelayedEarlyStopping
# --------------------------------------------------------
class DelayedEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores validation checks for the first `warmup_epochs`.

    It subclasses Lightning's EarlyStopping and simply skips calling the parent's
    validation hooks until the trainer has reached `warmup_epochs`.

    Configure the warmup period via the callback config key `warmup_epochs` (int).
    """
    def __init__(self, warmup_epochs: int = 10, *args, **kwargs):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(*args, **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        # trainer.current_epoch is 0-based; skip checks while < warmup
        if trainer.current_epoch < self.warmup_epochs:
            return
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return
        return super().on_validation_end(trainer, pl_module)

from lightning.pytorch.callbacks import ModelCheckpoint

class DelayedModelCheckpoint(ModelCheckpoint):
    def __init__(self, warmup_epochs: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_save_epoch = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        # 如果当前 epoch 小于设定的起始 epoch，直接跳过保存逻辑
        if trainer.current_epoch < self.start_save_epoch:
            return
        # 否则执行父类的正常保存逻辑
        super().on_validation_end(trainer, pl_module)

# ========================================================
# LGD-VAE 端到端 Pipeline: __init__ + fit + transform
# ========================================================
class LGDVAEPipeline:
    """
    一个简单的 end-to-end 管线：
      - __init__ 负责加载/整理 YAML 配置（相当于原来 train.py 的 argparse + cfg 部分）
      - fit(...) 负责训练（相当于原来 main() 里大部分代码）
      - transform(...) 统一调用 Inference 的生成接口
    """

    def __init__(
        self,
        dataset_name: str = "FiftyWords",
        seed: int | None = None,
        device: torch.device | None = None,
    ):
        """
        dataset_name: 覆盖 YAML 里的 data.dataset_name
        seed: 覆盖 YAML 中 experiment.seed
        device: 手动指定训练/推断设备（不指定就交给 Lightning 和 Inference 自己处理）
        """
        # 1) 读 YAML
        import socket
        hostname = socket.gethostname()
        is_iridis = "iridis" in hostname.lower() or "loginx" in hostname.lower()
        is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名
        self.is_iridis = is_iridis
        if is_iridis:
            print("[ENV] Detected Iridis HPC environment")
            cfg_path = "/home/cq2u24/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/LGD_VAE/config.yaml"
        elif is_mac:
            print("[ENV] Detected local macOS environment")
            cfg_path = ("/Users/qiuchuanhang/PycharmProjects/tsml-eval/tsml_eval/_wip/rt/transformations"
                        "/collection/imbalance/LGD_VAE/local_test.yaml")
        else:
            print(f"[ENV] Unknown environment {hostname}, fallback to iridis")
            cfg_path = "/home/cq2u24/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/LGD_VAE/config.yaml"

        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f)

        # 2) 覆盖 data.dataset_name & experiment.seed
        raw_cfg.setdefault("data", {})
        raw_cfg["data"]["dataset_name"] = dataset_name
        print(dataset_name)
        if seed is not None:
            raw_cfg.setdefault("experiment", {})
            raw_cfg["experiment"]["seed"] = seed

        # 3) 设置每个数据集专属的 ckpt 目录
        if "paths" in raw_cfg:
            ckpt_root = raw_cfg["paths"].get("ckpt_dir", "./checkpoints")
            raw_cfg["paths"]["ckpt_dir"] = os.path.join(ckpt_root, dataset_name+str(seed))

            # callbacks 中 checkpointing.dirpath 也同步
            raw_cfg.setdefault("callbacks", {}).setdefault("checkpointing", {})
            raw_cfg["callbacks"]["checkpointing"]["dirpath"] = raw_cfg["paths"]["ckpt_dir"]

        # 转为 DotDict，方便 cfg.xxx 的写法
        self.cfg = DotDict(raw_cfg)
        self.cfg_path = cfg_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.mean_ = None
        self.std_ = None

        # 4) 设置随机种子
        seed_val = seed
        pl.seed_everything(seed_val, workers=True)

        # 设备记录一下（Trainer 会自己处理，Inference 也会用到）
        self.device = device

        # 训练好后的对象
        self.trainer: pl.Trainer | None = None
        self.model: LitAutoEncoder | None = None
        self.infer: Inference | None = None

    # -------------------------
    # 内部工具：构建 DataLoader
    # -------------------------
    def _build_dataloaders(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray | None = None,
        y_te: np.ndarray | None = None,
    ) -> tuple[UCRDataset, UCRDataset, DataLoader, DataLoader]:
        cfg = self.cfg

        # 构建 Dataset（注意：此时 data 已经是 zscore 后的，所以 normalizer=None）
        _,C,L = X_tr.shape
        train_dataset = UCRDataset(
            data=X_tr,
            labels=y_tr,
            split="train",
            normalizer=None,
            augmentation_ratio=0.0,
        )
        if X_te is not None:
            eval_dataset = UCRDataset(
                data=X_te,
                labels=y_te,
                split="test",
                normalizer=None,
                augmentation_ratio=0.0,
            )
        else:
            eval_dataset = None
        print(
            f"{datetime.now()} : Train size: {len(train_dataset)}; "
            f"Eval size: {len(eval_dataset)if eval_dataset else None}."
            f"Series Channel and length: {C}, {L}"
        )

        if getattr(train_dataset, "sample_weights", None) is not None:
            print("Use WeightedRandomSampler for training data.")
            # print("Use SwitchableWeightedSampler for latent distillation training.")
            sampler = WeightedRandomSampler(
                weights=train_dataset.sample_weights,
                num_samples=len(train_dataset.sample_weights),
                replacement=True,
            )
            # sampler = SwitchableWeightedSampler(
            #     full_weights=train_dataset.sample_weights,
            #     majority_indices=train_dataset.majority_indices,
            #     switch_epoch=1)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.data.train_batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=cfg.data.loader_workers,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.data.train_batch_size,
                shuffle=True,
                num_workers=cfg.data.loader_workers,
            )
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=cfg.data.eval_batch_size,
                shuffle=False,
                num_workers=cfg.data.loader_workers,
            )
        else:
            eval_loader = None
        return train_dataset, eval_dataset, train_loader, eval_loader

    # -------------------------
    # 内部工具：构建 pl_model + Trainer
    # -------------------------
    def _build_model_and_trainer(self, train_dataset: UCRDataset) -> tuple[LitAutoEncoder, pl.Trainer]:
        cfg = self.cfg

        train_size = len(train_dataset)
        seq_len = int(train_dataset.data.shape[-1])
        cfg_model = dict(cfg.model)  # 拷一份，避免修改原配置

        # 简单按样本数调一调模型尺寸（跟你原来的逻辑一致）
        if self.is_iridis:
            if train_size <= 120:
                cfg_model.setdefault("embed_dim", 64)
                cfg_model.setdefault("enc_depth", 2)
                cfg_model.setdefault("dec_depth", 2)
                cfg_model.setdefault("n_heads", 4)
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 50)
            elif train_size > 520 and seq_len > 1000:
                cfg_model.setdefault("embed_dim", 128)
                cfg_model.setdefault("enc_depth", 4)
                cfg_model.setdefault("dec_depth", 4)
                cfg_model.setdefault("n_heads", 8)
                cfg_model.setdefault("latent_dim_global", 64)
                cfg_model.setdefault("latent_dim_class", 64)
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 200)
            else:
                cfg_model.setdefault("embed_dim", 96)
                cfg_model.setdefault("enc_depth", 4)
                cfg_model.setdefault("dec_depth", 4)
                cfg_model.setdefault("n_heads", 6)
                cfg_model.setdefault("latent_dim_global", 48)
                cfg_model.setdefault("latent_dim_class", 48)
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 100)

        # LightningModule 的 __init__ 参数过滤（防止多余键报错）
        sig = inspect.signature(LitAutoEncoder.__init__)
        valid_keys = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        # 补充 in_chans / seq_len
        if "in_chans" in valid_keys and "in_chans" not in cfg_model:
            cfg_model["in_chans"] = int(train_dataset.data.shape[1])
        if "seq_len" in valid_keys and "seq_len" not in cfg_model:
            cfg_model["seq_len"] = seq_len

        # 只保留 pl_model 需要的字段
        model_kwargs = {k: v for k, v in cfg_model.items() if k in valid_keys}

        # 构建 pl_model，注意把 class_freq 传进去
        autoencoder = LitAutoEncoder(
            **model_kwargs,
            weights=getattr(train_dataset, "class_freq", None),
        )
        # 把完整 cfg 挂上，方便 pl_model 里读取 optim/trainer 配置
        autoencoder.cfg = cfg

        # callbacks
        callbacks = []
        if "callbacks" in cfg and "checkpointing" in cfg.callbacks:
            callbacks.append(DelayedModelCheckpoint(**cfg.callbacks.checkpointing))

        if "callbacks" in cfg and "early_stopping" in cfg.callbacks:
            # allow an optional `warmup_epochs` key in the early_stopping config
            # so we can ignore early-stopping checks for the initial training epochs
            es_cfg = dict(cfg.callbacks.early_stopping)
            callbacks.append(DelayedEarlyStopping(**es_cfg))
        callbacks.append(PrintLossCallback())
        import time
        run_suffix = str(int(time.time()))
        # logger: 每个 dataset_name 一个子目录，版本用 seed
        logger = CSVLogger(
            save_dir=cfg.paths.logs_dir,
            name=self.dataset_name+str(self.seed),
            version=f'{cfg.experiment.seed}_{run_suffix}',
        )

        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=False,
            **(cfg.trainer if "trainer" in cfg else {}),
        )

        return autoencoder, trainer

    # -------------------------
    # 公共接口：fit
    # -------------------------
    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray | None = None,
        y_te: np.ndarray | None = None,
    ) -> LGDVAEPipeline:
        """
        训练 LGD-VAE.

        如果不传 X_tr/y_tr/X_te/y_te，则会自动按照 cfg 从 UCR 目录读取；
        如果你已经在外面自己做好了 split，就可以把 numpy 数组直接传进来。
        """
        cfg = self.cfg
        dataset_name = cfg.data.dataset_name
        # 1) 准备数据
        if  X_te is None or y_te is None:
            # 只支持 UCR 格式（保持和你的原脚本一致）
            if getattr(cfg.data, "format", "ucr") != "ucr":
                raise NotImplementedError("Only UCR format is implemented yet.")

            problem_path = cfg.paths.data_root
            resample_id = self.seed
            predefined_resample = getattr(cfg.data, "predefined_resample", False)
            print(f'random id in pipline is {resample_id}')
            # X_tr_, y_tr_, X_te_, y_te_ = load_ucr_splits(
            #     problem_path=problem_path,
            #     dataset_name=dataset_name,
            #     resample_id=resample_id,
            #     predefined_resample=predefined_resample,
            # )
            # assert np.array_equal(X_tr, X_tr_), "Train data mismatch!"
            # assert np.array_equal(y_tr, y_tr_), "Train labels mismatch!"

            # X_te, y_te = X_te_, y_te_
            X_te, y_te = X_tr, y_tr
        # apply z-score
        normalizer = ZScoreNormalizer().fit(X_tr)
        stats_dir = os.path.join(cfg.paths.work_root, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        np.savez(os.path.join(stats_dir, f"{dataset_name}_zscore.npz"),
                 mean=normalizer.mean_, std=normalizer.std_)
        self.mean_ = normalizer.mean_
        self.std_ = normalizer.std_
        print(f"dataset mean: {normalizer.mean_} and std: {normalizer.std_}")
        X_tr = normalizer.transform(X_tr)
        X_te = normalizer.transform(X_te)
        # 2) DataLoader + Dataset
        train_dataset, eval_dataset, train_loader, eval_loader = self._build_dataloaders(
            X_tr, y_tr, X_te, y_te
        )


        # 3) 构建模型 & Trainer
        autoencoder, trainer = self._build_model_and_trainer(train_dataset)

        # 4) 训练：如果已有 checkpoint 则优先尝试加载；加载失败再回退到重新训练
        ckpt_path = None
        ckpt_dir = getattr(self.cfg.paths, "ckpt_dir", None)
        if ckpt_dir is not None and os.path.isdir(ckpt_dir):
            print(f"loading checkpoint from {ckpt_dir}")
            ckpt_candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpt_candidates:
                import re
                def extract_epoch(name: str):
                    """ 解析文件名中的 epoch 数字，例如 'lgd-vae-epoch=50.ckpt' → 50 """
                    match = re.search(r"epoch=(\d+)", name)
                    return int(match.group(1)) if match else -1

                epoch_ckpts = [f for f in ckpt_candidates if "epoch=" in f]
                if epoch_ckpts:
                    ckpt_file = max(epoch_ckpts, key=extract_epoch)
                    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                else:
                    # 其次：fallback 用 last.ckpt（如果存在）
                    last_ckpts = [f for f in ckpt_candidates if "last" in f]
                    if last_ckpts:
                        ckpt_path = os.path.join(ckpt_dir, last_ckpts[0])
                    else:
                        # 最后：随便选一个（通常不会到这里）
                        ckpt_path = os.path.join(ckpt_dir, sorted(ckpt_candidates)[0])

                try:
                    print(f"[LGDVAEPipeline] Auto-loading checkpoint: {ckpt_path}")
                    from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.pl_model import LitAutoEncoder
                    self.infer = Inference.from_checkpoint(
                        ckpt_path,
                        model_class=LitAutoEncoder,
                        model_kwargs=None,  # 让它从 hyper_parameters 里自动捞
                        device=self.device,
                        strict=False,
                    )
                    print("load successfully!")
                    if self.mean_ and self.std_:
                        print(f"[LGDVAEPipeline] Loading mean and std: mean: {self.mean_}, std: {self.std_}")
                        self.infer.load_zscore_values(mean=self.mean_, std=self.std_)
                    return self
                except Exception as e:
                    print(f"[LGDVAEPipeline] Failed to load checkpoint ({e}), fallback to training from scratch.")
                    ckpt_path = None  # 回退到正常训练

        if eval_dataset is not None:
            trainer.fit(
                autoencoder,
                train_loader,
                eval_loader,
                ckpt_path=ckpt_path,
            )
        else:
            trainer.fit(
                autoencoder,
                train_loader,
                ckpt_path=ckpt_path,
            )

        # 5) 保存到 pipeline 成员里
        self.trainer = trainer
        self.model = autoencoder
        print("training finished! exiting here and storage the model...")

        import sys
        sys.exit(0)
        # # 训练完直接建一个 Inference wrapper（不从 ckpt 读，直接包内存模型）
        # self.infer = Inference(autoencoder, device=self.device)
        # if self.mean_ and self.std_:
        #     print(f"[LGDVAEPipeline] Loading mean and std: ")
        #     self.infer.load_zscore_values(mean=self.mean_, std=self.std_)
        #     print(f"[LGDVAEPipeline] Loaded mean and std: {self.infer.mean_, self.infer.std_}")
        # return self

    # -------------------------
    # 公共接口：transform / 生成
    # -------------------------
    def transform(self, mode: str, **kwargs) -> torch.Tensor:
        """
        统一的生成接口，内部直接调用 Inference 的方法。

        mode:
          - "vae_prior"      → 先验采样生成（generate_vae_prior）
          - "mix_pair"       → minority + majority pair 门控混合（generate_mix_pair）
          - "smote_latent"   → latent 空间 SMOTE 插值生成（generate_smote_latent）
          - "prototype"      → 基于 prototype 生成（generate_from_prototype）
        """
        if self.infer is None:
            raise RuntimeError("Pipeline is not fitted yet. Call `fit()` first.")

        mode = mode.lower()

        if mode == "prior":
            # kwargs: batch_size=..., device=...
            synthetics = self.infer.generate_vae_prior(**kwargs)
        elif mode == "pair":
            # kwargs: x_min=..., x_maj=..., use_y=...
            synthetics = self.infer.generate_mix_pair(**kwargs)
        elif mode == "latent":
            # kwargs: x_min1=..., x_min2=..., alpha=..., num_samples=...
            synthetics = self.infer.generate_smote_latent(**kwargs)
        elif mode == "prototype":
            # kwargs: x_min=..., use_y=...
            synthetics = self.infer.generate_from_prototype(**kwargs)
        else:
            raise ValueError(f"Unknown transform mode: {mode!r}")

        return synthetics
