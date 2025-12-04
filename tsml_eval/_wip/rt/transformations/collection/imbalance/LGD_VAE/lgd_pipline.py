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
        if val_loss is not None:
            print(f"[Epoch {epoch}] Val Loss={float(val_loss):.4f}")


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

        # DataLoader，支持 WeightedRandomSampler
        if getattr(train_dataset, "sample_weights", None) is not None:
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
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 150)
            elif train_size > 520 and seq_len > 300:
                cfg_model.setdefault("embed_dim", 128)
                cfg_model.setdefault("enc_depth", 6)
                cfg_model.setdefault("dec_depth", 4)
                cfg_model.setdefault("n_heads", 8)
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 500)
            else:
                cfg_model.setdefault("embed_dim", 96)
                cfg_model.setdefault("enc_depth", 4)
                cfg_model.setdefault("dec_depth", 4)
                cfg_model.setdefault("n_heads", 6)
                cfg.trainer["max_epochs"] = max(int(cfg.trainer.get("max_epochs", 0)), 300)

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
            callbacks.append(ModelCheckpoint(**cfg.callbacks.checkpointing))

        if "callbacks" in cfg and "early_stopping" in cfg.callbacks:
            callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))
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

        # 1) 准备数据
        if  X_te is None or y_te is None:
            # 只支持 UCR 格式（保持和你的原脚本一致）
            if getattr(cfg.data, "format", "ucr") != "ucr":
                raise NotImplementedError("Only UCR format is implemented yet.")

            dataset_name = cfg.data.dataset_name
            problem_path = cfg.paths.data_root
            resample_id = self.seed
            predefined_resample = getattr(cfg.data, "predefined_resample", False)
            print(f'random id in pipline is {resample_id}')
            X_tr_, y_tr_, X_te_, y_te_ = load_ucr_splits(
                problem_path=problem_path,
                dataset_name=dataset_name,
                resample_id=resample_id,
                predefined_resample=predefined_resample,
            )
            assert np.array_equal(X_tr, X_tr_), "Train data mismatch!"
            assert np.array_equal(y_tr, y_tr_), "Train labels mismatch!"

            X_te, y_te = X_te_, y_te_

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
                # 优先 last.ckpt，其次任意一个 ckpt 文件
                last_ckpt = [f for f in ckpt_candidates if "last" in f]
                ckpt_file = last_ckpt[0] if last_ckpt else sorted(ckpt_candidates)[0]
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                try:
                    print(f"[LGDVAEPipeline] Found existing checkpoint: {ckpt_path}, trying to load and skip training.")
                    # 直接从 checkpoint 加载 LightningModule；如果失败会进入 except
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

        # 训练完直接建一个 Inference wrapper（不从 ckpt 读，直接包内存模型）
        self.infer = Inference(autoencoder, device=self.device)
        if self.mean_ and self.std_:
            print(f"[LGDVAEPipeline] Loading mean and std: ")
            self.infer.load_zscore_values(mean=self.mean_, std=self.std_)
            print(f"[LGDVAEPipeline] Loaded mean and std: {self.infer.mean_, self.infer.std_}")
        return self

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
            # kwargs: batch_size=..., z_g_ref=..., sigma=..., device=...
            synthetics = self.infer.generate_from_prototype(**kwargs)
        else:
            raise ValueError(f"Unknown transform mode: {mode!r}")

        return synthetics
