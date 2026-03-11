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

        recon = metrics.get("train_recon_loss")
        kl_g = metrics.get("train/kl_g")
        kl_c = metrics.get("train/kl_c")
        align = metrics.get("train/align_loss")
        cls = metrics.get("train/cls_loss")
        center = metrics.get("train_loss_center")
        disentangle = metrics.get("train/disentangle_loss")
        total = metrics.get("train_loss")

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
        gmeans = metrics.get("eval_gen_g_means")
        macrof1 = metrics.get("eval_gen_f1_macro")
        acc = metrics.get("eval_acc")
        gen_metrics = metrics.get("eval_gen")
        if acc is not None:
            print(f"[Epoch {epoch}] Val acc={float(acc):.4f}"
                  # f", gmeans={float(gmeans):.4f}"
                  #   f", macrof1={float(macrof1):.4f}"
                  #   f", gen={float(gen_metrics):.4f}"
                  )


# --------------------------------------------------------
# EarlyStopping with warmup: DelayedEarlyStopping
# --------------------------------------------------------
class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, warmup_epochs: int = 10, *args, **kwargs):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(*args, **kwargs)
        print(f"[DelayedEarlyStopping] Initialized. Warmup: {self.warmup_epochs} epochs. Monitoring: {self.monitor}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        # 获取当前监控指标的值
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        # 1. Warmup 阶段
        if epoch < self.warmup_epochs:
            self.wait_count = 0  # 强制重置计数器
            if current_score is not None:
                print(f"[Epoch {epoch}] Warmup Phase: {self.monitor} = {current_score:.4f} (EarlyStopping Inactive)")
            else:
                print(f"[Epoch {epoch}] Warmup Phase: Waiting for {self.monitor}...")
            return

        # 2. 正常监控阶段
        # 在调用父类逻辑前记录一下旧的 wait_count
        old_wait_count = self.wait_count

        # 调用基类逻辑执行真正的检查
        super().on_validation_epoch_end(trainer, pl_module)

        # 3. 打印 Debug 信息
        if current_score is not None:
            status = "Improved" if self.wait_count == 0 else "No Improvement"
            best_score = self.best_score.item() if hasattr(self.best_score, "item") else self.best_score

            print(f"🔍 [Epoch {epoch}] Monitoring: {self.monitor} = {current_score:.4f} | "
                  f"Best = {best_score:.4f} | "
                  f"Patience = {self.wait_count}/{self.patience} ({status})")

            if self.wait_count >= self.patience:
                print(f"[Early Stop] Patience reached at epoch {epoch}. Stopping training...")
        else:
            print(f"[Epoch {epoch}] Warning: {self.monitor} not found in callback_metrics!")

    def on_train_end(self, trainer, pl_module):
        print("Training finished. Final EarlyStopping state: "
              f"Best {self.monitor} = {self.best_score:.4f}, total epochs = {trainer.current_epoch}")
        print("Best model path:", getattr(trainer.checkpoint_callback, "best_model_path", "N/A"))

from lightning.pytorch.callbacks import ModelCheckpoint


class DelayedModelCheckpoint(ModelCheckpoint):
    def __init__(self, warmup_epochs: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_save_epoch = int(warmup_epochs)
        print(f"💾 [DelayedModelCheckpoint] Initialized. Saving will start after Epoch {self.start_save_epoch}")

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_save_epoch:
            # 仅仅打印，不调用 super()，从而跳过保存逻辑
            if trainer.is_global_zero:  # 只在主进程打印，避免多卡训练刷屏
                print(f"⏳ [Epoch {epoch}] Checkpoint: Warmup phase, skipping save...")
            return

        # 记录一下保存前的路径，用于 debug
        old_best = self.best_model_path

        super().on_validation_end(trainer, pl_module)

        # 如果保存路径变了，说明存了新模型
        if self.best_model_path != old_best:
            print(f"🌟 [Epoch {epoch}] Checkpoint: New best model saved at {self.best_model_path}")

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
        task: str = "generation",  # generation / classification
    ):
        """
        dataset_name: 覆盖 YAML 里的 data.dataset_name
        seed: 覆盖 YAML 中 experiment.seed
        device: 手动指定训练/推断设备（不指定就交给 Lightning 和 Inference 自己处理
        mode: 运行模式，classification / generation if is classification will load pretrained model for feature extraction
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
        self.task = task.lower()

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
            rebalance=True,
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
            eval_dataset = UCRDataset(
            data=X_tr,
            labels=y_tr,
            split="eval",
            normalizer=None,
            augmentation_ratio=0.0,
            rebalance=False,
        )

        print(
            f"{datetime.now()} : Train size: {len(train_dataset)}; "
            f"Eval size: {len(eval_dataset)if eval_dataset else None}."
            f"Series Channel and length: {C}, {L}"
        )

        if getattr(train_dataset, "sample_weights", None) is not None:
            print("Use WeightedRandomSampler for training data.")
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.data.train_batch_size,
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
                batch_size=len(eval_dataset),
                shuffle=False,
                num_workers=cfg.data.loader_workers,
            )
        else:
            eval_loader = None
        return train_dataset, eval_dataset, train_loader, eval_loader

    # -------------------------
    # 内部工具：构建 pl_model + Trainer
    # -------------------------
    def _build_model_and_trainer(self, train_dataset: UCRDataset, oracle: torch.nn.Module = None, classifier=None) -> tuple[LitAutoEncoder, pl.Trainer]:
        cfg = self.cfg
        train_size = len(train_dataset)
        seq_len = int(train_dataset.data.shape[-1])
        cfg_model = dict(cfg.model)  # 拷一份，避免修改原配置

        # 简单按样本数调一调模型尺寸（跟你原来的逻辑一致）
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
            oracle_model=classifier,  # <--- 关键修改：注入裁判
            mean_=self.mean_,
            std_=self.std_,
        )
        # 把完整 cfg 挂上，方便 pl_model 里读取 optim/trainer 配置
        autoencoder.cfg = cfg

        # callbacks
        callbacks = []

        # 策略 B: 保存“重构质量最高”的模型 (Fidelity Best)
        # 这个模型生成的波形最平滑，最像真实数据
        callbacks.append(DelayedModelCheckpoint(
            dirpath=cfg.paths.ckpt_dir,
            filename="{eval_acc:.4f}-{epoch:02d}",
            monitor="eval_acc",
            mode="max",
            save_top_k=3,
            warmup_epochs=5,
            save_last=False
        ))
        callbacks.append(DelayedModelCheckpoint(
            dirpath=cfg.paths.ckpt_dir,
            filename="{train_loss:.4f}-{epoch:02d}",
            monitor="train_loss",
            mode="min",
            save_top_k=3,
            warmup_epochs=5,
            save_last=False
        ))

        if "callbacks" in cfg and "early_stopping" in cfg.callbacks:
            # allow an optional `warmup_epochs` key in the early_stopping config
            # so we can ignore early-stopping checks for the initial training epochs
            callbacks.append(DelayedEarlyStopping(**cfg.callbacks.early_stopping))
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

    # 在 LGDVAEPipeline 类中添加
    def _train_static_oracle(self, X_tr, y_tr, seq_len, in_chans):
        """在 VAE 训练前，训练一个静态判别器作为 Oracle，并选择验证集 Macro F1 最好的版本"""
        print("🚀 Training static Oracle (discriminator) with Val-split and Macro-F1 monitoring...")

        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint
        import tempfile
        import shutil
        import os

        # 1. 划分训练集和验证集 (例如 8:2)，使用 stratify 保证类别不平衡比例一致
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
        )

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
            batch_size=32, shuffle=False
        )

        # 2. 实例化 TimesNet (确保 TimesNetQualityClassifier 内部有 self.log("val_f1_macro", ...))
        from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.pl_model import \
            TimesNetQualityClassifier

        oracle = TimesNetQualityClassifier(
            input_channels=in_chans,
            seq_len=seq_len,
            num_classes=len(np.unique(y_tr)),
            d_model=64,
            lr=1e-3
        )

        # 3. 设置临时路径用于保存最优模型权重
        tmp_ckpt_dir = tempfile.mkdtemp()
        checkpoint_callback = ModelCheckpoint(
            dirpath=tmp_ckpt_dir,
            filename="best_oracle",
            monitor="val_f1_macro",
            mode="max",
            save_top_k=1,
            save_weights_only=True
        )

        # 4. 训练
        trainer = pl.Trainer(
            max_epochs=50,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback],
            logger=False,  # 如果需要可视化可以开启 TensorBoard
            enable_progress_bar=False,
            enable_checkpointing=True  # 必须开启以使用 callback
        )

        trainer.fit(oracle, train_loader, val_loader)

        # 5. 加载表现最好的权重
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(
                f"✨ Loading best Oracle weights from {best_model_path} (Score: {checkpoint_callback.best_model_score:.4f})")
            # 直接加载到当前 oracle 实例
            ckpt = torch.load(best_model_path)
            oracle.load_state_dict(ckpt['state_dict'])

        # 6. 冻结并清理
        oracle.eval()
        for param in oracle.parameters():
            param.requires_grad = False

        # 清理临时文件
        shutil.rmtree(tmp_ckpt_dir)

        print('✅ Static Oracle training and selection completed.')
        return oracle

    def _tournament_selection(self, ckpt_paths, X_tr, y_tr, k_folds=5):
        from sklearn.model_selection import StratifiedKFold
        import torch
        import numpy as np
        from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.pl_model import \
            train_and_eval_classifier
        import gc
        if self.mean_ is not None and self.std_ is not None:
            X_tr = X_tr * self.std_ + self.mean_
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        ckpt_scores = {}

        for path in ckpt_paths:
            print(f"🧐 Evaluating CKPT: {os.path.basename(path)}")
            temp_vae = Inference.from_checkpoint(path, LitAutoEncoder, device=self.device)

            fold_f1s = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
                # 1. 提取训练折并转为原始量级 (使用 .copy() 确保安全)
                x_train_fold = X_tr[train_idx]
                y_train_fold = y_tr[train_idx]

                # 2. 准备生成用的输入 (必须是归一化后的，因为 VAE Encoder 见过的是归一化分布)
                x_min_raw = torch.from_numpy(
                    x_train_fold[y_train_fold == self.cfg.model.minority_class_id]).float().to(self.device)

                # 3. 生成 9 倍新样本 (Inference 会自动反归一化回原始量级)
                with torch.no_grad():
                    # 注意：如果 Inference 接口没写 num_variations，请确保内部 repeat 逻辑
                    num_min = x_min_raw.size(0)
                    indexs = torch.randint(0, num_min, (num_min * 9,), device=self.device)
                    x_min_raw_expanded = x_min_raw[indexs]
                    x_gen = temp_vae.generate_vae_prior(x_min_raw_expanded, alpha=0.5)
                    print('   Generated samples shape:', x_gen.shape)
                # 4. 提取并转换验证折量级
                x_val_raw = X_tr[val_idx]
                y_val_fold = y_tr[val_idx]

                # 5. 构建统一原始量级的增强训练集
                # 将 numpy 的原始训练折转为 tensor 并移至设备
                x_train_tensor = torch.from_numpy(x_train_fold).float().to(self.device)
                x_combined = torch.cat([x_train_tensor, x_gen], dim=0)

                y_gen = torch.full((x_gen.size(0),), self.cfg.model.minority_class_id, device=self.device)
                y_combined = torch.cat([torch.from_numpy(y_train_fold).long().to(self.device), y_gen], dim=0)

                # 6. 训练与评估
                metrics = train_and_eval_classifier(
                    x_combined, y_combined,
                    torch.from_numpy(x_val_raw).float().to(self.device),
                    torch.from_numpy(y_val_fold).long().to(self.device),
                    input_chans=self.cfg.model.in_chans,
                    seq_len=X_tr.shape[-1],
                    device=self.device
                )
                print(f'   Fold {fold + 1}/{k_folds} - Metrics: {metrics}')
                fold_f1s.append(metrics["val_acc"])

            avg_f1 = np.mean(fold_f1s)
            ckpt_scores[path] = avg_f1
            print(f"   -> Avg val_acc: {avg_f1:.4f}")

            # 显存清理
            del temp_vae
            torch.cuda.empty_cache()
            gc.collect()

        return max(ckpt_scores, key=ckpt_scores.get)
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

        print("🚀 Starting LGD-VAE training pipeline...")
        print("train a rf classifier to check the generated data quality")
        y_tr = y_tr.astype(np.int64)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)
        clf.fit(X_tr.squeeze(), y_tr)

        if X_te is None:
            print("use train data for evaluation since test data is not provided.")
            X_te = X_tr
            y_te = y_tr
        else:
            y_te = y_te.astype(np.int64)
        cfg = self.cfg
        dataset_name = cfg.data.dataset_name
        # X_train_maj = X_tr[y_tr != cfg.model.minority_class_id]
        # X_train_min = X_tr[y_tr == cfg.model.minority_class_id]
        means = X_tr.mean(axis=2, keepdims=True)  # 得到 (n_sample, 1, 1)
        stds = X_tr.std(axis=2, keepdims=True)  # 得到 (n_sample, 1, 1)
        X_norm = (X_tr - means) / (stds + 1e-8)

        print(f"X_norm shape: {X_norm.shape}")
        # 验证：随机选一个样本看均值是否为 0
        print(f"Sample 0 mean: {X_norm[0, 0, :].mean():.4f}")
        # normalizer_maj = ZScoreNormalizer().fit(X_train_maj)
        # self.normalizer = ZScoreNormalizer().fit(X_train_min)
        # print(f"dataset maj mean: {normalizer_maj.mean_} and std: {normalizer_maj.std_}")
        # stats_dir = os.path.join(cfg.paths.work_root, "stats")
        # os.makedirs(stats_dir, exist_ok=True)
        # np.savez(os.path.join(stats_dir, f"{dataset_name}_zscore.npz"),
        #          mean=self.normalizer.mean_, std=self.normalizer.std_)

        # print(f"dataset mean: {self.normalizer.mean_} and std: {self.normalizer.std_}")
        # X_tr[y_tr != cfg.model.minority_class_id] = normalizer_maj.transform(X_tr[y_tr != cfg.model.minority_class_id])
        # X_tr[y_tr == cfg.model.minority_class_id] = self.normalizer.transform(X_tr[y_tr == cfg.model.minority_class_id])

        X_tr = X_norm
        minority_mask = (y_tr == cfg.model.minority_class_id)
        self.mean_ = means
        self.std_ = stds

        # self.mean_ = self.normalizer.mean_
        # self.std_ = self.normalizer.std_

        # prerebalance use smote

        # 2) DataLoader + Dataset
        train_dataset, eval_dataset, train_loader, eval_loader = self._build_dataloaders(
            X_tr, y_tr, X_te, y_te
        )

        # 3) 构建模型 & Trainer
        # ------------------ 新增：Oracle 预训练 ------------------
        seq_len = X_tr.shape[-1]
        in_chans = X_tr.shape[1]
        # 我们在这里先练一个 Oracle
        # self.static_oracle = self._train_static_oracle(X_tr, y_tr, seq_len, in_chans)
        # -------------------------------------------------------
        # 3. 构建模型 (传入 Oracle)
        # autoencoder, trainer = self._build_model_and_trainer(train_dataset, oracle=self.static_oracle)
        autoencoder, trainer = self._build_model_and_trainer(train_dataset, classifier=clf)

        # 4) 训练：如果已有 checkpoint 则优先尝试加载；加载失败再回退到重新训练
        ckpt_path = None
        ckpt_dir = getattr(self.cfg.paths, "ckpt_dir", None)
        env_ckpt_path = os.environ.get("LGD_VAE_CHECKPOINT_PATH")

        # ---------------- Step 1: 确定 ckpt_path 路径 ----------------
        if env_ckpt_path and os.path.exists(env_ckpt_path):
            print(f"[Experiment] Loading specific checkpoint from env: {env_ckpt_path}")
            ckpt_path = env_ckpt_path
        else:
            # 只有在没有环境变量指定时，才去目录自动搜索
            if ckpt_dir is not None and os.path.isdir(ckpt_dir):
                # print(f"loading checkpoint from {ckpt_dir}")
                ckpt_candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

                if ckpt_candidates:
                    print(f"[Tournament] Found {len(ckpt_candidates)} checkpoints. Starting selection...")

                    # 构建完整的候选路径列表
                    full_ckpt_paths = [os.path.join(ckpt_dir, f) for f in ckpt_candidates]
                    ckpt_path = os.path.join(ckpt_dir, sorted(ckpt_candidates)[0])
                    # 调用选拔赛逻辑（见下文实现）
                    # 传入原始归一化后的数据 X_tr, y_tr
                    # ckpt_path = self._tournament_selection(full_ckpt_paths, X_tr, y_tr, k_folds=5)
                    #
                    # print(f"🥇 Tournament winner selected: {os.path.basename(ckpt_path)}")
                    # print(f"[Experiment] No env var found, auto-loading best model from {ckpt_dir}...")
                    # import re
                    # def extract_loss(name: str):
                    #     match = re.search(r"eval_gen=([0-9]+\.?[0-9]*)", name)
                    #     if match:
                    #         return float(match.group(1))
                    #     return float('inf')
                    #
                    # # 过滤出包含 eval/gen_f1_macro 的文件
                    # eval_ckpts = [f for f in ckpt_candidates if "eval_gen=" in f]
                    #
                    # if eval_ckpts:
                    #     # 使用 max() 找到 eval/gen_f1_macro 最大的文件
                    #     best_ckpt_file = max(eval_ckpts, key=extract_loss)
                    #     ckpt_path = os.path.join(ckpt_dir, best_ckpt_file)
                    #     print(f"Loading best model: {best_ckpt_file}")
                    # else:
                    #     # 其次：fallback 用 last.ckpt（如果存在）
                    #     last_ckpts = [f for f in ckpt_candidates if "last" in f]
                    #     if last_ckpts:
                    #         ckpt_path = os.path.join(ckpt_dir, last_ckpts[0])
                    #     else:
                    #         # 最后：随便选一个
                    #         ckpt_path = os.path.join(ckpt_dir, sorted(ckpt_candidates)[0])
                else:
                    print(f"[Experiment] Checkpoint directory exists but is empty.")

        # ---------------- Step 2: 尝试加载 ----------------
        if ckpt_path is not None:  # <--- [关键修改] 只有找到了路径才尝试加载
            try:
                print(f"[LGDVAEPipeline] Attempting to load checkpoint: {ckpt_path}")

                self.infer = Inference.from_checkpoint(
                    ckpt_path,
                    model_class=LitAutoEncoder,
                    model_kwargs=None,
                    device=self.device,
                    strict=False,
                )
                print("load successfully!")
                return self

            except Exception as e:
                print(f"[LGDVAEPipeline] Failed to load checkpoint ({e}).")
                print("[LGDVAEPipeline] Fallback to training from scratch.")
                ckpt_path = None  # 确保传给 trainer 的是 None，让它从头开始
        else:
            print("[LGDVAEPipeline] No checkpoint found. Starting training from scratch.")

        # ... 代码继续执行，进入 trainer.fit() ...
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

        # --- 新增：基于分类选拔赛的最优模型选择 ---
        print("Starting Tournament to select the best CKPT based on classification performance...")

        # 1. 获取 Top-3 CKPT 路径
        checkpoint_callback = trainer.checkpoint_callback
        ckpt_dir = checkpoint_callback.dirpath
        best_k_models = checkpoint_callback.best_k_models  # dict: {path: score}
        ckpt_paths = list(best_k_models.keys())

        # 3. 在该目录下搜索所有符合命名的 ckpt 文件
        # 这样可以捞到由其他 callback（比如监控 train_loss 的）保存的文件
        import glob
        if ckpt_dir and os.path.exists(ckpt_dir):
            search_pattern = os.path.join(ckpt_dir, "epoch=*.ckpt")
            all_local_ckpts = glob.glob(search_pattern)

            # 合并并去重（统一转为绝对路径防止重复）
            # 使用 set 确保即便 best_k 里的文件在文件夹里被搜到，也只保留一份
            all_ckpt_paths = list(set([os.path.abspath(p) for p in ckpt_paths] +
                                      [os.path.abspath(p) for p in all_local_ckpts]))
        else:
            print(f"Warning: Directory {ckpt_dir} not found. Using only callback records.")
            all_ckpt_paths = ckpt_paths

        # 4. 打印结果
        print(f'Total CKPTs found in {ckpt_dir}: {len(all_ckpt_paths)}')
        print('finish training, exit here')
        import sys
        sys.exit(0)
        for path in all_ckpt_paths:
            print(f' - {os.path.basename(path)}')
        best_path = self._tournament_selection(all_ckpt_paths, X_tr, y_tr)

        # 3. 加载最终胜出的模型
        print(f"🥇 Tournament winner: {best_path}")
        self.infer = Inference.from_checkpoint(
            best_path,
            model_class=LitAutoEncoder,
            model_kwargs=None,
            device=self.device,
            strict=False,
        )
        return self


    # -------------------------
    # 公共接口：transform / 生成
    # -------------------------
    # def transform(self, mode: str, **kwargs) -> torch.Tensor:
    #     """
    #     统一的生成接口，内部直接调用 Inference 的方法。
    #
    #     mode:
    #       - "vae_prior"      → 先验采样生成（generate_vae_prior）
    #       - "mix_pair"       → minority + majority pair 门控混合（generate_mix_pair）
    #       - "smote_latent"   → latent 空间 SMOTE 插值生成（generate_smote_latent）
    #       - "prototype"      → 基于 prototype 生成（generate_from_prototype）
    #     """
    #     if self.infer is None:
    #         raise RuntimeError("Pipeline is not fitted yet. Call `fit()` first.")
    #
    #     mode = mode.lower()
    #     if mode == "classification" and self.task=="classification":
    #         # kwargs: x=...
    #         synthetics = self.infer.feature_extract(**kwargs)
    #     if mode == "prior":
    #         # kwargs: batch_size=..., device=...
    #         synthetics = self.infer.generate_vae_prior(**kwargs)
    #     elif mode == "pair":
    #         # kwargs: x_min=..., x_maj=..., use_y=...
    #         synthetics = self.infer.generate_mix_pair(**kwargs)
    #     elif mode == "latent":
    #         # kwargs: x_min1=..., x_min2=..., alpha=..., num_samples=...
    #         synthetics = self.infer.generate_smote_latent(**kwargs)
    #     elif mode == "prototype":
    #         # kwargs: x_min=..., use_y=...
    #         synthetics = self.infer.generate_from_prototype(**kwargs)
    #     else:
    #         raise ValueError(f"Unknown transform mode: {mode!r}")
    #
    #     return synthetics

    def transform(self, x_min, mode: str=None, threshold=0.7, max_retries=10, alpha = None):
        """
        带有拒绝采样的少数类生成
        x_min: 原始少数类样本 [B, C, T]
        threshold: 判别器信心阈值，建议 0.8 以上
        max_retries: 尝试生成的最大轮次，防止死循环
        """
        self.infer.model.eval()
        # 假设你已经加载了一个预训练好的分类器到 self.discriminator
        # 如果没有，可以使用 pl_model.py 中的 TimesNetQualityClassifier
        all_needed = x_min.size(0)
        device = x_min.device
        discriminator = getattr(self, "static_oracle", None)

        if discriminator is None:
            print("⚠️ Warning: No static oracle found, performing standard generation.")
            return self.infer.generate_vae_prior(x_min=x_min, alpha=0.5)

        discriminator.to(x_min.device)
        batch_size = x_min.size(0)
        final_samples = []

        # 我们循环尝试，直到收集够 batch_size 数量的高质量样本

        for i in range(max_retries):
            current_needed = batch_size - sum(len(s) for s in final_samples)
            if current_needed <= 0:
                break

            # 1. 调用你现有的生成逻辑 (Alpha=0.5 插值)
            # 对应 model.py 中的 generate_vae_prior 实现
            candidates = self.infer.generate_vae_prior(x_min=x_min, alpha=0.5)

            # 2. 判别器评估 (确保输入维度正确 [B, C, T])
            with torch.no_grad():
                # 这里的 discriminator 需预先训练并加载
                logits = discriminator(candidates)
                probs = torch.nn.functional.softmax(logits, dim=1)

            # 3. 筛选少数类 (ID=1) 且信心值达标的样本
            minority_id = self.cfg.model.minority_class_id
            conf = probs[:, minority_id]
            mask = (conf > threshold) & (conf <= 0.99)

            valid_candidates = candidates[mask]
            if len(valid_candidates) > 0:
                final_samples.append(valid_candidates)

        # 4. 汇总结果
        if not final_samples:
            # 如果多次尝试都失败，回退到原始生成或报错
            print("[LGDVAEPipeline] No valid samples found.")
            return self.infer.generate_vae_prior(x_min=x_min)

        all_generated = torch.cat(final_samples, dim=0)
        if len(all_generated) <= all_needed:
            n_needed = all_needed - len(all_generated)
            print(f"[LGDVAEPipeline] Only {len(all_generated)} valid samples found, need {n_needed} more. Generating additional samples without filtering...")
            additional = self.infer.generate_vae_prior(x_min=x_min[:n_needed], alpha=0.5)
            all_generated = torch.cat([all_generated, additional], dim=0)
        return all_generated[:batch_size]
