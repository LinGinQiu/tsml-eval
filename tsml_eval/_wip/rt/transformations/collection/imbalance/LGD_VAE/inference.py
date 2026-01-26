import contextlib
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import numpy as np


class Inference:
    """Lightweight wrapper for loading LGD-VAE checkpoints and running generation.

    Originally this file was used for MAE-style deterministic imputation with explicit masks.
    For the LGD-VAE setting, we drop MAE-specific masking logic and keep only:
      - checkpoint loading / device management
      - optional z-score utilities
      - thin wrappers around LGD-VAE generation interfaces.
    """

    @staticmethod
    def _load_checkpoint_obj(ckpt_path: str, map_location: str | torch.device = "cpu") -> dict:
        """Load a Lightning-style checkpoint and return the raw object."""
        # 明确关掉 weights_only，避免你刚才那个报错
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)

    @classmethod
    def from_checkpoint(
            cls,
            ckpt_path: str,
            model_class: type[nn.Module],
            *,
            model_kwargs: Optional[dict] = None,
            device: Optional[torch.device] = None,
            strict: bool = False,
            map_location: str | torch.device = "cpu",
    ) -> "Inference":
        """
        Build an Inference object by loading weights from a checkpoint.

        优先级：
        1. 如果用户手动传了 model_kwargs，就用用户的；
        2. 否则看 ckpt["hyper_parameters"] 里有没有能拿来构造模型的键；
        3. 最后才用空的 {}.
        """
        obj = cls._load_checkpoint_obj(ckpt_path, map_location=map_location)

        # 1) 拿 state_dict
        if isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
        else:
            state_dict = obj

        # 2) 决定要用的构造参数
        if model_kwargs is not None:
            use_kwargs = dict(model_kwargs)  # 用户说了算
        else:
            # 尝试从 ckpt 里捞
            hparams = {}
            if isinstance(obj, dict):
                hparams = obj.get("hyper_parameters", {}) or {}
            # 有些键是优化器用的，不是模型需要的，滤掉
            ban_keys = {"lr", "weight_decay", "betas", "optimizer", "scheduler"}
            use_kwargs = {k: v for k, v in hparams.items() if k not in ban_keys}

        # 3) 构建模型
        model = model_class(**use_kwargs)

        # 4) 尝试加载权重，保留你原来的前缀修正逻辑
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            if strict:
                raise
            fixed = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    fixed[k[len("model."):]] = v
                else:
                    fixed[k] = v
            model.load_state_dict(fixed, strict=False)

        # 5) 放到设备上
        if device is not None:
            model = model.to(device)
        model.eval()
        return cls(model, device=device)

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.model.eval()
        if device is None:
            # Try to infer from model parameters
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)

        self.mean_, self.std_ = None, None



    # -----------------------------
    # Z-score utilities and helpers
    # -----------------------------
    def load_zscore_stats(self, stats_path: str) -> tuple[Tensor, Tensor]:
        """Load train z-score stats and ensure float32 dtype for MPS compatibility."""
        obj = np.load(stats_path)
        mean_np = obj["mean"]
        std_np = obj["std"]
        mean = torch.from_numpy(mean_np).float().to(self.device)
        std = torch.from_numpy(std_np).float().to(self.device)
        return mean, std

    def load_zscore_values(self, mean:np.array, std:np.array):
        mean = torch.from_numpy(mean).float().to(self.device)
        std = torch.from_numpy(std).float().to(self.device)
        self.mean_, self.std_ = mean, std
        return self


    @staticmethod
    def apply_zscore(x: Tensor, mean: Tensor, std: Tensor, eps: float = 1e-8) -> Tensor:
        """Apply z-score normalization: (x - mean) / (std + eps). Shapes must be broadcastable.
        x: [B,C,T], mean/std: typically [1,C,1].
        """
        return (x - mean) / (std + eps)

    @staticmethod
    def invert_zscore(x_norm: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        """Invert z-score normalization: x_norm * std + mean. Shapes must be broadcastable."""
        return x_norm * std + mean

    def feature_extract(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        extract latent features from input time series
        """
        lite_model = getattr(self.model, "model", self.model)
        if hasattr(lite_model, "feature_extract"):
            if self.mean_ and self.std_:
                x = self.apply_zscore(x, self.mean_, self.std_)
            return lite_model.feature_extract(x.to(self.device))
        raise AttributeError("Underlying model does not implement `feature_extract`.")
    # -----------------------------
    # Generation helpers for LGD-VAE
    # -----------------------------
    def generate_vae_prior(
        self,x_min: Tensor,alpha: Optional[float] = None,) -> Tensor:
        """
        use vae prior to interpret the latent embedding to generate new samples
        """
        lite_model = getattr(self.model, "model", self.model)
        if hasattr(lite_model, "generate_vae_prior"):
            if self.mean_ and self.std_:
                x_min = self.apply_zscore(x_min, self.mean_, self.std_)
            return self.invert_zscore(
                lite_model.generate_vae_prior(x_min.to(self.device),alpha),
                mean=self.mean_,
                std=self.std_
            )
        raise AttributeError("Underlying model does not implement `generate_from_pair`.")

    def generate_mix_pair(self, x_min: Tensor, x_maj: Tensor, use_y:bool=False,) -> Tensor:
        """
        方式二：给一条 minority 序列和一条 majority 序列，生成门控混合版。
        要求底层模型（或其 .model）实现 `generate_from_pair` 方法。
        """
        lite_model = getattr(self.model, "model", self.model)
        if hasattr(lite_model, "generate_from_pair"):
            if self.mean_ and self.std_:
                x_min = self.apply_zscore(x_min, self.mean_, self.std_)
                x_maj = self.apply_zscore(x_maj, self.mean_, self.std_)
            return self.invert_zscore(
                lite_model.generate_from_pair(x_min.to(self.device), x_maj.to(self.device), use_y=use_y),
                mean = self.mean_,
                std = self.std_
            )
        raise AttributeError("Underlying model does not implement `generate_from_pair`.")

    def generate_smote_latent(
        self,
        x_min1: Tensor,
        x_min2: Tensor,
        alpha: Optional[float] = None,
        num_samples: int = 1,
    ) -> Tensor:
        """
        方式三：在少数类 latent 空间做 SMOTE 风格的插值生成。
        要求底层模型（或其 .model）实现 `generate_from_latent_smote` 方法。
        """
        lite_model = getattr(self.model, "model", self.model)
        if hasattr(lite_model, "generate_from_latent_smote"):
            if self.mean_ and self.std_:
                x_min1 = self.apply_zscore(x_min1, self.mean_, self.std_)
                x_min2 = self.apply_zscore(x_min2, self.mean_, self.std_)
            return self.invert_zscore(
                lite_model.generate_from_latent_smote(
                x_min1.to(self.device),
                x_min2.to(self.device),
                alpha=alpha,
                num_samples=num_samples,),
                mean = self.mean_,
                std = self.std_
            )
        raise AttributeError("Underlying model does not implement `generate_from_latent_smote`.")

    def generate_from_prototype(
        self,
            x_min: Tensor) -> Tensor:
        """

        """
        lite_model = getattr(self.model, "model", self.model)
        if hasattr(lite_model, "generate_from_prototype"):
            if self.mean_ and self.std_:
                x_min = self.apply_zscore(x_min, self.mean_, self.std_)
            return self.invert_zscore(
                lite_model.generate_from_prototype(x_min.to(self.device)),
                mean=self.mean_,
                std=self.std_
            )
        raise AttributeError("Underlying model does not implement `generate_from_pair`.")
