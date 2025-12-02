# src/nn/lgd_vae_model.py

import math
from typing import Dict, Optional, Tuple
from matplotlib.animation import FuncAnimation
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def cross_covariance_loss(zg, zc):
    """
    z_g, z_c: (B, D)
    """
    B = zg.size(0)
    if B <= 2:
        return zg.new_tensor(0.0)
    zg = zg - zg.mean(dim=0, keepdim=True)
    zc = zc - zc.mean(dim=0, keepdim=True)

    cov = (zg.T @ zc) / (B - 1)   # (D, D)
    loss = (cov ** 2).mean()
    # 保险起见再防一层
    if torch.isnan(loss) or torch.isinf(loss):
        # 你可以在这里 print 一些 debug 信息
        # print("NaN in cross_covariance_loss, B:", B)
        return zg.new_tensor(0.0)
    return loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)           # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数
        pe = pe.unsqueeze(0)   # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


class ConvEmbedder(nn.Module):
    """和你之前 MAE 里的一样：输入 [B, C, T] → 输出 [B, T, D]."""
    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        y = self.conv(x)          # [B, D, T']
        return y.transpose(1, 2)  # [B, T', D]


class LatentHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            # 可选
            # nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    很简单的 decoder：latent → 线性 → Transformer → 线性回到通道数
    这里我们假定要还原到固定长度 seq_len.
    """
    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        out_channels: int,
        seq_len: int,
        dec_depth: int = 2,
        n_heads: int = 4,
        d_hid: int = 128,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(latent_dim, d_model)
        if num_classes is not None:
            self.y_embed = nn.Linear(num_classes, d_model)
        else:
            self.y_embed = None
        self.pos = PositionalEncoding(d_model, dropout, max_len=seq_len)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, n_heads, d_hid, dropout, batch_first=True
                )
                for _ in range(dec_depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_channels)

    def forward(self, z: Tensor,  y_onehot: Optional[Tensor] = None) -> Tensor:
        """
        z: [B, latent_dim]
        return: [B, C, T]
        """
        B = z.size(0)
        x = self.proj(z)                 # [B, D]
        if y_onehot is not None and self.y_embed is not None:
            y_emb = self.y_embed(y_onehot)  # [B, D]
            x = x + y_emb                   # 或 cat 再线性
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, D] 所有时间步共享这个latent
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.out(x)                  # [B, T, C]
        return x.transpose(1, 2)         # [B, C, T]


class LatentGatedDualVAE(nn.Module):
    """
    Latent-Gated Dual-VAE for Minority-Aware Time-Series Generation

    输入:  x: [B, C, T], y: [B] (int 或 long)
    输出:  一个 dict, 里面有 recon, loss 各种项
    """
    def __init__(
        self,
        in_chans: int = 1,
        seq_len: int = 200,
        d_model: int = 64,
        enc_depth: int = 2,
        n_heads: int = 4,
        d_hid: int = 128,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        latent_dim_global: int = 32,
        latent_dim_class: int = 32,
        minority_class_id: int = 1,
        dropout: float = 0.1,
        dec_depth: int = 2,
        gate_hidden: int = 64,
        align_lambda: float = 0.1,
        num_classes: int = None,
        cls_lambda: float = 0.1,
        use_prototype: bool = False,
        n_prototypes: int = 1,
        proto_lambda: float = 0.1,
        recon_metric: str = "mse",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.seq_len = seq_len
        self.minority_class_id = minority_class_id
        self.align_lambda = align_lambda
        self.recon_metric = recon_metric

        # 1) encoder: conv → PE → transformer
        self.embed = ConvEmbedder(
            in_channels=in_chans,
            d_model=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pos = PositionalEncoding(d_model, dropout, max_len=seq_len)

        self.encoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_hid,
                    dropout,
                    batch_first=True,
                )
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # majority global latent running mean (learnable center)
        self.z_g_maj_mean = nn.Parameter(torch.zeros(1, latent_dim_global))
        nn.init.normal_(self.z_g_maj_mean, mean=0.0, std=0.02)

        # EMA buffer to keep a running estimate of majority global mean
        self.register_buffer("z_g_maj_ema", torch.zeros(1, latent_dim_global))
        self.register_buffer("z_g_maj_ema_inited", torch.tensor(False, dtype=torch.bool))
        self.z_g_maj_momentum = 0.1


        # 2) two latent heads
        self.global_head_p = LatentHead(d_model, latent_dim_global)
        self.class_head_p = LatentHead(d_model, latent_dim_class)

        self.global_head_n = LatentHead(d_model, latent_dim_global)
        self.class_head_n = LatentHead(d_model, latent_dim_class)

        # 3) gate: 用 minority 的 z_c 预测一个 [0,1] 的门控，决定 z_g_min 和 z_g_maj_mean 的混合
        self.gate = nn.Sequential(
            nn.Linear(latent_dim_class, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, latent_dim_global),
            nn.Sigmoid(),
        )

        # 4) decoder
        total_latent = latent_dim_global + latent_dim_class
        self.decoder = Decoder(
            latent_dim=total_latent,
            d_model=d_model,
            out_channels=in_chans,
            seq_len=seq_len,
            dec_depth=dec_depth,
            n_heads=n_heads,
            d_hid=d_hid,
            dropout=dropout,
            num_classes=num_classes,
        )

        # 5) classifier
        self.num_classes = num_classes
        self.cls_lambda = cls_lambda
        # 6）Conditional vae
        self.y_embed_latent = None
        self.y_embed_dec = None

        # ---- Learnable Minority Prototype(s) ----
        self.use_prototype = use_prototype
        self.n_prototypes = int(n_prototypes)
        self.proto_lambda = float(proto_lambda)

        if num_classes is not None:
            # 用 class latent 做分类
            self.classifier = nn.Linear(total_latent, num_classes)
            self.y_embed_latent = nn.Linear(num_classes, d_model)  # 给 encoder latent 用
            self.y_embed_dec = nn.Linear(num_classes, d_model)  # 给 decoder 用
        else:
            self.classifier = None

        if self.use_prototype:
            if self.n_prototypes <= 1:
                # single learnable prototype vector for minority class latent (z_c space)
                self.proto_minority = nn.Parameter(torch.zeros(latent_dim_class))
                nn.init.normal_(self.proto_minority, mean=0.0, std=0.02)
            else:
                # K learnable prototypes with soft assignment
                self.proto_minority = nn.Parameter(torch.zeros(self.n_prototypes, latent_dim_class))
                nn.init.normal_(self.proto_minority, mean=0.0, std=0.02)
                # temperature for softmax over prototypes
                self.register_buffer("proto_tau", torch.tensor(0.5))

    # --------- helper ---------
    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        # KL(q||p), p ~ N(0, I)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # --------- forward parts ---------

    def encode_feat(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x = self.embed(x)  # [B, T', D]
        x = self.pos(x)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.enc_norm(x)  # [B, T', D]
        feat = x.mean(dim=1)  # [B, D]
        return feat

    def encode_latent_branches(self, feat: Tensor, y: Optional[Tensor] = None) \
            -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B = feat.size(0)
        device = feat.device
        if y is None:
            # 没标签时退化用“positive 分支”或随便一个（建议统一用 min 分支）
            mu_g, logvar_g = self.global_head_p(feat)
            mu_c, logvar_c = self.class_head_p(feat)
        else:
            y = y.to(device)
            is_pos = (y == self.minority_class_id)  # or y==1
            is_neg = ~is_pos

            G = self.global_head_p.fc_mu.out_features
            C = self.class_head_p.fc_mu.out_features

            mu_g = torch.empty(B, G, device=device)
            logvar_g = torch.empty(B, G, device=device)
            mu_c = torch.empty(B, C, device=device)
            logvar_c = torch.empty(B, C, device=device)

            # positive branch
            if is_pos.any():
                feat_pos = feat[is_pos]
                mu_g_pos, logvar_g_pos = self.global_head_p(feat_pos)
                mu_c_pos, logvar_c_pos = self.class_head_p(feat_pos)
                mu_g[is_pos] = mu_g_pos
                logvar_g[is_pos] = logvar_g_pos
                mu_c[is_pos] = mu_c_pos
                logvar_c[is_pos] = logvar_c_pos

            # negative branch
            if is_neg.any():
                feat_neg = feat[is_neg]
                mu_g_neg, logvar_g_neg = self.global_head_n(feat_neg)
                mu_c_neg, logvar_c_neg = self.class_head_n(feat_neg)
                mu_g[is_neg] = mu_g_neg
                logvar_g[is_neg] = logvar_g_neg
                mu_c[is_neg] = mu_c_neg
                logvar_c[is_neg] = logvar_c_neg

        return mu_g, logvar_g, mu_c, logvar_c

    def encode(self,x: Tensor, y: Optional[Tensor] = None) \
            -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        feat = self.encode_feat(x)  # [B, D]
        if y is not None and self.num_classes is not None and self.y_embed_latent is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_cond = self.y_embed_latent(y_onehot)  # [B, D]
            feat = feat + y_cond  # 或者 cat 再线性投影
        mu_g, logvar_g, mu_c, logvar_c = self.encode_latent_branches(feat, y)
        return feat, mu_g, logvar_g, mu_c, logvar_c


    def forward(
            self,
            x: Tensor,
            y: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        训练/推理统一入口
        """
        device = x.device
        feat, mu_g, logvar_g, mu_c, logvar_c = self.encode(x, y)
        # AE version: use deterministic latents (no sampling)
        z_g = mu_g   # [B, G]
        z_c = mu_c   # [B, C]

        disentangle_loss = torch.tensor(0.0, device=device)
        if y is None:
            # 没给标签就当全是 minority，用自己的 z_g
            z_g_final = z_g
            align_loss = torch.tensor(0.0, device=device)
        else:
            y = y.to(device)
            is_min = (y == self.minority_class_id)          # [B]
            is_maj = ~is_min
            # 如果 batch 里一个多数都没有，那就退化成自己用自己
            if is_maj.any():
                z_g_maj = z_g[is_maj]                      # [B_maj, G]
                # batch-wise majority mean (no grad, for stats only)
                z_c_maj = z_c[is_maj]
                disentangle_loss = cross_covariance_loss(z_g_maj, z_c_maj)

                batch_mean = z_g_maj.mean(dim=0, keepdim=True).detach()  # [1, G]

                # update EMA buffer during training
                if self.training:
                    if not self.z_g_maj_ema_inited:
                        self.z_g_maj_ema.copy_(batch_mean)
                        self.z_g_maj_ema_inited.fill_(True)
                        # also use this to initialize the learnable center once
                        with torch.no_grad():
                            self.z_g_maj_mean.copy_(batch_mean)
                    else:
                        m = self.z_g_maj_momentum
                        self.z_g_maj_ema.mul_(1.0 - m).add_(m * batch_mean)

                # use the learnable majority center as z_g_maj_mean
                # (fall back to current batch mean before EMA is inited)
                if self.z_g_maj_ema_inited:
                    z_g_maj_mean = self.z_g_maj_mean  # [1, G], trainable
                else:
                    z_g_maj_mean = batch_mean  # startup phase


                # minority 部分用 gate 融合
                z_g_final = z_g.clone()

                if is_min.any():
                    z_c_min = z_c[is_min]                  # [B_min, C]
                    gate = self.gate(z_c_min)              # [B_min, G], in [0,1]
                    z_g_min = z_g[is_min]                  # [B_min, G]
                    z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj_mean
                    z_g_final[is_min] = z_g_mix

                # 对齐损失: 让 minority 的 z_g 靠近 majority 的均值
                if is_min.any():
                    align_loss = (z_g[is_min] - z_g_maj_mean).pow(2).mean()
                else:
                    align_loss = torch.tensor(0.0, device=device)
            else:
                # 全是 minority，没法对齐
                z_g_final = z_g
                align_loss = torch.tensor(0.0, device=device)

        # 拼 latent → decode
        z_full = torch.cat([z_g_final, z_c], dim=1)         # [B, G+C]

        if y is not None and self.num_classes is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_onehot = None
        recon = self.decoder(z_full, y_onehot=y_onehot)                        # [B, C, T]

        # reconstruction loss
        # 如果你的数据有 NaN，可以加 nan_to_num
        recon_loss = 0
        if self.recon_metric == 'soft_dtw':
            from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
            recon_loss, _, _ = dilate_loss(outputs=recon, targets=x, device=x.device, alpha=1)
        elif self.recon_metric == 'dilate':
            from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
            recon_loss, _, _ = dilate_loss(outputs=recon, targets=x, device=x.device, alpha=0.5)
        elif self.recon_metric == 'mse':
            recon_loss = ((recon - x) ** 2).mean()
        else:
            raise NotImplementedError

        # AE version: no KL divergence term
        kl_g = torch.tensor(0.0, device=device)
        kl_c = torch.tensor(0.0, device=device)

        cls_loss = torch.tensor(0.0, device=device)
        cls_logits = None
        if self.classifier is not None and y is not None:

            logits_base = self.classifier(z_full)  # [B, num_classes]
            # 默认先用 base 来算 loss
            cls_loss = F.cross_entropy(logits_base, y)
            cls_logits = logits_base  # 用于 pl 里算 F1 / Acc

            is_min = (y == self.minority_class_id)
            is_maj = ~is_min
            if is_min.any() and is_maj.any():
                z_c_min = z_c[is_min]  # [B_min, C]

                # majority 的 global center：优先用可学习的 center
                if self.z_g_maj_ema_inited:
                    z_g_center = self.z_g_maj_mean  # [1, G]
                else:
                    # 还没 init 时退化成当前 batch 的 majority 均值
                    z_g_center = z_g[is_maj].mean(dim=0, keepdim=True)  # [1, G]

                # 扩展到每个 minority 样本
                B_min = z_c_min.size(0)
                z_g_center_exp = z_g_center.expand(B_min, -1)  # [B_min, G]

                # 构造“reinforcement minority” latent：z_g_maj_mean ⊕ z_c_min
                z_full_cf = torch.cat([z_g_center_exp, z_c_min], dim=1)  # [B_min, G+C]

                # 这些样本都当作 minority，标签就是 minority_class_id
                y_cf = torch.full(
                    (B_min,),
                    fill_value=self.minority_class_id,
                    dtype=y.dtype,
                    device=device,
                )

                # 拼到一起算一个更强的 CE：真实样本 + 额外 minority 样本
                logits_cf = self.classifier(z_full_cf)  # [B_min, num_classes]
                logits_all = torch.cat([logits_base, logits_cf], dim=0)  # [B + B_min, num_classes]
                y_all = torch.cat([y, y_cf], dim=0)  # [B + B_min]

                cls_loss = F.cross_entropy(logits_all, y_all)
                # 注意：cls_logits 还是用原始 batch 的 logits_base，保持和 dataloader 的 y 对齐
                cls_logits = logits_base

        if not self.z_g_maj_ema_inited:
            loss_maj_center = torch.tensor(0.0, device=device)
        else:
            loss_maj_center = F.mse_loss(self.z_g_maj_mean, self.z_g_maj_ema.detach())

        out = {
            "recon_loss": recon_loss,
            "kl_g": kl_g,
            "kl_c": kl_c,
            "disentangle_loss": disentangle_loss,
            "align_loss": align_loss,
            "loss_center": loss_maj_center,
            "cls_loss": cls_loss,
            "cls_logits": cls_logits,
            "recon": recon,
            "z_g": z_g,
            "z_c": z_c,
        }

        # expose attention prototype if computed
        if self.use_prototype and hasattr(self, "n_prototypes") and self.n_prototypes > 1 and 'proto_attn' not in locals():
            # nothing to add
            pass

        return out
    # ---- generation
    @torch.no_grad()
    def generate_vae_prior(self, num_samples:int, device=None) -> Tensor:
        """
        y: [B] int labels
        return: x_gen [B, C, T]
        """
        if device is None:
            device = next(self.parameters()).device

        y = torch.ones(num_samples, device=device).long()
        B = num_samples

        G = self.global_head_p.fc_mu.out_features
        C = self.class_head_p.fc_mu.out_features

        z_g_sample = torch.randn(B, G, device=device)
        z_c_sample = torch.randn(B, C, device=device)

        # optional: 用 gate 做一些 minority-aware 的混合逻辑
        gate = self.gate(z_c_sample)
        z_g_mix = gate * z_g_sample + (1.0 - gate) * z_g_sample  # 暂时等价

        z_full = torch.cat([z_g_mix, z_c_sample], dim=1)

        if self.num_classes is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_onehot = None

        x_gen = self.decoder(z_full, y_onehot=y_onehot)
        return x_gen


    # 一个简单的生成接口：给 minority 的一条 x，同时给一条 majority 的 x，当场生成“混合版”
    @torch.no_grad()
    def generate_from_pair(self, x_min: Tensor, x_maj: Tensor, use_y:bool=False) -> Tensor:
        """
        x_min, x_maj: [1, C, T]
        return: [1, C, T] generated
        """
        if use_y:
            y_min = torch.ones(x_min.shape[0], device=x_min.device).long()
            y_maj = torch.zeros(x_maj.shape[0], device=x_maj.device).long()
        else:
            y_min = None
            y_maj = None
        _, mu_g_min, logvar_g_min, mu_c_min, logvar_c_min = self.encode(x_min, y=y_min)
        _, mu_g_maj, logvar_g_maj, _, _ = self.encode(x_maj, y=y_maj)

        # AE version: use encoder means directly (no sampling)
        z_g_min = mu_g_min
        z_g_maj = mu_g_maj
        z_c_min = mu_c_min

        gate = self.gate(z_c_min)  # [1, G]
        z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj

        z_full = torch.cat([z_g_mix, z_c_min], dim=1)
        if use_y:
            y = y_min
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_onehot = None
        x_gen = self.decoder(z_full, y_onehot=y_onehot)
        return x_gen

    @torch.no_grad()
    def generate_from_latent_smote(
        self,
        x_min1: Tensor,
        x_min2: Tensor,
        alpha: Optional[float] = None,
        num_samples: int = 1,
    ) -> Tensor:
        """
        方式三：在 minority 的 latent 空间做 SMOTE 风格插值生成。

        Args:
            x_min1: 第一条少数类序列，[B, C, T] 或 [1, C, T]
            x_min2: 第二条少数类序列，[B, C, T] 或 [1, C, T]，形状需与 x_min1 相同
            alpha: 若给定，则使用固定插值系数 alpha ∈ [0,1]；
                   若为 None 且 num_samples==1，则用 0.5；
                   若为 None 且 num_samples>1，则每个样本随机采样 alpha ~ U(0,1)。
            num_samples: 生成多少条插值样本。

        Returns:
            - 若 num_samples==1: 返回 [B, C, T]
            - 若 num_samples>1: 返回 [num_samples, B, C, T]
        """
        device = x_min1.device
        y_min1 = torch.ones(x_min1.shape[0], device=x_min1.device).long()
        y_min2 = torch.ones(x_min2.shape[0], device=x_min2.device).long()
        # 编码两条少数类样本
        _, mu_g1, logvar_g1, mu_c1, logvar_c1 = self.encode(x_min1,y=y_min1)
        _, mu_g2, logvar_g2, mu_c2, logvar_c2 = self.encode(x_min2,y=y_min2)

        # AE version: use deterministic encoder outputs
        z_g1 = mu_g1  # [B, G]
        z_g2 = mu_g2  # [B, G]
        z_c1 = mu_c1  # [B, C]
        z_c2 = mu_c2  # [B, C]

        B = z_g1.size(0)

        if num_samples == 1:
            if alpha is None:
                alpha_val = 0.5
            else:
                alpha_val = float(alpha)
            a = torch.full((B, 1), alpha_val, device=device)
            z_g_mix = (1.0 - a) * z_g1 + a * z_g2
            z_c_mix = (1.0 - a) * z_c1 + a * z_c2

            z_full = torch.cat([z_g_mix, z_c_mix], dim=1)   # [B, G+C]
            x_gen = self.decoder(z_full)                    # [B, C, T]
            return x_gen
        else:
            xs = []
            for _ in range(num_samples):
                if alpha is None:
                    a = torch.rand(B, 1, device=device)
                else:
                    a = torch.full((B, 1), float(alpha), device=device)
                z_g_mix = (1.0 - a) * z_g1 + a * z_g2
                z_c_mix = (1.0 - a) * z_c1 + a * z_c2
                z_full = torch.cat([z_g_mix, z_c_mix], dim=1)
                x_gen_k = self.decoder(z_full)  # [B, C, T]
                xs.append(x_gen_k)

            return torch.stack(xs, dim=0)  # [num_samples, B, C, T]

    @torch.no_grad()
    def generate_from_prototype(
        self,
        batch_size: int = 1,
        z_g_ref: Optional[Tensor] = None,
        sigma: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Prototype-aware minority sample generation.

        Args:
            batch_size: number of samples to generate
            z_g_ref: optional reference global latent(s) [B,G]; if None, sample N(0,I)
            sigma: std for Gaussian perturbation around prototype in z_c space
            device: torch device

        Returns:
            x_gen: [B, C, T]
        """
        if device is None:
            device = next(self.parameters()).device

        G = self.global_head_p.fc_mu.out_features
        C = self.class_head_p.fc_mu.out_features

        if self.use_prototype:
            if getattr(self, "n_prototypes", 1) <= 1:
                proto = self.proto_minority.to(device).unsqueeze(0).expand(batch_size, -1)  # [B,C]
            else:
                # sample a prototype index uniformly
                protos = self.proto_minority.to(device)
                idx = torch.randint(low=0, high=protos.size(0), size=(batch_size,), device=device)
                proto = protos[idx]  # [B,C]
        else:
            proto = torch.zeros(batch_size, C, device=device)

        noise = torch.randn(batch_size, C, device=device) * float(sigma)
        z_c_sample = proto + noise  # [B, C]

        if z_g_ref is None:
            z_g_sample = torch.randn(batch_size, G, device=device)
        else:
            if z_g_ref.dim() == 1:
                z_g_ref = z_g_ref.unsqueeze(0)
            if z_g_ref.size(0) == 1 and batch_size > 1:
                z_g_ref = z_g_ref.expand(batch_size, -1)
            z_g_sample = z_g_ref.to(device)

        # optionally gate-mix with majority-like global using the same gating net
        gate = self.gate(z_c_sample)  # [B, G] in [0,1]
        z_g_mix = gate * z_g_sample + (1.0 - gate) * z_g_sample  # here identical; keep hook for future mixing

        z_full = torch.cat([z_g_mix, z_c_sample], dim=1)  # [B, G+C]
        x_gen = self.decoder(z_full)  # [B, C, T]
        return x_gen
