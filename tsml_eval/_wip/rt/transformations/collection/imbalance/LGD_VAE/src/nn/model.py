# src/nn/lgd_vae_model.py

import math
from typing import Dict, Optional, Tuple
from matplotlib.animation import FuncAnimation
import torch
from sympy.abc import alpha
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
        seq_len: int = 200,
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


class MultiScaleConvEmbedder(nn.Module):
    """
    多尺度卷积嵌入层 (Multi-Scale Conv Embedder)
    输入 [B, C, T] -> 输出 [B, T, D]

    原理：
    并行通过多个不同 kernel_size 或 dilation 的卷积层，
    利用 padding='same' 保证时间维度不改变且中心对齐，
    最后将不同感受野提取的特征拼接起来。
    """

    def __init__(
            self,
            in_channels: int,
            d_model: int = 64,
            kernel_sizes= [1, 3, 5, 3],
            dilations=[1, 1, 1, 4],
            stride: int = 1,
    ) -> None:
        super().__init__()

        assert len(kernel_sizes) == len(dilations), "kernel_sizes 和 dilations 列表长度必须一致"

        self.num_branches = len(kernel_sizes)
        assert d_model % self.num_branches == 0, f"d_model ({d_model}) 必须能被分支数 ({self.num_branches}) 整除"

        self.out_channels_per_branch = d_model // self.num_branches

        self.branches = nn.ModuleList()

        for k, d in zip(kernel_sizes, dilations):
            # 关键点：为了保证输出 T 不变且时间对齐，需要计算 padding
            # Formula: output = (input + 2*padding - dilation*(kernel-1) - 1)/stride + 1
            # 当 stride=1 时，要保持 output=input，需要 2*padding = dilation*(kernel-1)
            # PyTorch 的 padding='same' 仅支持 stride=1 且 dilation 使得 padding 为两边对称的情况
            # 这里我们手动写 padding='same' (PyTorch >= 1.10) 或者手动计算

            # 使用 padding='same' 能够自动处理奇数偶数核的对齐问题，这是最方便的
            # 注意：padding='same' 要求 stride=1
            if stride == 1:
                pad_mode = 'same'
                padding_val = 0  # 占位，实际由 pad_mode 控制
            else:
                # 如果 stride != 1，很难保证不同核对齐，通常不建议这样做多尺度
                raise ValueError("对于多尺度拼接，建议保持 stride=1 以保证时间对齐")

            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.out_channels_per_branch,
                kernel_size=k,
                stride=stride,
                padding=pad_mode,
                dilation=d,
                bias=False
            )
            self.branches.append(conv_layer)

        # 可选：如果你希望拼接后特征融合得更好，可以加一个 1x1 卷积混合一下
        self.project = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        branch_outputs = []
        for conv in self.branches:
            out = conv(x)  # [B, d_model//num_branches, T]
            branch_outputs.append(out)

        # 在通道维度 (dim=1) 拼接
        y = torch.cat(branch_outputs, dim=1)  # [B, d_model, T]

        y = self.project(y) # 可选

        return y.transpose(1, 2)  # [B, T, D]

class LatentHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = latent_dim
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

        self.pos = PositionalEncoding(d_model, dropout, max_len=seq_len)
        if num_classes is not None:
            self.y_embed = nn.Linear(num_classes, d_model)
            # 新增融合层
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.y_embed = None
            self.fusion = None
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, n_heads, d_hid, dropout, batch_first=True, norm_first=True
                )
                for _ in range(dec_depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),  # 先在 d_model 维度内部混合一下
            nn.GELU(),  # 加入非线性
            nn.Linear(d_model, out_channels)  # 最终投影到信号通道
        )

    def forward(self, z: Tensor,  y_onehot: Optional[Tensor] = None) -> Tensor:
        """
        z: [B, latent_dim]
        return: [B, C, T]
        """
        B = z.size(0)
        x = self.proj(z)                 # [B, D]
        if y_onehot is not None and self.y_embed is not None:
            y_emb = self.y_embed(y_onehot)  # [B, D]
            x = torch.cat([x, y_emb], dim=1)  # [B, 2*D]
            x = self.fusion(x)  # [B, D]
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, D] 所有时间步共享这个latent
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.out(x)                  # [B, T, C]
        return x.transpose(1, 2)         # [B, C, T]

# from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.cfamg import HiddenNetworksOPtions
# class LatentGatedDualVAE(nn.Module):
#     """
#     Latent-Gated Dual-VAE for Minority-Aware Time-Series Generation
#
#     输入:  x: [B, C, T], y: [B] (int 或 long)
#     输出:  一个 dict, 里面有 recon, loss 各种项
#     """
#     def __init__(
#         self,
#         in_chans: int = 1,
#         seq_len: int = 200,
#         d_model: int = 64,
#         enc_depth: int = 2,
#         n_heads: int = 4,
#         d_hid: int = 128,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: int = 1,
#         latent_dim_global: int = 32,
#         latent_dim_class: int = 32,
#         minority_class_id: int = 1,
#         dropout: float = 0.1,
#         dec_depth: int = 2,
#         gate_hidden: int = 64,
#         align_lambda: float = 0.1,
#         num_classes: int = None,
#         cls_lambda: float = 0.1,
#         use_prototype: bool = False,
#         n_prototypes: int = 1,
#         proto_lambda: float = 0.1,
#         recon_metric: str = "mse",
#     ):
#         super().__init__()
#         self.in_chans = in_chans
#         self.seq_len = seq_len
#         self.minority_class_id = minority_class_id
#         self.align_lambda = align_lambda
#         self.recon_metric = recon_metric
#         hidden_dim = [32, 64, d_model]
#         dropout_list = [0.1, 0.1, 0.2]
#         latent_dim = d_model
#
#         self.embed = HiddenNetworksOPtions(input_dim=in_chans * seq_len,
#                                                      hidden_dim=hidden_dim,
#                                                      dropout_list=dropout_list,
#                                                      model_type='encoder')
#         self.encoder_hidden = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
#         if num_classes is not None:
#             latent_dim = latent_dim+16  # for y embedding
#         self.decoder = nn.Sequential(
#             HiddenNetworksOPtions(input_dim=latent_dim, hidden_dim=hidden_dim, dropout_list=dropout_list,
#                                   model_type='decoder'),
#             nn.Linear(hidden_dim[0] if isinstance(hidden_dim, list) else hidden_dim, in_chans * seq_len))
#
#
#
#         # majority global latent running mean (learnable center)
#         self.z_g_maj_mean = nn.Parameter(torch.zeros(1, latent_dim_global))
#         nn.init.normal_(self.z_g_maj_mean, mean=0.0, std=0.02)
#
#         # EMA buffer to keep a running estimate of majority global mean
#         self.register_buffer("z_g_maj_ema", torch.zeros(1, latent_dim_global))
#         self.register_buffer("z_g_maj_ema_inited", torch.tensor(False, dtype=torch.bool))
#         self.z_g_maj_momentum = 0.1
#
#
#         # 2) two latent heads
#         if num_classes is not None:
#             feature_dim = d_model + 16  # for y embedding
#         else:
#             feature_dim = d_model
#         self.global_head_p = LatentHead(feature_dim, latent_dim_global)
#         self.class_head_p = LatentHead(feature_dim, latent_dim_class)
#
#         self.global_head_n = LatentHead(feature_dim, latent_dim_global)
#         self.class_head_n = LatentHead(feature_dim, latent_dim_class)
#
#
#         # 3) gate: 用 minority 的 z_c 预测一个 [0,1] 的门控，决定 z_g_min 和 z_g_maj_mean 的混合
#         self.gate = nn.Sequential(
#             nn.Linear(latent_dim_class, gate_hidden),
#             nn.ReLU(),
#             nn.Linear(gate_hidden, latent_dim_global),
#             nn.Sigmoid(),
#         )
#
#         # 4) decoder
#         total_latent = latent_dim_global + latent_dim_class
#         # 5) classifier
#         self.num_classes = num_classes
#         self.cls_lambda = cls_lambda
#         # 6）Conditional vae
#         self.y_embed_latent = None
#         self.y_embed_dec = None
#
#         # ---- Learnable Minority Prototype(s) ----
#         self.use_prototype = use_prototype
#         self.n_prototypes = int(n_prototypes)
#         self.proto_lambda = float(proto_lambda)
#
#         if num_classes is not None:
#             # 用 class latent 做分类
#             self.classifier = nn.Sequential(
#                 nn.Linear(latent_dim_global, num_classes),
#                 nn.ReLU(),
#                 nn.Softmax(dim=1))
#             self.y_embed_latent = nn.Linear(num_classes, 16)  # 给 encoder latent 用
#             self.y_embed_dec = nn.Linear(num_classes, 16)  # 给 decoder 用
#         else:
#             self.classifier = None
#
#         if self.use_prototype:
#             if self.n_prototypes <= 1:
#                 # single learnable prototype vector for minority class latent (z_c space)
#                 self.proto_minority = nn.Parameter(torch.zeros(latent_dim_class))
#                 nn.init.normal_(self.proto_minority, mean=0.0, std=0.02)
#             else:
#                 # K learnable prototypes with soft assignment
#                 self.proto_minority = nn.Parameter(torch.zeros(self.n_prototypes, latent_dim_class))
#                 nn.init.normal_(self.proto_minority, mean=0.0, std=0.02)
#                 # temperature for softmax over prototypes
#                 self.register_buffer("proto_tau", torch.tensor(0.5))
#
#     # --------- helper ---------
#     @staticmethod
#     def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     @staticmethod
#     def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
#         # KL(q||p), p ~ N(0, I)
#         return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
#
#     # --------- forward parts ---------
#
#     def encode_feat(self, x: Tensor) -> Tensor:
#         # x: [B, C, T]
#         x = x.view(x.size(0), -1)
#         x = self.embed(x)  # [B, d_model]
#         return x
#
#     def encode_latent_branches(self, feat: Tensor, y: Optional[Tensor] = None) \
#             -> tuple[Tensor, Tensor, Tensor, Tensor]:
#         B = feat.size(0) # B, d_model
#         device = feat.device
#         if y is None:
#             # 没标签时退化用“positive 分支”或随便一个（建议统一用 min 分支）
#             mu_g, logvar_g = self.global_head_p(feat)
#             mu_c, logvar_c = self.class_head_p(feat)
#         else:
#             y = y.to(device)
#             is_pos = (y == self.minority_class_id)  # or y==1
#             is_neg = ~is_pos
#
#             G = self.global_head_p.fc_mu.out_features
#             C = self.class_head_p.fc_mu.out_features
#
#             mu_g = torch.empty(B, G, device=device)
#             logvar_g = torch.empty(B, G, device=device)
#             mu_c = torch.empty(B, C, device=device)
#             logvar_c = torch.empty(B, C, device=device)
#
#             # positive branch
#             if is_pos.any():
#                 feat_pos = feat[is_pos]
#                 mu_g_pos, logvar_g_pos = self.global_head_p(feat_pos)
#                 mu_c_pos, logvar_c_pos = self.class_head_p(feat_pos)
#                 mu_g[is_pos] = mu_g_pos
#                 logvar_g[is_pos] = logvar_g_pos
#                 mu_c[is_pos] = mu_c_pos
#                 logvar_c[is_pos] = logvar_c_pos
#
#             # negative branch
#             if is_neg.any():
#                 feat_neg = feat[is_neg]
#                 mu_g_neg, logvar_g_neg = self.global_head_n(feat_neg)
#                 mu_c_neg, logvar_c_neg = self.class_head_n(feat_neg)
#                 mu_g[is_neg] = mu_g_neg
#                 logvar_g[is_neg] = logvar_g_neg
#                 mu_c[is_neg] = mu_c_neg
#                 logvar_c[is_neg] = logvar_c_neg
#
#         return mu_g, logvar_g, mu_c, logvar_c
#
#     def encode(self,x: Tensor, y: Optional[Tensor] = None) \
#             -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         feat = self.encode_feat(x)  # [B, d_model]
#         if y is not None and self.num_classes is not None and self.y_embed_latent is not None:
#             y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
#             y_cond = self.y_embed_latent(y_onehot)  # [B, D]
#             feat = torch.cat((feat, y_cond), dim=1)
#         mu_g, logvar_g, mu_c, logvar_c = self.encode_latent_branches(feat, y)
#         return feat, mu_g, logvar_g, mu_c, logvar_c
#
#     def cdecoder(self, z_full, y_onehot=None):
#         if y_onehot is not None and self.num_classes is not None and self.y_embed_dec is not None:
#             y_cond = self.y_embed_dec(y_onehot)  # [B, D]
#             z_full = torch.cat((z_full, y_cond), dim=1)
#         z_full = z_full.float()
#         recon = self.decoder(z_full)  # [B, C, T]
#         recon = recon.view(recon.size(0), self.in_chans, self.seq_len)
#         return recon
#     def forward(
#             self,
#             x: Tensor,
#             y: Optional = None,
#     ) -> dict[str, Tensor]:
#         """
#         训练/推理统一入口
#         """
#         recon_x = None
#
#         if isinstance(x, list):
#             recon_x = x[1]
#             x = x[0]
#
#         device = x.device
#         feat, mu_g, logvar_g, mu_c, logvar_c = self.encode(x, y)
#         z_g = self.reparameterize(mu_g, logvar_g)   # [B, G]
#         z_c = self.reparameterize(mu_c, logvar_c)   # [B, C]
#
#         disentangle_loss = torch.tensor(0.0, device=device)
#         if y is None:
#             # 没给标签就当全是 minority，用自己的 z_g
#             z_g_final = z_g
#             align_loss = torch.tensor(0.0, device=device)
#         else:
#             y = y.to(device)
#             is_min = (y == self.minority_class_id)          # [B]
#             is_maj = ~is_min
#             # 如果 batch 里一个多数都没有，那就退化成自己用自己
#             if is_maj.any():
#                 z_g_maj = z_g[is_maj]                      # [B_maj, G]
#                 # batch-wise majority mean (no grad, for stats only)
#                 z_c_maj = z_c[is_maj]
#                 disentangle_loss = cross_covariance_loss(z_g_maj, z_c_maj)
#
#                 batch_mean = z_g_maj.mean(dim=0, keepdim=True).detach()  # [1, G]
#
#                 # update EMA buffer during training
#                 if self.training:
#                     if not self.z_g_maj_ema_inited:
#                         self.z_g_maj_ema.copy_(batch_mean)
#                         self.z_g_maj_ema_inited.fill_(True)
#                         # also use this to initialize the learnable center once
#                         with torch.no_grad():
#                             self.z_g_maj_mean.copy_(batch_mean)
#                     else:
#                         m = self.z_g_maj_momentum
#                         self.z_g_maj_ema.mul_(1.0 - m).add_(m * batch_mean)
#
#                 # use the learnable majority center as z_g_maj_mean
#                 # (fall back to current batch mean before EMA is inited)
#                 if self.z_g_maj_ema_inited:
#                     z_g_maj_mean = self.z_g_maj_mean  # [1, G], trainable
#                 else:
#                     z_g_maj_mean = batch_mean  # startup phase
#
#
#                 # minority 部分用 gate 融合
#                 z_g_final = z_g.clone()
#
#                 if is_min.any():
#                     z_c_min = z_c[is_min]
#                     z_g_min = z_g[is_min]                  # [B_min, G]
#                     gate = self.gate(z_c_min)              # [B_min, G], in [0,1]
#                     z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj_mean
#                     z_g_final[is_min] = z_g_mix
#                     # 对齐损失: 让 minority 的 z_g 靠近 majority 的均值
#                     align_loss = (z_g_mix - z_g_maj_mean.detach()).pow(2).mean()
#                 else:
#                     align_loss = torch.tensor(0.0, device=device)
#             else:
#                 # 全是 minority，没法对齐
#                 z_g_final = z_g
#                 align_loss = torch.tensor(0.0, device=device)
#
#         # 拼 latent → decode
#         z_full = torch.cat([z_g_final, z_c], dim=1)         # [B, G+C]
#
#         if y is not None and self.num_classes is not None:
#             y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
#         else:
#             y_onehot = None
#         recon = self.cdecoder(z_full, y_onehot=y_onehot)                        # [B, C, T]
#
#         # reconstruction loss
#         # 如果你的数据有 NaN，可以加 nan_to_num
#         recon_loss = 0
#         if recon_x is None:
#             recon_x = x
#         recon_x.to(device)
#         if self.recon_metric == 'soft_dtw':
#             from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
#             recon_loss, _, _ = dilate_loss(outputs=recon, targets=recon_x, device=x.device, alpha=1)
#         elif self.recon_metric == 'dilate':
#             from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
#             recon_loss, _, _ = dilate_loss(outputs=recon, targets=recon_x, device=x.device, alpha=0.5)
#         elif self.recon_metric == 'mse':
#             recon_loss = ((recon - recon_x) ** 2).mean()
#         else:
#             raise NotImplementedError
#
#         kl_g = self.kl_loss(mu_g, logvar_g)
#         kl_c = self.kl_loss(mu_c, logvar_c)
#
#         cls_loss = torch.tensor(0.0, device=device)
#         cls_logits = None
#         if self.classifier is not None and y is not None:
#             is_min = (y == self.minority_class_id)
#             is_maj = ~is_min
#
#             if is_min.any() and is_maj.any():
#                 # 同时有少数类和多数类样本才算分类损失
#                 logits_base = self.classifier(z_g_final)  # [B, num_classes]
#                 # 默认先用 base 来算 loss
#                 cls_loss = F.cross_entropy(logits_base, y)
#                 cls_logits = logits_base  # 用于 pl 里算 F1 / Acc
#                 z_c_min = z_c[is_min]  # [B_min, C]
#
#                 # majority 的 global pool：优先用 batch 中的多数样本 global
#                 z_g_maj_pool = z_g[is_maj]  # [B_maj, G]
#
#                 B_min = int(z_c_min.size(0))
#                 B_maj = int(z_g_maj_pool.size(0))
#
#
#         if not self.z_g_maj_ema_inited:
#             loss_maj_center = torch.tensor(0.0, device=device)
#         else:
#             loss_maj_center = F.mse_loss(self.z_g_maj_mean, self.z_g_maj_ema.detach())
#
#         out = {
#             "recon_loss": recon_loss,
#             "kl_g": kl_g,
#             "kl_c": kl_c,
#             "disentangle_loss": disentangle_loss,
#             "align_loss": align_loss,
#             "loss_center": loss_maj_center,
#             "cls_loss": cls_loss,
#             "cls_logits": cls_logits,
#             "recon": recon,
#             "z_g": z_g,
#             "z_c": z_c,
#         }
#
#         # expose attention prototype if computed
#         if self.use_prototype and hasattr(self, "n_prototypes") and self.n_prototypes > 1 and 'proto_attn' not in locals():
#             # nothing to add
#             pass
#
#         return out
#     # ---- generation
#     @torch.no_grad()
#     def generate_vae_prior(self, x_min: Tensor, alpha: Optional = None,) -> Tensor:
#         """
#         x_min [B, C, T]
#         return: [B, C, T] generated
#         """
#         y_min = torch.ones(x_min.shape[0], device=x_min.device).long()
#
#         _, mu_g_min, logvar_g_min, mu_c_min, logvar_c_min = self.encode(x_min, y=y_min)
#
#         z_g_min = self.reparameterize(mu_g_min, logvar_g_min)
#         z_c_min = self.reparameterize(mu_c_min, logvar_c_min)
#         z_c_min_prior = torch.randn_like(z_c_min)
#         dim_c = z_c_min.size(1)
#         if alpha is None:
#             alpha = 0.15
#             alpha = torch.full((x_min.size(0), dim_c), alpha, device=x_min.device)
#         else:
#             alpha = torch.tensor(alpha, device=x_min.device) # [B]
#             alpha = alpha.unsqueeze(1).expand(-1, dim_c)
#         z_g_min = z_g_min
#         z_c_min = (1.0 - alpha) * z_c_min + alpha * z_c_min_prior
#         if self.z_g_maj_ema_inited:
#             z_g_maj = self.z_g_maj_mean  # [1, G], trainable
#             z_g_mix = z_g_maj.expand(x_min.size(0), -1)
#             z_full = torch.cat([z_g_mix, z_c_min], dim=1)
#         else:
#             z_full = torch.cat([z_g_min, z_c_min], dim=1)
#         y = y_min
#         y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
#         x_gen = self.cdecoder(z_full, y_onehot=y_onehot)
#         return x_gen
#
#
#     # 一个简单的生成接口：给 minority 的一条 x，同时给一条 majority 的 x，当场生成“混合版”
#     @torch.no_grad()
#     def generate_from_pair(self, x_min: Tensor, x_maj: Tensor, use_y:bool=True) -> Tensor:
#         """
#         x_min, x_maj: [1, C, T]
#         return: [1, C, T] generated
#         """
#         if use_y:
#             y_min = torch.ones(x_min.shape[0], device=x_min.device).long()
#             y_maj = torch.zeros(x_maj.shape[0], device=x_maj.device).long()
#         else:
#             y_min = None
#             y_maj = None
#         _, mu_g_min, logvar_g_min, mu_c_min, logvar_c_min = self.encode(x_min, y=y_min)
#         _, mu_g_maj, logvar_g_maj, _, _ = self.encode(x_maj, y=y_maj)
#
#         z_g_min = self.reparameterize(mu_g_min, logvar_g_min)
#         z_g_maj = self.reparameterize(mu_g_maj, logvar_g_maj)
#         z_c_min = self.reparameterize(mu_c_min, logvar_c_min)
#
#         gate = self.gate(z_c_min)  # [1, G]
#         z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj
#
#         z_full = torch.cat([z_g_mix, z_c_min], dim=1)
#         if use_y:
#             y = y_min
#             y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
#         else:
#             y_onehot = None
#         x_gen = self.cdecoder(z_full, y_onehot=y_onehot)
#         return x_gen
#
#     @torch.no_grad()
#     def generate_from_latent_smote(
#         self,
#         x_min1: Tensor,
#         x_min2: Tensor,
#         alpha: Optional[float] = None,
#         num_samples: int = 1,
#     ) -> Tensor:
#         """
#         方式三：在 minority 的 latent 空间做 SMOTE 风格插值生成。
#
#         Args:
#             x_min1: 第一条少数类序列，[B, C, T] 或 [1, C, T]
#             x_min2: 第二条少数类序列，[B, C, T] 或 [1, C, T]，形状需与 x_min1 相同
#             alpha: 若给定，则使用固定插值系数 alpha ∈ [0,1]；
#                    若为 None 且 num_samples==1，则用 0.5；
#                    若为 None 且 num_samples>1，则每个样本随机采样 alpha ~ U(0,1)。
#             num_samples: 生成多少条插值样本。
#
#         Returns:
#             - 若 num_samples==1: 返回 [B, C, T]
#             - 若 num_samples>1: 返回 [num_samples, B, C, T]
#         """
#         device = x_min1.device
#         y_min1 = torch.ones(x_min1.shape[0], device=x_min1.device).long()
#         y_min2 = torch.ones(x_min2.shape[0], device=x_min2.device).long()
#         # 编码两条少数类样本
#         _, mu_g1, logvar_g1, mu_c1, logvar_c1 = self.encode(x_min1,y=y_min1)
#         _, mu_g2, logvar_g2, mu_c2, logvar_c2 = self.encode(x_min2,y=y_min2)
#
#         z_g1 = self.reparameterize(mu_g1, logvar_g1)  # [B, G]
#         z_g2 = self.reparameterize(mu_g2, logvar_g2)  # [B, G]
#         z_c1 = self.reparameterize(mu_c1, logvar_c1)  # [B, C]
#         z_c2 = self.reparameterize(mu_c2, logvar_c2)  # [B, C]
#
#         B = z_g1.size(0)
#
#         if num_samples == 1:
#             if alpha is None:
#                 alpha_val = 0.5
#             else:
#                 alpha_val = float(alpha)
#             a = torch.full((B, 1), alpha_val, device=device)
#             z_g_mix = (1.0 - a) * z_g1 + a * z_g2
#             z_c_mix = (1.0 - a) * z_c1 + a * z_c2
#             if self.z_g_maj_ema_inited:
#                 z_g_maj = self.z_g_maj_mean  # [1, G], trainable
#                 # print('Using majority prototype for generation.')
#                 gate = self.gate(z_c_mix)  # [1, G]
#                 z_g_mix = gate * z_g_mix + (1.0 - gate) * z_g_maj
#
#             z_full = torch.cat([z_g_mix, z_c_mix], dim=1)   # [B, G+C]
#             y_onehot = F.one_hot(y_min1, num_classes=self.num_classes).float()
#             x_gen = self.cdecoder(z_full, y_onehot=y_onehot)
#             return x_gen
#         else:
#             raise NotImplementedError("num_samples > 1 is not implemented yet.")
#             # xs = []
#             # for _ in range(num_samples):
#             #     if alpha is None:
#             #         a = torch.rand(B, 1, device=device)
#             #     else:
#             #         a = torch.full((B, 1), float(alpha), device=device)
#             #     z_g_mix = (1.0 - a) * z_g1 + a * z_g2
#             #     z_c_mix = (1.0 - a) * z_c1 + a * z_c2
#             #     z_full = torch.cat([z_g_mix, z_c_mix], dim=1)
#             #     x_gen_k = self.cdecoder(z_full)  # [B, C, T]
#             #     xs.append(x_gen_k)
#
#             # return torch.stack(xs, dim=0)  # [num_samples, B, C, T]
#
#     @torch.no_grad()
#     def generate_from_prototype(
#         self, x_min: Tensor) -> Tensor:
#         """
#         x_min [1, C, T]
#         return: [1, C, T] generated
#         """
#         y_min = torch.ones(x_min.shape[0], device=x_min.device).long()
#
#         _, mu_g_min, logvar_g_min, mu_c_min, logvar_c_min = self.encode(x_min, y=y_min)
#
#         z_g_min = self.reparameterize(mu_g_min, logvar_g_min)
#         z_c_min = self.reparameterize(mu_c_min, logvar_c_min)
#         if self.z_g_maj_ema_inited:
#             z_g_maj = self.z_g_maj_mean  # [1, G], trainable
#             # print('Using majority prototype for generation.')
#             gate = self.gate(z_c_min)  # [1, G]
#             z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj
#
#             z_full = torch.cat([z_g_mix, z_c_min], dim=1)
#         else:
#             z_full = torch.cat([z_g_min, z_c_min], dim=1)
#         y = y_min
#         y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
#         x_gen = self.cdecoder(z_full, y_onehot=y_onehot)
#         return x_gen


class LatentGatedDualVAE(nn.Module):
    """
    Latent-Gated Dual-VAE for Minority-Aware Time-Series Generation
    Modified: CVAE fusion changed from Addition to Concatenation.
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

        # 1) encoder: conv -> PE -> transformer
        self.embed = MultiScaleConvEmbedder(
            in_channels=in_chans,
            d_model=d_model,
        )
        self.pos = PositionalEncoding(d_model, dropout, max_len=seq_len+1)

        self.encoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(enc_depth)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # majority global latent running mean (learnable center)
        self.z_g_maj_mean = nn.Parameter(torch.zeros(1, latent_dim_global))
        nn.init.normal_(self.z_g_maj_mean, mean=0.0, std=0.02)

        # EMA buffer to keep a running estimate of majority global mean
        self.register_buffer("z_g_maj_ema", torch.zeros(1, latent_dim_global))
        self.register_buffer("z_g_maj_ema_inited", torch.tensor(False, dtype=torch.bool))
        self.z_g_maj_momentum = 0.1

        # -----------------------------------------------------------
        # [MODIFIED] 2) two latent heads & Conditional Embedding Logic
        # -----------------------------------------------------------

        # 预先计算 LatentHead 的输入维度
        # 原始逻辑是 add，所以维度是 d_model
        # 修改为 cat，维度变为 d_model + embedding_dim (此处假设 embedding 维度也是 d_model)
        head_in_dim = d_model
        if num_classes is not None:
            # 因为下面 y_embed_latent 输出维度设为了 d_model，所以拼接后是 2 * d_model
            head_in_dim = d_model + d_model

        self.global_head_p = LatentHead(head_in_dim, latent_dim_global)
        self.class_head_p = LatentHead(head_in_dim, latent_dim_class)

        self.global_head_n = LatentHead(head_in_dim, latent_dim_global)
        self.class_head_n = LatentHead(head_in_dim, latent_dim_class)

        # 3) gate: 用 minority 的 z_c 预测一个 [0,1] 的门控
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

        # 5) classifier & Conditional Embeddings
        self.num_classes = num_classes
        self.cls_lambda = cls_lambda

        self.y_embed_latent = None
        self.y_embed_dec = None

        if num_classes is not None:
            # 用 class latent 做分类
            self.classifier = None #nn.Linear(d_model, num_classes)
            # 给 encoder latent 用，输出维度保持 d_model
            self.y_embed_latent = nn.Linear(num_classes, d_model)
            # 给 decoder 用
            self.y_embed_dec = nn.Linear(num_classes, d_model)
        else:
            self.classifier = None

    # --------- helper ---------
    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # --------- forward parts ---------

    def encode_feat(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x = self.embed(x)  # [B, T', D]
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos(x)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.enc_norm(x)  # [B, T', D]
        feat = x[:, 1:].mean(dim=1)  # [B, D]

        return feat

    def encode_latent_branches(self, feat: Tensor, y: Optional[Tensor] = None) \
            -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B = feat.size(0)
        device = feat.device

        # 注意：这里的 feat 已经是拼接过 label embedding 的特征 (如果 y 存在)
        # 所以 feat 的维度可能是 d_model (无 y) 或者 2*d_model (有 y)

        if y is None:
            mu_g, logvar_g = self.global_head_p(feat)
            mu_c, logvar_c = self.class_head_p(feat)
        else:
            y = y.to(device)
            is_pos = (y == self.minority_class_id)
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

    def encode(self, x: Tensor, y: Optional[Tensor] = None) \
            -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        feat = self.encode_feat(x)  # [B, d_model]
        #  stop conditional
        if y is not None and self.num_classes is not None and self.y_embed_latent is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_cond = self.y_embed_latent(y_onehot)  # [B, d_model]

            # -----------------------------------------------------------
            # [MODIFIED] Concatenation instead of Addition
            # -----------------------------------------------------------
            # feat = feat + y_cond  <-- OLD
            feat = torch.cat([feat, y_cond], dim=1)  # [B, d_model * 2]

        mu_g, logvar_g, mu_c, logvar_c = self.encode_latent_branches(feat, y)
        return feat, mu_g, logvar_g, mu_c, logvar_c

    # ... forward, cdecoder, generate_* methods 保持不变 (只要调用 encode 即可) ...
    # 为了完整性，这里保留 forward 方法，因为它依赖 encode
    def forward(
            self,
            x: Tensor,
            y: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        训练/推理统一入口
        """
        device = x.device
        feat_x, mu_g, logvar_g, mu_c, logvar_c = self.encode(x, y)
        z_g = self.reparameterize(mu_g, logvar_g)  # [B, G]
        z_c = self.reparameterize(mu_c, logvar_c)  # [B, C]

        disentangle_loss = torch.tensor(0.0, device=device)
        if y is None:
            # 没给标签就当全是 minority，用自己的 z_g
            z_g_final = z_g
            align_loss = torch.tensor(0.0, device=device)
        else:
            y = y.to(device)
            is_min = (y == self.minority_class_id)  # [B]
            is_maj = ~is_min
            # 如果 batch 里一个多数都没有，那就退化成自己用自己
            if is_maj.any():
                z_g_maj = z_g[is_maj]  # [B_maj, G]
                # batch-wise majority mean (no grad, for stats only)
                z_c_maj = z_c[is_maj]
                disentangle_loss = cross_covariance_loss(z_g_maj, z_c_maj)

                batch_mean = z_g_maj.mean(dim=0, keepdim=True).detach()  # [1, G]

                # update EMA buffer during training
                if self.training:
                    if not self.z_g_maj_ema_inited:
                        self.z_g_maj_ema.copy_(batch_mean)
                        self.z_g_maj_ema_inited.fill_(True)
                        with torch.no_grad():
                            self.z_g_maj_mean.copy_(batch_mean)
                    else:
                        m = self.z_g_maj_momentum
                        self.z_g_maj_ema.mul_(1.0 - m).add_(m * batch_mean)

                if self.z_g_maj_ema_inited:
                    z_g_maj_mean = self.z_g_maj_mean  # [1, G], trainable
                else:
                    z_g_maj_mean = batch_mean  # startup phase

                # minority 部分用 gate 融合
                z_g_final = z_g.clone()

                if is_min.any():
                    z_c_min = z_c[is_min]
                    z_g_min = z_g[is_min]  # [B_min, G]
                    gate = self.gate(z_c_min)  # [B_min, G], in [0,1]
                    z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj_mean
                    z_g_final[is_min] = z_g_mix
                    # 对齐损失
                    align_loss = (z_g_mix - z_g_maj_mean.detach()).pow(2).mean()
                else:
                    align_loss = torch.tensor(0.0, device=device)
            else:
                z_g_final = z_g
                align_loss = torch.tensor(0.0, device=device)

        # 拼 latent -> decode
        z_full = torch.cat([z_g_final, z_c], dim=1)  # [B, G+C]

        if y is not None and self.num_classes is not None:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_onehot = None
        recon = self.decoder(z_full, y_onehot=y_onehot)

        # reconstruction loss
        recon_loss = 0
        recon_loss_mse = ((recon - x) ** 2).mean()
        if self.recon_metric == 'soft_dtw':
            from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
            recon_loss, _, _ = dilate_loss(outputs=recon, targets=x, device=x.device, alpha=1)
        elif self.recon_metric == 'dilate':
            from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.loss.dilate_loss import dilate_loss
            recon_loss, _, _ = dilate_loss(outputs=recon, targets=x, device=x.device, alpha=0.5)
        elif self.recon_metric == 'mse':
            recon_loss = recon_loss_mse
        else:
            raise NotImplementedError
        metric_mse_value = recon_loss_mse.detach().item()

        kl_g = self.kl_loss(mu_g, logvar_g)
        kl_c = self.kl_loss(mu_c, logvar_c)

        cls_loss = torch.tensor(0.0, device=device)
        cls_logits = None
        if self.classifier is not None and y is not None:
            # 1. 基础分类 Loss (对原始样本)
            logits_base = self.classifier(feat_x)
            cls_loss_base = F.cross_entropy(logits_base, y)

            # 2. [关键修改] 针对 Batch 内样本做 Latent Mixup 增强
            # 目的：防止模型死记硬背重复的样本，强迫它理解样本之间的连续性

            # 生成随机排列索引
            lam = torch.distributions.Beta(0.5, 0.5).sample((feat_x.size(0), 1)).to(device)
            perm = torch.randperm(feat_x.size(0), device=device)

            feat_mix = lam * feat_x + (1 - lam) * feat_x[perm]
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            y_mix = lam * y_onehot + (1 - lam) * y_onehot[perm]

            logits_mix = self.classifier(feat_mix)
            cls_loss_mix = -torch.sum(y_mix * F.log_softmax(logits_mix, dim=1), dim=1).mean()

            cls_loss = cls_loss_base + cls_loss_mix

            # 用于显示的 logits 依然是 base
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
            "recon_loss_mse": metric_mse_value,
        }
        return out

    @torch.no_grad()
    def feature_extract(self, x: Tensor) -> Tensor:
        feat = self.encode_feat(x)
        return feat

    # @torch.no_grad()
    # def generate_vae_prior(self, x_min: Tensor, num_variations: int = 1, alpha: float = 0.5) -> Tensor:
    #     """
    #     [修改版] 生成策略：Latent Space Mixup (插值)
    #
    #     Args:
    #         x_min: 一批真实的少数类样本 [B, C, T]
    #         num_variations: 每个样本生成几个变体 (当前代码建议设为 1，保持 Batch Size 一致)
    #         alpha: Mixup 的强度。
    #                alpha 接近 0: 生成结果很像原始 x_min
    #                alpha 接近 0.5: 生成结果是两个样本的中间形态 (创新性最强)
    #                alpha 接近 1.0: 生成结果很像被打乱的另一个样本
    #     """
    #     device = x_min.device
    #     B = x_min.size(0)
    #
    #     # 1. 强制 Label 为少数类
    #     y_min = torch.full((B,), self.minority_class_id, device=device, dtype=torch.long)
    #
    #     # 2. 编码得到 Latent 分布
    #     # 注意：这里我们拿到的 mu 和 logvar 是这一批样本的
    #     _, mu_g, logvar_g, mu_c, logvar_c = self.encode(x_min, y=y_min)
    #
    #     # 3. 重参数化得到 z
    #     z_g = self.reparameterize(mu_g, logvar_g)
    #     z_c = self.reparameterize(mu_c, logvar_c)
    #
    #     # 4. [核心修改] 执行 Latent Mixup
    #     # 也就是：不加噪声，而是找“另一个少数类兄弟”来杂交
    #
    #     # 生成打乱的索引 (Shuffle)
    #     perm = torch.randperm(B, device=device)
    #
    #     # 采样插值系数 lambda
    #     # 这里的 alpha 控制 Beta 分布的形状。
    #     # 如果你希望生成多样性大，用 Beta(0.5, 0.5)
    #     # 如果你希望保守一点，直接用固定的数值，比如 0.2
    #
    #     # 方案 A: 随机插值 (推荐)
    #     lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(device)
    #
    #     # 方案 B: 固定插值 (如果你想控制变量调试)
    #     # lam = torch.full((B, 1), 0.5, device=device)
    #
    #     # 执行混合
    #     z_g_mix = lam * z_g + (1 - lam) * z_g[perm]
    #     z_c_mix = lam * z_c + (1 - lam) * z_c[perm]
    #
    #     # 5. [关键] Gate 的处理
    #     # 训练时我们用了 Gate 融合 Majority Mean，是为了让 loss 下降。
    #     # 但在生成少数类时，如果 Gate 打开，生成的波形会像多数类，这会导致下游分类器 F1 降低。
    #     # 建议：强制关闭 Gate，或者只保留很小的比例。
    #
    #     use_gate = False  # <--- 建议设为 False，完全信任少数类自己的特征
    #
    #     if use_gate and self.z_g_maj_ema_inited:
    #         gate = self.gate(z_c_mix)
    #         z_g_maj = self.z_g_maj_mean
    #         z_g_final = gate * z_g_mix + (1.0 - gate) * z_g_maj
    #     else:
    #         # 纯正的少数类特征
    #         z_g_final = z_g_mix
    #
    #     # 6. 拼接并解码
    #     z_full = torch.cat([z_g_final, z_c_mix], dim=1)
    #
    #     y_onehot = F.one_hot(y_min, num_classes=self.num_classes).float()
    #
    #     x_gen = self.decoder(z_full, y_onehot=y_onehot)
    #
    #     return x_gen
    @torch.no_grad()
    def generate_vae_prior(self, x_min: Tensor, alpha: Optional[float] = None, ) -> Tensor:
        # 复用原有逻辑
        y_min = torch.ones(x_min.shape[0], device=x_min.device).long()
        _, mu_g_min, logvar_g_min, mu_c_min, logvar_c_min = self.encode(x_min, y=y_min)
        z_g_min = self.reparameterize(mu_g_min, logvar_g_min)
        z_c_min = self.reparameterize(mu_c_min, logvar_c_min)
        z_g_min_prior = torch.randn_like(z_g_min)
        z_c_min_prior = torch.randn_like(z_c_min)
        if alpha is None:
            alpha_val = 0.15
        else:
            alpha_val = float(alpha)
        z_g_min = (1.0 - alpha_val) * z_g_min + alpha_val * z_g_min_prior
        z_c_min = (1.0 - alpha_val) * z_c_min + alpha_val * z_c_min_prior
        if self.z_g_maj_ema_inited:
            z_g_maj = self.z_g_maj_mean
            gate = self.gate(z_c_min)
            z_g_mix = gate * z_g_min + (1.0 - gate) * z_g_maj
            z_full = torch.cat([z_g_mix, z_c_min], dim=1)
        else:
            z_full = torch.cat([z_g_min, z_c_min], dim=1)
        y = y_min
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        x_gen = self.decoder(z_full, y_onehot=y_onehot)
        return x_gen
