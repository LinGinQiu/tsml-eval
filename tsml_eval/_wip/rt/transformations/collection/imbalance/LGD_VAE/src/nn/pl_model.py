import torch
import torch.nn as nn
import torch.nn.functional as F
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.model import LatentGatedDualVAE
import lightning.pytorch as pl
import torchmetrics
from typing import Tuple
from torchmetrics import Accuracy, F1Score, Recall
from TimesNet.models import TimesNet
class TSQualityClassifier(pl.LightningModule):
    def __init__(self, input_channels, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 定义特征提取器
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128,  64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        # 初始化评估指标
        # num_classes=2 时，task 可以设为 'multiclass' 也可以设为 'binary'
        # 这里用 multiclass 确保通用性
        task = "multiclass"
        self.acc_metric = Accuracy(task=task, num_classes=num_classes)
        self.f1_macro = F1Score(task=task, num_classes=num_classes, average='macro')
        self.recall_per_class = Recall(task=task, num_classes=num_classes, average='none')


    def forward(self, x):
        if x.ndim == 2:  # 如果是 (Batch, Length) 自动补齐通道维度
            x = x.unsqueeze(1)
        elif x.shape[1] != self.hparams.input_channels:
            x = x.transpose(1, 2)

        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.acc_metric(preds, y)
        f1 = self.f1_macro(preds, y)
        rec = self.recall_per_class(preds, y)
        g_means = torch.prod(rec).pow(1 / len(rec))

        metrics = {"val_loss": loss, "val_acc": acc, "val_f1_macro": f1, "val_g_means": g_means}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# define the LightningModule


# --- 核心辅助函数 ---
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


# --- 简化版 Inception 块 (如果你有现成的可以 import) ---
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.kernels = nn.ModuleList()
        for i in range(num_kernels):
            self.kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        res_list = [kernel(x) for kernel in self.kernels]
        return torch.mean(torch.stack(res_list, dim=-1), dim=-1)


# --- TimesBlock 核心 ---
class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, seq_len, top_k, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, (length - self.seq_len), N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        return torch.sum(res * period_weight, -1) + x


# --- 最终分类器模块 ---
class TimesNetQualityClassifier(pl.LightningModule):
    def __init__(self, input_channels, seq_len, num_classes=2, d_model=64, d_ff=256, top_k=5, e_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 1. 线性投影进入特征空间 (代替 DataEmbedding 以简化)
        self.enc_embedding = nn.Linear(input_channels, d_model)

        # 2. 堆叠 TimesBlock
        self.model = nn.ModuleList([
            TimesBlock(d_model, d_ff, seq_len, top_k, num_kernels=6)
            for _ in range(e_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)

        # 3. 输出层
        self.projection = nn.Linear(d_model * seq_len, num_classes)

        # 4. 指标
        task = "multiclass"
        self.acc_metric = Accuracy(task=task, num_classes=num_classes)
        self.f1_macro = F1Score(task=task, num_classes=num_classes, average='macro')
        self.recall_per_class = Recall(task=task, num_classes=num_classes, average='none')

    def forward(self, x):
        # 确保输入是 [B, T, C]
        if x.shape[1] == self.hparams.input_channels and x.shape[2] != self.hparams.input_channels:
            x = x.transpose(1, 2)

        # Embedding
        enc_out = self.enc_embedding(x)  # [B, T, d_model]

        # TimesNet 变换
        for i in range(self.hparams.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Classification Head
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # Flatten
        return self.projection(output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.acc_metric(preds, y)
        f1 = self.f1_macro(preds, y)
        rec = self.recall_per_class(preds, y)
        g_means = torch.prod(rec).pow(1 / len(rec))

        metrics = {"val_loss": loss, "val_acc": acc, "val_f1_macro": f1, "val_g_means": g_means}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return metrics

    def on_validation_epoch_end(self):
        self.acc_metric.reset()
        self.f1_macro.reset()
        self.recall_per_class.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
class LitAutoEncoder(pl.LightningModule):
    def __init__(self,
        in_chans=51,
        seq_len = 100,
        embed_dim=64,
        depth=2,
        num_heads=4,
        latent_dim_global=32,
        latent_dim_class=32,
        minority_class_id=1,
        dropout = 0.1,
        decoder_depth=2,
        gate_hidden= 64,
        align_lambda = 0.10,  # \lambda_align，用于 \mathcal{L}_{align}
        cls_lambda = 0.1,  # 若不开类别监督，这里为 0
        kl_g_lambda = 1.00,  # 全局 KL 系数（beta-VAE 风格）
        kl_c_lambda = 1.00,  # 类别 KL 系数
        recon_lambda = 1.00,  # 重构项系数
        disentangle_lambda = 0.10,
        center_lambda = 0.0,
        cls_embed=False,
        weights = None,
        d_hid = 128,
        kernel_size = 3,
        stride = 1,
        padding =1,
        recon_metric = 'mse',
        val_data = None,
        oracle_model: nn.Module = None,
        lr = 1e-3,
        mean_ = None,
        std_ = None
                 ):

        super().__init__()
        self.model = LatentGatedDualVAE(
            in_chans=in_chans,
            seq_len=seq_len,
            d_model=embed_dim,
            enc_depth=depth,
            n_heads=num_heads,
            d_hid=d_hid,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            latent_dim_global=latent_dim_global,
            latent_dim_class=latent_dim_class,
            minority_class_id = minority_class_id,
            dropout=dropout,
            dec_depth=decoder_depth,
            gate_hidden = gate_hidden,
            num_classes=(len(weights) if cls_embed and weights is not None else None),
            recon_metric=recon_metric
        )

        self.align_lambda = float(align_lambda)
        self.cls_lambda = float(cls_lambda)
        self.disentangle_lambda = float(disentangle_lambda)
        self.center_lambda = float(center_lambda)
        self.kl_g_lambda = float(kl_g_lambda)
        self.kl_c_lambda = float(kl_c_lambda)
        self.recon_lambda = float(recon_lambda)
        self.warmup_epochs = 5.0
        self.minority_class_id = minority_class_id
        self.input_channels = in_chans
        self.validation_step_outputs = []
        self.val_data = val_data
        self.train_data = []
        self.train_data_sample = True
        self.oracle = oracle_model
        self.mean_ = mean_
        self.std_ = std_
        if self.oracle:
            self.oracle.eval()
            for param in self.oracle.parameters():
                param.requires_grad = False  # 确保不被误训练
        if cls_embed and weights is not None:
            num_c = len(weights)
            print(weights)
            self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_c, average="macro")
            self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_c)
            # self.valid_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_c, average="macro")
            self.valid_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_c, average=None)
        else:
            self.train_f1 = None
            self.valid_acc = None
            self.valid_f1 = None

        self.lr = lr
        self.fig = None
        self.save_hyperparameters()

    def on_train_epoch_start(self):
        print(f"Starting epoch {self.current_epoch}")
        current_epoch = self.current_epoch

        # 2. 获取 train_dataloader
        # 注意：self.trainer.train_dataloader 通常是当前正在使用的 loader
        loader = self.trainer.train_dataloader

        # 3. 找到 sampler 并设置 epoch
        # 有些情况下 loader 可能被 Lightning 包装过，所以最好加个 hasattr 判断
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(current_epoch)

            # (可选) 打印一下确认切换状态
            if current_epoch == loader.sampler.switch_epoch:
                print(f"\n[Info] Epoch {current_epoch}: Switching to FULL weighted sampling!")

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            batch,y = batch
        else:
            y = None

        # 1) 当前 epoch 下的有效权重
        turn_off_dilate =False
        recon_w, kl_g_w, kl_c_w, align_w, disentangle_w, center_w, cls_w,turn_off_dilate = self._current_loss_weights()
        # === 2) 把有效权重传进模型 ===
        out = self.model(batch, y, turn_off_dilate=turn_off_dilate)

        recon_loss = out.get("recon_loss", 0.0)
        kl_g = out.get("kl_g", 0.0)
        kl_c = out.get("kl_c", 0.0)
        align_loss = out.get("align_loss", 0.0)
        disentangle_loss = out.get("disentangle_loss", 0.0)
        loss_center = out.get("loss_center", 0.0)
        cls_loss = out.get("cls_loss", None)

        # 3) 在 PL 里组合 total loss
        loss = recon_w * recon_loss + kl_g_w * kl_g + kl_c_w * kl_c
        loss = loss + align_w * align_loss
        loss = loss + disentangle_w * disentangle_loss
        loss = loss + center_w * loss_center
        if cls_loss is not None:
            loss = loss + cls_w * cls_loss

        self.log("train/align_weight_eff", align_w, prog_bar=True, sync_dist=True)
        self.log("train/disentangle_weight_eff", disentangle_w, prog_bar=True, sync_dist=True)
        self.log("train/center_weight_eff", center_w, prog_bar=True, sync_dist=True)

        # 5) 总 loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # 6) 各项 raw loss
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/kl_g", kl_g, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/kl_c", kl_c, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/align_loss", align_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/disentangle_loss", disentangle_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_center", loss_center, on_step=False, on_epoch=True, sync_dist=True)

        if cls_loss is not None:
            self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
            if (
                    self.train_f1 is not None
                    and out.get("cls_logits") is not None
                    and y is not None
            ):
                f1_val = self.train_f1(out["cls_logits"], y)
                self.log("train/F1_step", f1_val, prog_bar=False, sync_dist=True)

        return loss

    def on_train_epoch_end(self,):
        if self.train_f1:
            self.train_f1.reset()


    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch, None

        if self.train_data_sample:
            is_min = (y == self.minority_class_id)
            is_maj = ~is_min
            # 必须使用 .detach() 避免计算图积压
            self.train_data.append({
                "x_majority": x[is_maj].detach(),
                "x_minority": x[is_min].detach()
            })


    def on_validation_epoch_end(self):
        # 1. 检查是否有数据
        if not self.train_data:
            return

        X_majority = torch.cat([d["x_majority"] for d in self.train_data], dim=0)
        X_minority = torch.cat([d["x_minority"] for d in self.train_data], dim=0)
        # 如果你只想采样一次训练集作为 benchmark，保留此 flag
        # 如果希望每轮都刷新，就在末尾把此 flag 设为 True
        # self.train_data_sample = False

        device = X_majority.device

        # 2. 划分训练/测试 (如果没有外部验证集)
        if self.val_data is None:
            num_min = X_minority.size(0)
            num_maj = X_majority.size(0)

            # 计算 70% 的分界点
            idx_min = max(1, int(num_min * 0.7))
            idx_maj = max(1, int(num_maj * 0.7))

            # 生成随机打乱的索引 (注意保持在同一个 device 上，防止显存拷贝报错)
            perm_min = torch.randperm(num_min, device=X_minority.device)
            perm_maj = torch.randperm(num_maj, device=X_majority.device)

            # 根据随机索引切分数据
            minority_train = X_minority[perm_min[:idx_min]]
            minority_test = X_minority[perm_min[idx_min:]]

            majority_train = X_majority[perm_maj[:idx_maj]]
            majority_test = X_majority[perm_maj[idx_maj:]]
        else:
            minority_train = X_minority
            majority_train = X_majority
            # 确保 val_data 也在正确的设备上
            v_x, v_y = self.val_data["x_test"].to(device), self.val_data["y_test"].to(device)
            minority_test = v_x[v_y == self.minority_class_id]
            majority_test = v_x[v_y != self.minority_class_id]

        # 3. 双分类器筛选生成
        target_count = minority_train.size(0) * 9
        # new_minority = self.get_filtered_samples(minority_train, target_count)
        new_minority = self.model.generate_vae_prior(minority_train, num_variations=9, alpha=0.5)
        # 4. 构建评估集
        all_x_train = torch.cat([majority_train, minority_train, new_minority], dim=0)
        all_y_train = torch.cat([
            torch.full((len(majority_train),), 0, device=device),
            torch.full((len(minority_train) + len(new_minority),), 1, device=device)
        ], dim=0)

        all_x_test = torch.cat([majority_test, minority_test], dim=0)
        all_y_test = torch.cat([
            torch.full((len(majority_test),), 0, device=device),
            torch.full((len(minority_test),), 1, device=device)
        ], dim=0)
        res_g, res_f1, res_acc = 0., 0., 0.
        res_gen = res_f1
        from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.inference import Inference
        inference = Inference
        all_x_train = all_x_train
        all_x_test = all_x_test
        # 3. 每隔 N 个 Epoch 执行分类器评估
        if self.current_epoch >= 0:
            metrics = train_and_eval_classifier(
                all_x_train, all_y_train,
                all_x_test, all_y_test,
                input_chans=self.input_channels,
                seq_len=all_x_train.shape[-1],
                device=self.device
            )
            res_g = metrics["val_g_means"]
            res_f1 = metrics["val_f1_macro"]
            res_acc = metrics["val_acc"]
            res_gen = res_f1

        # 4. 统一在分支外 Log，确保参数永远一致
        self.log("eval/gen_g_means", res_g, prog_bar=True)
        self.log("eval/gen_f1_macro", res_f1, prog_bar=True)
        self.log("eval/acc", res_acc, prog_bar=True)
        self.log("eval_gen", res_gen, prog_bar=True)

        # 5. 清空列表
        self.train_data.clear()
        # 如果希望下一轮继续收集，重置 flag (取决于你的逻辑需求)


    def configure_optimizers(self):
        # 统一从 self.cfg 读取（由 train.py 赋值为 DotDict/ dict）
        cfg = getattr(self, "cfg", {}) or {}

        optim_cfg = cfg.get("optim", {}) or {}
        trainer_cfg = cfg.get("trainer", {}) or {}

        lr = float(optim_cfg.get("lr", 1e-3))
        wd = float(optim_cfg.get("weight_decay", 0.0))
        raw_betas = optim_cfg.get("betas", (0.9, 0.999))
        if isinstance(raw_betas, (list, tuple)) and len(raw_betas) == 2:
            betas: tuple[float, float] = (float(raw_betas[0]), float(raw_betas[1]))
        else:
            betas = (0.9, 0.999)

        opt_name = str(optim_cfg.get("optimizer", "adam")).lower()
        if opt_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=lr, betas=betas, weight_decay=wd)

        scheduler_name = str(optim_cfg.get("scheduler", "cosine")).lower()
        if scheduler_name == "cosine":
            cos = optim_cfg.get("cosine", {}) or {}
            T_max = int(trainer_cfg.get("max_epochs", cos.get("T_max", 10)))
            eta_min = float(cos.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
            return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

        return opt

    def _current_loss_weights(self):
        """Compute effective loss weights with epoch-based warmup."""
        epoch = float(self.current_epoch)
        factor = min(1.0, max(0.0, epoch / self.warmup_epochs))
        if epoch <= 5:
            turn_off_dilate = True
        else:
            turn_off_dilate = False
        recon_weight = self.recon_lambda

        # 只有正则化项需要 Warmup
        kl_g_weight = self.kl_g_lambda * factor
        kl_c_weight = self.kl_c_lambda * factor
        align_weight = self.align_lambda * factor
        center_weight = self.center_lambda * factor
        disentangle_weight = self.disentangle_lambda * factor

        # 分类损失始终保持满额，提供强监督信号
        cls_weight = self.cls_lambda

        return recon_weight, kl_g_weight, kl_c_weight, align_weight, disentangle_weight, center_weight, cls_weight, turn_off_dilate

    def get_filtered_samples(self, x_min, target_num, threshold=0.7):
        if self.oracle is not None:
            self.oracle.to(self.device)
            self.oracle.eval()

        num_variations = 9
        if self.oracle is None:
            return self.model.generate_vae_prior(x_min, num_variations=num_variations, alpha=0.5)

        # 1. 扩大生成范围（比如生成 5 倍于目标的样本量）
        candidate_multiplier = 5
        candidates = self.model.generate_vae_prior(
            x_min.repeat(candidate_multiplier, 1, 1),
            num_variations=num_variations,
            alpha=0.5
        )

        # 2. 让 Static Oracle 评价这些样本
        with torch.no_grad():
            logits = self.oracle(candidates)
            probs = torch.softmax(logits, dim=1)

            # 提取属于少数类（ID=1）的概率
            minority_probs = probs[:, self.minority_class_id]
            mask = (minority_probs > threshold) & (minority_probs <= 0.999)
            valid_candidates = candidates[mask]
            if len(valid_candidates) >= target_num:
                print(
                    f"Generated {len(candidates)} candidates, with valid condidates {len(valid_candidates)} based on oracle confidence.")
                valid_candidates = valid_candidates[:target_num]
            elif len(valid_candidates) < target_num and len(valid_candidates) > 0:
                print(
                    f"Warning: Only {len(valid_candidates)} candidates passed the oracle filter between "
                    f"threshold {threshold} and 0.999, which is less than the target {target_num}. "
                    f"Use some higher condidates to add in.")
                n_num_needed = target_num - len(valid_candidates)
                higher_mask = minority_probs > 0.999
                higher_candidates = candidates[higher_mask]
                print("higher candidates num:", len(higher_candidates))
                if len(higher_candidates) >= n_num_needed:
                    valid_candidates = torch.cat([valid_candidates, higher_candidates[:n_num_needed]], dim=0)
                else:
                    valid_candidates = torch.cat([valid_candidates, higher_candidates], dim=0)
            elif len(valid_candidates) == 0:
                print(f"Warning: No candidates passed the oracle filter between threshold {threshold} and 0.999. Returning unfiltered samples.")
                valid_candidates = candidates[:target_num]
        print(f"Generated {len(candidates)} candidates, selected top {len(valid_candidates)} based on oracle confidence.")
        return valid_candidates

def train_and_eval_classifier(train_data, train_labels, test_data, test_labels, input_chans, seq_len, device):
    # 1. 创建一个临时文件夹，专门存放这次评估的 checkpoint
    import tempfile
    from torch.utils.data import DataLoader, TensorDataset
    import os
    import shutil
    temp_dir = tempfile.mkdtemp()

    # 2. 准备数据
    # 注意：这里的 normalisation 是针对评估阶段的分类器而言的，确保它们在同一分布上训练和测试
    # 检查数据是否为原始数据
    # normalisation
    train_data = (train_data - train_data.mean(dim=0)) / (train_data.std(dim=0) + 1e-6)
    test_data = (test_data - test_data.mean(dim=0)) / (test_data.std(dim=0) + 1e-6)
    train_ds = TensorDataset(train_data, train_labels)
    test_ds = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # 3. 初始化分类器 (使用 TimesNetQualityClassifier)
    # 这里的参数根据你的数据情况调整
    # clf = TimesNetQualityClassifier(
    #     input_channels=input_chans,
    #     seq_len=seq_len,
    #     num_classes=2,
    #     d_model=64,
    #     top_k=3,
    #     lr=1e-3
    # )  # 让 Trainer 自己处理设备
    clf = TSQualityClassifier(
        input_channels=input_chans,
        num_classes=2,
        lr=1e-3
    )
    # 让 Trainer 自己处理设备
    # 4. 配置 Checkpoint (保持不变)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=temp_dir,
        filename="best_eval",
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1,
        save_weights_only=True
    )

    # 5. Trainer (保持不变)
    eval_trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False
    )

    try:
        # --- 训练 ---
        eval_trainer.fit(clf, train_loader, test_loader)

        # --- 【关键修正】重新加载最佳模型进行验证 ---
        # 1. 获取最佳模型的路径
        best_model_path = checkpoint_callback.best_model_path

        if best_model_path:
            # 2. 重新加载最佳权重
            best_model = TSQualityClassifier.load_from_checkpoint(
                input_channels=input_chans,
                num_classes=2,
                lr=1e-3
            )

            # 3. 在测试集上运行一次验证，获取该状态下的所有指标
            # validate 返回的是一个列表，里面是每个 dataloader 的结果字典
            val_results = eval_trainer.validate(best_model, test_loader, verbose=False)[0]

            results = {
                "val_f1_macro": float(val_results.get("val_f1_macro", 0.0)),
                "val_acc": float(val_results.get("val_acc", 0.0)),
                "val_g_means": float(val_results.get("val_g_means", 0.0))
            }
        else:
            # 极端情况：如果一次 validation 都没跑完（比如报错了）
            results = {
                "val_f1_macro": 0.0, "val_acc": 0.0, "val_g_means": 0.0
            }

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

            # 2. 【新增】显式清理模型和 Trainer 占用的显存
        del clf
        del eval_trainer
        del best_model
        # 如果有 best_model 也记得 del best_model

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    return results
