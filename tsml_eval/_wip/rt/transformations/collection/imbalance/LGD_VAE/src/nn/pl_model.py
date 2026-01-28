import torch
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.model import LatentGatedDualVAE
import lightning.pytorch as pl
import torchmetrics
from typing import Tuple


# define the LightningModule
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
        lr = 1e-3):

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
        recon_w, kl_g_w, kl_c_w, align_w, disentangle_w, center_w, cls_w = self._current_loss_weights()

        # === 2) 把有效权重传进模型 ===
        out = self.model(batch, y)

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
            "train/loss",
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
        self.log("train/loss_center", loss_center, on_step=False, on_epoch=True, sync_dist=True)
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

        recon_w, kl_g_w, kl_c_w, align_w, disentangle_w, center_w, cls_w = self._current_loss_weights()
        out = self.model(x, y)

        recon_loss = out.get("recon_loss", 0.0)
        kl_g = out.get("kl_g", 0.0)
        kl_c = out.get("kl_c", 0.0)
        align_loss = out.get("align_loss", 0.0)
        disentangle_loss = out.get("disentangle_loss", 0.0)
        loss_center = out.get("loss_center", 0.0)
        cls_loss = out.get("cls_loss", None)

        loss = recon_w * recon_loss + kl_g_w * kl_g + kl_c_w * kl_c
        loss = loss + align_w * align_loss
        loss = loss + disentangle_w * disentangle_loss
        loss = loss + center_w * loss_center
        if cls_loss is not None:
            loss = loss + cls_w * cls_loss

        # log 各项
        self.log("eval_recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/kl_g", kl_g, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/kl_c", kl_c, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/align_loss", align_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/disentangle_loss", disentangle_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/loss_center", loss_center, on_step=False, on_epoch=True, sync_dist=True)
        if cls_loss is not None:
            self.log("eval/cls_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
            if out.get("cls_logits") is not None and y is not None:
                # [修改点 2] 更新 valid_f1
                if self.valid_acc is not None:
                    self.valid_acc.update(out["cls_logits"], y)
                if self.valid_f1 is not None:
                    self.valid_f1.update(out["cls_logits"], y)

        self.log("eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # 1. 计算所有类别的 F1 (返回一个向量，例如 [0.98, 0.45])
        all_f1_scores = self.valid_f1.compute()
        # 2. 提取少数类 F1
        # self.hparams.minority_class_id 是你在 __init__ 里存下来的那个 ID
        min_id = self.hparams.minority_class_id
        # 即使你只有两个类，all_f1_scores[min_id] 也能精准拿到少数类的分数
        f1_min = all_f1_scores[min_id]
        # 3. 记录日志 (这个名字 'eval_f1_min' 要跟你的 Monitor 对应)
        self.log("eval_f1_min", f1_min, prog_bar=True)

        # 4. (可选) 如果你想看多数类 F1，也可以记下来对比
        f1_maj = all_f1_scores[0]
        self.log("eval_f1_maj", f1_maj)
        self.valid_f1.reset()

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

        recon_weight = self.recon_lambda

        # 只有正则化项需要 Warmup
        kl_g_weight = self.kl_g_lambda * factor
        kl_c_weight = self.kl_c_lambda * factor
        align_weight = self.align_lambda * factor
        center_weight = self.center_lambda * factor
        disentangle_weight = self.disentangle_lambda * factor

        # 分类损失始终保持满额，提供强监督信号
        cls_weight = self.cls_lambda

        return recon_weight, kl_g_weight, kl_c_weight, align_weight, disentangle_weight, center_weight, cls_weight
