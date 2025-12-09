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
        latent_dim_class=16,
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

        if cls_embed and weights is not None:
            self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=len(weights), average="macro")
            self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(weights))
        else:
            self.train_f1 = None
            self.valid_acc = None

        self.lr = lr
        self.fig = None
        self.save_hyperparameters()

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
        self.log("train/recon_loss", recon_loss, prog_bar=False, sync_dist=True)
        self.log("train/kl_g", kl_g, prog_bar=False, sync_dist=True)
        self.log("train/kl_c", kl_c, prog_bar=False, sync_dist=True)
        self.log("train/align_loss", align_loss, prog_bar=False, sync_dist=True)
        self.log("train/disentangle_loss", disentangle_loss, prog_bar=False, sync_dist=True)
        self.log("train/loss_center", loss_center, prog_bar=False, sync_dist=True)

        if cls_loss is not None:
            self.log("train/cls_loss", cls_loss, prog_bar=False, sync_dist=True)
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

        # use fix weights for validation
        recon_w = self.recon_lambda
        kl_g_w = self.kl_g_lambda
        kl_c_w = self.kl_c_lambda
        align_w = self.align_lambda
        center_w = self.center_lambda
        cls_w = self.cls_lambda  # 现在不 warmup 分类
        disentangle_w = self.disentangle_lambda
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
        self.log("eval/recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/kl_g", kl_g, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/kl_c", kl_c, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/align_loss", align_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/disentangle_loss", disentangle_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("eval/loss_center", loss_center, on_step=False, on_epoch=True, sync_dist=True)
        if cls_loss is not None:
            self.log("eval/cls_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
            if (
                    self.valid_acc is not None
                    and out.get("cls_logits") is not None
                    and y is not None
            ):
                self.valid_acc.update(out["cls_logits"], y)

        self.log(
            "eval/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_end(self,):
        if self.valid_acc:
            self.log("eval/acc", self.valid_acc.compute())
            self.valid_acc.reset()

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

        # 可以以后丢到 config 里
        align_warmup_epochs = 10.0
        kl_warmup_epochs = 10.0
        center_warmup_epochs = 10.0

        # 线性 0 → 1
        align_factor = min(1.0, max(0.0, epoch / align_warmup_epochs))
        kl_factor = min(1.0, max(0.0, epoch / kl_warmup_epochs))
        center_factor = min(1.0, max(0.0, epoch / center_warmup_epochs))

        recon_weight = self.recon_lambda
        kl_g_weight = self.kl_g_lambda * kl_factor
        kl_c_weight = self.kl_c_lambda * kl_factor
        align_weight = self.align_lambda * align_factor
        center_weight = self.center_lambda * center_factor
        cls_weight = self.cls_lambda  # 现在不 warmup 分类
        disentangle_weight = self.disentangle_lambda * align_factor



        return recon_weight, kl_g_weight, kl_c_weight, align_weight, disentangle_weight, center_weight, cls_weight
