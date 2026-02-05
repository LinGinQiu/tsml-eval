import torch
import torch.nn as nn
import torch.nn.functional as F
from tsml_eval._wip.rt.transformations.collection.imbalance.LGD_VAE.src.nn.model import LatentGatedDualVAE
import lightning.pytorch as pl
import torchmetrics
from typing import Tuple
from torchmetrics import Accuracy, F1Score, Recall

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
        self.classifier = nn.Linear(128, num_classes)

        # 初始化评估指标
        # num_classes=2 时，task 可以设为 'multiclass' 也可以设为 'binary'
        # 这里用 multiclass 确保通用性
        task = "multiclass"
        self.acc_metric = Accuracy(task=task, num_classes=num_classes)
        self.f1_metric = Accuracy(task=task, num_classes=num_classes, average='macro')  # 也就是 Macro-F1 的逻辑
        self.f1_macro = F1Score(task=task, num_classes=num_classes, average='macro')

        # G-Means 是每个类别召回率（Recall）的几何平均值
        # 我们先记录每个类的 Recall
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

        # 计算指标
        acc = self.acc_metric(preds, y)
        f1 = self.f1_macro(preds, y)

        # 计算 G-Means: 先拿所有类的 Recall，然后求乘积再开方
        recalls = self.recall_per_class(preds, y)
        g_means = torch.prod(recalls).pow(1 / len(recalls))

        # 日志记录，这样你在训练生成模型时可以拿到这些结果
        metrics = {
            "val_loss": loss,
            "val_acc": acc,
            "val_f1_macro": f1,
            "val_g_means": g_means
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
        self.minority_class_id = minority_class_id
        self.input_channels = in_chans
        self.validation_step_outputs = []

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
        recom_loss_mse = out.get("recon_loss_mse", None)

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
        self.log("train/recon_loss_mse", recom_loss_mse, on_step=False, on_epoch=True, sync_dist=True)
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

        is_min = (y == self.minority_class_id)  # [B]
        is_maj = ~is_min
        X_majortiy = x[is_maj]
        X_minority = x[is_min]
        num_minority = X_minority.size(0)
        num_majority = X_majortiy.size(0)
        minority_test =X_minority[:num_minority//2]
        majority_test = X_majortiy[:num_majority//2]
        majority_train = X_majortiy[num_majority//2:]
        minority_train = X_minority[num_minority//2:]
        n_generation_per_minority = 9
        new_minority = self.model.generate_vae_prior(x_min=minority_train, num_variations=9)

        # 构建本次 Batch 的评估数据
        X_eval_train = torch.cat([majority_train, minority_train, new_minority], dim=0)
        y_eval_train = torch.cat([
            torch.full((len(majority_train),), 0, device=x.device),
            torch.full((len(minority_train) + len(new_minority),), 1, device=x.device)
        ], dim=0)

        X_eval_test = torch.cat([majority_test, minority_test], dim=0)
        y_eval_test = torch.cat([
            torch.full((len(majority_test),), 0, device=x.device),
            torch.full((len(minority_test),), 1, device=x.device)
        ], dim=0)

        # 关键：将这些数据暂时存入 outputs，在 epoch 结束时统一处理
        eval_dict= {
            "x_train": X_eval_train.detach(),
            "y_train": y_eval_train.detach(),
            "x_test": X_eval_test.detach(),
            "y_test": y_eval_test.detach()
        }
        self.validation_step_outputs.append(eval_dict)

    def on_validation_epoch_end(self):
        # 1. 检查是否有数据
        if not self.validation_step_outputs:
            self.log("eval/gen_g_means", 0., prog_bar=False)
            self.log("eval/gen_f1_macro", 0., prog_bar=False)
            self.log("eval/acc", 0., prog_bar=False)
            return

        # 2. 汇总所有 batch 的数据
        all_x_train = torch.cat([o["x_train"] for o in self.validation_step_outputs], dim=0)
        all_y_train = torch.cat([o["y_train"] for o in self.validation_step_outputs], dim=0)
        all_x_test = torch.cat([o["x_test"] for o in self.validation_step_outputs], dim=0)
        all_y_test = torch.cat([o["y_test"] for o in self.validation_step_outputs], dim=0)

        # 3. 每隔 N 个 Epoch 执行分类器评估
        if self.current_epoch >20: #(self.current_epoch + 1) % 1 == 0 and
            # 调用之前写的 train_and_eval_classifier 函数
            metrics = train_and_eval_classifier(
                all_x_train, all_y_train,
                all_x_test, all_y_test,
                input_chans=self.input_channels,
                num_classes=2,
                device=self.device
            )

            # 日志记录
            self.log("eval/gen_g_means", metrics["val_g_means"], prog_bar=True)
            self.log("eval/gen_f1_macro", metrics["val_f1_macro"], prog_bar=True)
            self.log("eval/acc", metrics["val_acc"], prog_bar=True)
        else:
            self.log("eval/gen_g_means", 0., prog_bar=False)
            self.log("eval/gen_f1_macro", 0., prog_bar=False)
            self.log("eval/acc", 0., prog_bar=False)

        # 4. 关键：手动清空列表，防止显存爆炸
        self.validation_step_outputs.clear()

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


def train_and_eval_classifier(train_data, train_labels, test_data, test_labels, input_chans, num_classes, device):
    # 1. 组建 DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(train_data, train_labels)
    test_ds = TensorDataset(test_data, test_labels)

    # 判别器训练不需要太多 Epoch，3-5 次足以看出生成质量
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    clf = TSQualityClassifier(input_channels=input_chans, num_classes=num_classes).to(device)

    # 3. 使用轻量级 Trainer 训练
    # logger=False 和 enable_checkpointing=False 极其重要，防止产生垃圾文件
    eval_trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False  # 关闭进度条，避免洗屏
    )

    eval_trainer.fit(clf, train_loader, test_loader)

    # 4. 获取结果
    return eval_trainer.callback_metrics
