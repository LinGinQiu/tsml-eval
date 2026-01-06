import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer

# ---------------------------------------------------------
class MGVAE_model(nn.Module):
    """
    标准的 VAE 架构，用于 MGVAE 的骨干网络 [cite: 523]
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 假设数据归一化到 [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------------------------------------------------------
# 损失函数
# ---------------------------------------------------------
def loss_function(recon_x, x, mu, logvar):
    """
    标准 VAE Loss: Reconstruction + KL Divergence
    注意：论文中使用了基于多数类的混合先验，但在快速实现中，
    EWC 和 Pre-train 是防止坍塌的最关键因素 [cite: 304]。
    """
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# ---------------------------------------------------------
# EWC (Elastic Weight Consolidation) 工具类 [cite: 147, 153]
# ---------------------------------------------------------
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean = {}  # 存储旧参数 (Pre-trained weights)
        self.fisher = {}  # 存储 Fisher 信息矩阵 (参数重要性)

        self._compute_fisher()

    def _compute_fisher(self):
        # 初始化 Fisher 矩阵
        for n, p in self.params.items():
            self.mean[n] = p.clone().detach()
            self.fisher[n] = torch.zeros_like(p.data)

        self.model.eval()
        for batch_idx, (data,) in enumerate(self.dataloader):
            data = data.to(self.device)
            self.model.zero_grad()

            # 使用 VAE loss 的梯度近似 Fisher 信息
            recon_batch, mu, logvar = self.model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    # Fisher = Expectation of gradients squared
                    self.fisher[n] += p.grad.data.pow(2) / len(self.dataloader)

        self.model.train()  # 切回训练模式

    def penalty(self, model):
        """计算 EWC 惩罚项 """
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.fisher[n] * (p - self.mean[n]).pow(2)
            loss += _loss.sum()
        return loss


# ---------------------------------------------------------
# 训练与生成流程
# ---------------------------------------------------------
def train_mgvae_pipeline(majority_data, minority_data, input_dim, ewc_lambda=500, epochs_pre=50, epochs_fine=50,device=None):
    device = device

    # 1. 数据准备
    maj_loader = DataLoader(TensorDataset(majority_data), batch_size=100, shuffle=True)
    min_loader = DataLoader(TensorDataset(minority_data), batch_size=10, shuffle=True)  # 少数类 Batch 较小

    model = MGVAE_model(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"--- Stage 1: Pre-training on Majority Data ({epochs_pre} epochs) [cite: 142] ---")
    model.train()
    for epoch in range(epochs_pre):
        total_loss = 0
        for batch_idx, (data,) in enumerate(maj_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Pre-train Epoch {epoch}: Average Loss: {total_loss / len(maj_loader.dataset):.4f}')

    print("--- Calculating Fisher Information for EWC [cite: 152] ---")
    # 在多数类数据上计算参数重要性
    ewc = EWC(model, maj_loader, device)

    print(f"--- Stage 2: Fine-tuning on Minority Data ({epochs_fine} epochs) with EWC [cite: 143, 147] ---")
    # 注意：此时我们是在微调同一个模型，目的是让它适应少数类，但不要忘记多数类的结构
    for epoch in range(epochs_fine):
        total_loss = 0
        for batch_idx, (data,) in enumerate(min_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            # 基础 VAE Loss
            vae_loss = loss_function(recon_batch, data, mu, logvar)

            # EWC 正则项: lambda * sum(F * (theta - theta_old)^2)
            ewc_loss = ewc.penalty(model)

            # 总损失
            loss = vae_loss + ewc_lambda * ewc_loss

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Fine-tune Epoch {epoch}: Average Loss: {total_loss / len(min_loader.dataset):.4f}')

    return model


def generate_samples(model, majority_samples):
    """
    Algorithm 1: Generating Process [cite: 109]
    使用多数类样本作为先验引导生成新的少数类样本
    Mapping: Majority -> z -> New Minority
    """
    model.eval()
    with torch.no_grad():
        # 1. 编码多数类样本得到潜在变量 z [cite: 111]
        mu, logvar = model.encode(majority_samples)
        z = model.reparameterize(mu, logvar)

        # 2. 解码得到新的少数类样本 [cite: 112]
        generated_minority = model.decode(z)

    return generated_minority


class MGVAE(BaseCollectionTransformer):
    """
    MGVAE 转换器封装
    """

    def __init__(self, ewc_lambda=500, epochs_pre=50, epochs_fine=50, n_jobs: int = 1,
            random_state=None,
    ):

        super().__init__()
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ewc_lambda = ewc_lambda
        self.epochs_pre = epochs_pre
        self.epochs_fine = epochs_fine
        self.model = None
        self._device = None
        self.n_generate_samples = 0

    def _fit(self, X, y=None):
        import socket
        hostname = socket.gethostname()
        is_iridis = "iridis" in hostname.lower() or "loginx" in hostname.lower()
        is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名

        if is_iridis:
            print("[ENV] Detected Iridis HPC environment")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif is_mac:
            print("[ENV] Detected local macOS environment")
            self._device = torch.device("cpu")
        else:
            print("[ENV] Unknown environment, fallback to current dir")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device in oversampler is {self._device}")
        X = X.reshape(X.shape[0], -1)  # 展平输入
        self.input_dim = X.shape[-1]
        self._random_state = check_random_state(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        label_majority = classes[np.argmax(counts)]
        label_minority = classes[np.argmin(counts)]
        self._cls_maj = label_majority
        self._cls_min = label_minority
        n_generate_samples = counts[np.argmax(counts)] - counts[np.argmin(counts)]
        self.n_generate_samples = n_generate_samples
        X_majority = torch.tensor(X[y == label_majority], dtype=torch.float32)
        X_minority = torch.tensor(X[y == label_minority], dtype=torch.float32)
        self.model = train_mgvae_pipeline(
            majority_data=X_majority,
            minority_data=X_minority,
            input_dim=self.input_dim,
            ewc_lambda=self.ewc_lambda,
            epochs_pre=self.epochs_pre,
            epochs_fine=self.epochs_fine
        )
        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        C, L = X.shape[1], X.shape[2]
        if self.model is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        X_majority = torch.tensor(X.reshape(X.shape[0], -1)[y == self._cls_maj], dtype=torch.float32).to(self._device)
        X_new = generate_samples(self.model, X_majority).cpu().detach().numpy()
        index = self._random_state.choice(X_new.shape[0], size=self.n_generate_samples, replace=False)
        X_new = X_new[index][:,np.newaxis, :]
        y_new = np.array([self._cls_min] * self.n_generate_samples)
        X_resampled.append(X_new)
        y_resampled.append(y_new)
        X_synthetic = np.vstack(X_resampled)
        y_synthetic = np.hstack(y_resampled)
        return X_synthetic, y_synthetic
# ==========================================
# 示例用法 (使用随机数据模拟)
# ==========================================
if __name__ == "__main__":
    from tsml_eval._wip.rt.transformations.collection.imbalance._utils import _plot_series_list

    dataset_name = 'AllGestureWiimoteX_eq'
    smote = MGVAE(random_state=0)
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
