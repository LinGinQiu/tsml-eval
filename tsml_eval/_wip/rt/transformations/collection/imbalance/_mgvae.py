import numpy as np
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
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
# 1. 辅助数学函数 (实现 Log-Sum-Exp Trick 防止数值溢出)
# ---------------------------------------------------------
def log_normal_diag(x, mu, logvar):
    """
    计算 x 在高斯分布 N(mu, exp(logvar)) 下的对数概率
    """
    # const = -0.5 * math.log(2 * math.pi)
    # 忽略常数项通常不影响优化，但为了精确复现最好加上
    return -0.5 * (logvar + (x - mu).pow(2) / logvar.exp())


def log_mean_exp(value, dim=0):
    """
    计算 log(mean(exp(value)))，使用 log-sum-exp 技巧
    log(1/N * sum(exp(x))) = log(sum(exp(x))) - log(N)
    """
    return torch.logsumexp(value, dim=dim) - math.log(value.size(dim))


# ---------------------------------------------------------
# [cite_start]2. 核心：Majority-Guided Loss Function [cite: 100, 446]
# ---------------------------------------------------------
def loss_function_mgvae(model, x, x_majority_sample, mu, logvar, z, recon_x):
    """
    实现论文 Eq. (4) 和 Algorithm 2
    x: 当前训练的 batch (可能是少数类，也可能是多数类)
    x_majority_sample: 用于构建先验的多数类样本 (Batch size S)
    """
    batch_size = x.size(0)

    # 1. Reconstruction Loss (重构误差)
    # 论文中通常使用 SSE (Sum Squared Error) 或 BCE，这里保持与原代码一致
    MSE = F.mse_loss(recon_x, x, reduction='sum') / batch_size

    # 2. 计算 log q(z|x) (当前样本的后验概率)
    # z: [batch_size, latent_dim]
    # mu, logvar: [batch_size, latent_dim]
    log_q_z = log_normal_diag(z, mu, logvar)
    log_q_z = torch.sum(log_q_z, dim=1)  # [batch_size]

    # 3. 计算 log p(z) (基于多数类的混合先验概率)
    # 我们需要计算 z 在 "S个多数类样本形成的高斯混合分布" 下的概率
    # Prior p(z) = 1/S * sum_k N(z | mu_k, var_k)

    with torch.no_grad():
        # 获取多数类样本的分布参数，作为先验的组件
        # 这些参数在计算 Loss 时被视为常数 (detach)
        mu_maj, logvar_maj = model.encode(x_majority_sample)

    # 扩展维度以便进行广播计算 (Pairwise distance)
    # z_expand:      [batch_size, 1, latent_dim]
    # mu_maj_expand: [1, S, latent_dim]
    z_expand = z.unsqueeze(1)
    mu_maj_expand = mu_maj.unsqueeze(0)
    logvar_maj_expand = logvar_maj.unsqueeze(0)

    # 计算 log N(z_i | mu_k, var_k)
    # 结果 shape: [batch_size, S, latent_dim]
    log_p_z_components = log_normal_diag(z_expand, mu_maj_expand, logvar_maj_expand)

    # 在 latent_dim 维度求和 -> log probability per component
    # shape: [batch_size, S]
    log_p_z_components = torch.sum(log_p_z_components, dim=2)

    # 计算混合概率 log(1/S * sum(exp(...)))
    # shape: [batch_size]
    log_p_z = log_mean_exp(log_p_z_components, dim=1)

    # 4. KL Divergence 近似: E[log q(z|x) - log p(z)]
    # 注意：这里我们计算的是 batch 上的均值
    KL = (log_q_z - log_p_z).mean()

    # Total Loss
    # 论文通常是 sum over batch，还是 mean?
    # Algorithm 2 显示是 1/B * sum(...)，即 mean。
    return MSE + KL


# ---------------------------------------------------------
# [cite_start]3. 修改后的训练循环 (Train Pipeline) [cite: 469]
# ---------------------------------------------------------
def train_mgvae_pipeline(majority_data, minority_data, input_dim, ewc_lambda=500, epochs_pre=50, epochs_fine=50,
                         device=None, S=100):
    """
    S: 每次计算先验时采样的多数类样本数量 (Algorithm 2 中的 S)
    """
    device = device if device else torch.device("cpu")

    # 转换为 Dataset
    maj_dataset = TensorDataset(majority_data)
    min_dataset = TensorDataset(minority_data)

    maj_loader = DataLoader(maj_dataset, batch_size=100, shuffle=True)
    min_loader = DataLoader(min_dataset, batch_size=10, shuffle=True)

    # 用于随机采样的完整多数类数据 (放在 GPU 上以便快速索引，如果显存够的话)
    majority_data_full = majority_data.to(device)
    num_maj = majority_data_full.size(0)

    model = MGVAE_model(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Helper: 随机采样 S 个多数类样本作为先验 ---
    def get_majority_prior_samples(n_samples):
        idx = torch.randperm(num_maj)[:n_samples]
        return majority_data_full[idx]

    # ==========================
    # Stage 1: Pre-training
    # ==========================
    print(f"--- Stage 1: Pre-training on Majority Data ---")
    model.train()
    for epoch in range(epochs_pre):
        total_loss = 0
        for batch_idx, (data,) in enumerate(maj_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # 1. Forward
            recon_batch, mu, logvar = model(data)
            # 2. Reparameterize (为了计算 Loss 中的 z)
            z = model.reparameterize(mu, logvar)

            # 3. 采样 Prior 参考样本 (来自多数类)
            # Algorithm 2 Line 3: Randomly down-sample S majority samples
            x_maj_prior = get_majority_prior_samples(S)

            # 4. 计算 MGVAE Loss
            loss = loss_function_mgvae(model, data, x_maj_prior, mu, logvar, z, recon_batch)

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Pre-train Epoch {epoch}: Loss: {total_loss / len(maj_loader):.4f}')

    # ==========================
    # EWC Calculation
    # ==========================
    print("--- Calculating Fisher Information for EWC ---")
    ewc = EWC(
        model=model,
        dataloader=maj_loader,
        device=device,
        loss_fn_ref=loss_function_mgvae,  # 传入刚才定义的函数
        majority_data_tensor=majority_data_full,  # 传入完整 tensor
        S=S
    )

    # ==========================
    # Stage 2: Fine-tuning
    # ==========================
    print(f"--- Stage 2: Fine-tuning on Minority Data ---")
    for epoch in range(epochs_fine):
        total_loss = 0
        for batch_idx, (data,) in enumerate(min_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            z = model.reparameterize(mu, logvar)

            # 同样需要采样多数类作为先验指导！
            # 这是论文核心：即使在微调少数类时，Latent Space 也是由多数类定义的
            x_maj_prior = get_majority_prior_samples(S)

            # MGVAE Loss
            vae_loss = loss_function_mgvae(model, data, x_maj_prior, mu, logvar, z, recon_batch)

            # EWC Regularization
            ewc_loss = ewc.penalty(model)

            loss = vae_loss + ewc_lambda * ewc_loss

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Fine-tune Epoch {epoch}: Loss: {total_loss / len(min_loader):.4f}')

    return model


# ---------------------------------------------------------
# EWC (Elastic Weight Consolidation) 工具类 [cite: 147, 153]
# ---------------------------------------------------------
import torch


class EWC:
    def __init__(self, model, dataloader, device, loss_fn_ref, majority_data_tensor, S=100):
        """
        Args:
            model: 预训练好的模型
            dataloader: 多数类数据的 DataLoader (用于计算 Fisher)
            device: 计算设备
            loss_fn_ref: 引用外部定义的 loss_function_mgvae 函数
            majority_data_tensor: 完整的多数类数据 Tensor (用于随机采样 Prior)
            S: 每次计算 Loss 时采样的 Prior 数量
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.loss_fn = loss_fn_ref
        self.majority_data_tensor = majority_data_tensor
        self.S = S

        # 获取需要计算重要性的参数
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.mean = {}  # 旧参数 θ*
        self.fisher = {}  # Fisher 信息矩阵 F

        self._compute_fisher()

    def _get_prior_samples(self):
        """从多数类数据中随机采样 S 个样本用于构建先验"""
        idx = torch.randperm(self.majority_data_tensor.size(0))[:self.S]
        return self.majority_data_tensor[idx].to(self.device)

    def _compute_fisher(self):
        # 1. 保存旧参数 (Pre-trained weights)
        for n, p in self.params.items():
            self.mean[n] = p.clone().detach()
            self.fisher[n] = torch.zeros_like(p.data)

        self.model.eval()  # 评估模式，但这不影响梯度回传计算 Fisher

        # 2. 遍历多数类数据计算梯度平方期望
        # Fisher F = E[ (grad log p(x))^2 ]
        for batch_idx, (data,) in enumerate(self.dataloader):
            data = data.to(self.device)
            self.model.zero_grad()

            # Forward
            recon_batch, mu, logvar = self.model(data)
            z = self.model.reparameterize(mu, logvar)

            # 关键修改：获取先验样本，并使用 MGVAE 的 Loss
            x_maj_prior = self._get_prior_samples()

            # 使用与训练时相同的 Loss 函数计算曲率 [cite: 153]
            loss = self.loss_fn(self.model, data, x_maj_prior, mu, logvar, z, recon_batch)

            loss.backward()

            # 累积梯度的平方
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    # 将梯度平方累加，最后取平均
                    self.fisher[n] += p.grad.data.pow(2) / len(self.dataloader)

        self.model.train()  # 切回训练模式

    def penalty(self, model):
        """计算 EWC 惩罚项: lambda * sum( F * (theta - theta_old)^2 ) [cite: 157]"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.mean[n]).pow(2)
                loss += _loss.sum()
        return loss


# ---------------------------------------------------------
# 训练与生成流程
# ---------------------------------------------------------

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
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
        X = (X - mean) / std  # 简单标准化
        self.mean = mean
        self.std = std
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
            epochs_fine=self.epochs_fine,
            device=self._device,
        )
        return self

    def _transform(self, X, y=None):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        C, L = X.shape[1], X.shape[2]
        if self.model is None:
            raise RuntimeError("The model has not been trained yet. Call 'fit' first.")
        X_normized = (X.reshape(X.shape[0], -1) - self.mean) / self.std
        X_majority = torch.tensor(X_normized[y == self._cls_maj], dtype=torch.float32).to(self._device)
        X_new = generate_samples(self.model, X_majority).cpu().detach().numpy()
        # 反标准化
        X_new = X_new * self.std + self.mean
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

    dataset_name = 'AconityMINIPrinterLarge_eq'
    smote = MGVAE(random_state=0)
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
