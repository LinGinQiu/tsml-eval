import torch
import numpy as np
from torch import nn
from typing import List, Tuple


# Basic class of loss
class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super().__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


# =======================================
# MDD calculation and utilities functions
def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b + 1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super().__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def calculate_mdd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    mdd = (HistoLoss(ori_data[:, 1:, :], n_bins=50, name='marginal_distribution')(
        gen_data[:, 1:, :])).detach().cpu().numpy()
    return mdd.item()


# =======================================
# ACF calculation and utilities functions
def acf_torch(x: torch.Tensor, max_lag: int, dim: tuple[int] = (0, 1)) -> torch.Tensor:
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                    torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0))
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super().__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


def calculate_acd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    acf = (ACFLoss(ori_data, name='auto_correlation', stationary=True)(gen_data)).detach().cpu().numpy()
    return acf.item()


# =======================================
# SD calculation and utilities functions
def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super().__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real)

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake)
        return self.norm_foo(skew_fake - self.skew_real)


def calculate_sd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    skewness = SkewnessLoss(x_real=ori_data, name='skew')
    sd = skewness.compute(gen_data).mean()
    sd = float(sd.numpy())
    return sd


# =======================================
# KD calculation and utilities functions
def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super().__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real)

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake)
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)


def calculate_kd(ori_data, gen_data):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    kurtosis = KurtosisLoss(x_real=ori_data, name='kurtosis')
    kd = kurtosis.compute(gen_data).mean()
    kd = float(kd.numpy())
    return kd


def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    if n_series == 1:
        ori_data = np.transpose(ori_data, (0, 2, 1))
        gen_data = np.transpose(gen_data, (0, 2, 1))
    from tsml_eval._wip.rt.distances import pairwise_distance
    distance = pairwise_distance(ori_data, gen_data, method='euclidean')
    distance = distance.mean()
    return distance


def calculate_dtw(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    if n_series == 1:
        ori_data = np.transpose(ori_data, (0, 2, 1))
        gen_data = np.transpose(gen_data, (0, 2, 1))
    from tsml_eval._wip.rt.distances import pairwise_distance
    distance = pairwise_distance(ori_data, gen_data, method='euclidean')
    distance = distance.mean()
    return distance


import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def make_sure_path_exist(path):
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self


    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data


    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


# adapt from https://github.com/jsyoon0823/TimeGAN, https://openreview.net/forum?id=ez6VHWvuXEx

def visualize_tsne(ori_data, gen_data, result_path, save_file_name):
    if ori_data.shape[-1] == 1:
        ori_data = np.transpose(ori_data, (0, 2, 1))
        gen_data = np.transpose(gen_data, (0, 2, 1))

    sample_num = min(ori_data.shape[0], gen_data.shape[0])

    idx = np.random.permutation(len(ori_data))[:sample_num]
    ori_data = ori_data[idx]
    idx = np.random.permutation(len(gen_data))[:sample_num]
    gen_data = gen_data[idx]

    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    colors = ["C0" for i in range(sample_num)] + ["C1" for i in range(sample_num)]

    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000, random_state=42)  # 40, 300
    tsne_results = tsne.fit_transform(prep_data_final)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.scatter(tsne_results[:sample_num, 0], tsne_results[:sample_num, 1],
               c=colors[:sample_num], alpha=0.5, label="Original", s=5)
    ax.scatter(tsne_results[sample_num:, 0], tsne_results[sample_num:, 1],
               c=colors[sample_num:], alpha=0.5, label="Generated", s=5)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)
    save_path = os.path.join(result_path, 'tsne_' + save_file_name + '.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')


def visualize_distribution(ori_data, gen_data, result_path, save_file_name):
    sample_num = min(ori_data.shape[0], gen_data.shape[0])

    idx = np.random.permutation(len(ori_data))[:sample_num]
    ori_data = ori_data[idx]
    idx = np.random.permutation(len(gen_data))[:sample_num]
    gen_data = gen_data[idx]


    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sns.kdeplot(prep_data.flatten(), color='C0', linewidth=2, label='Original', ax=ax)

    # Plotting KDE for generated data on the same axes
    sns.kdeplot(prep_data_hat.flatten(), color='C1', linewidth=2, linestyle='--', label='Generated', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(0, 1)
    for pos in ['top', 'right']:
        ax.spines[pos].set_visible(False)
    save_path = os.path.join(result_path, 'distribution_' + save_file_name + '.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
