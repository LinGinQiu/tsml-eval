import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import DataLoader, Dataset

from TimesNet.data_provider.uea import Normalizer, interpolate_missing, subsample
from TimesNet.utils.augmentation import run_augmentation_single

warnings.filterwarnings("ignore")


class UEAloader(Dataset):
    def __init__(self, args, data_zip, flag=None):
        self.args = args
        self.flag = flag
        self.data, self.labels = data_zip[flag.lower()]  # data: (n_samples, 1, seq_len)
        self.max_seq_len = self.data.shape[-1]
        normalizer = Normalizer()
        self.data = normalizer.normalize(self.data)
        self.data = np.transpose(
            self.data, (0, 2, 1)
        )  # final (n_samples, seq_len, channels)

    def instance_norm(self, case):
        mean = case.mean(dim=0, keepdim=True)
        std = case.std(dim=0, keepdim=True) + 1e-5
        return (case - mean) / std

    def __getitem__(self, ind):
        batch_x = self.data[ind]  # shape (seq_len, channels)
        label = self.labels[ind]

        if self.flag.lower() == "train" and self.args.augmentation_ratio > 0:
            batch_x_np = batch_x[np.newaxis, :, :]  # (1, seq_len, channels)
            batch_x_np, label, _ = run_augmentation_single(
                batch_x_np, np.array([label]), self.args
            )
            batch_x = batch_x_np[0]

        return (
            self.instance_norm(torch.from_numpy(batch_x).float()),
            torch.tensor(label).long(),
        )

    def __len__(self):
        return len(self.data)
