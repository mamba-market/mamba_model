"""custom criket score dataset"""
import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader, Dataset


class CriketScoreDataSet(Dataset):
    """
    Criket Dataset

    Data preparation
        a separate pkl format data to be loaded with pandas

    :param dataset_path: CriketScore dataset path

    """
    def __init__(self, dataset_path):
        data = pandas.read_pickle(dataset_path).to_numpy()
        self.items = data[:, :5].astype(int)
        self.targets = data[:, -1]
        self.field_dims =  np.max(self.items, axis=0) + 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


class CriketScoreDataSetWithCatAndNum(Dataset):
    def __init__(self, dataframe, cat_cols, num_cols, target_col):
        self.dataframe = dataframe
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target = dataframe[target_col].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return (torch.tensor(list(map(float, self.dataframe.iloc[idx][self.cat_cols].values)), dtype=torch.long),
                torch.tensor(list(map(float, self.dataframe.iloc[idx][self.num_cols].values)), dtype=torch.float32),
                torch.tensor(self.target[idx], dtype=torch.float32))