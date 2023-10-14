"""custom criket score dataset"""
import numpy as np
import pandas
import torch.utils.data


class CriketScoreDataSet(torch.utils.data.Dataset):
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
