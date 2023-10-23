"""custom criket score dataset"""
import logging
import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)


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
        return (torch.tensor(list(map(int, self.dataframe.iloc[idx][self.cat_cols].values)), dtype=torch.long),
                torch.tensor(list(map(float, self.dataframe.iloc[idx][self.num_cols].values)), dtype=torch.float32),
                torch.tensor(self.target[idx], dtype=torch.long))


def balance_dataset(df, target_column):
    """
    Balances the dataset by either upsampling or downsampling based on the target_sample_size.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - target_column (str): The name of the target column that needs to be balanced.
    - target_sample_size (int): The desired number of samples per unique value in the target column.

    Returns:
    - resampled_df (pd.DataFrame): The balanced dataframe.
    """

    # Initialize an empty dataframe to store the resampled data
    resampled_df = pandas.DataFrame()

    # Iterate over each unique value in target_column
    target_sample_size = int(df.groupby(target_column).size().median())
    for value in df[target_column].unique():
        subset = df[df[target_column] == value]

        if len(subset) > target_sample_size:
            # Downsample without replacement
            resampled_data = subset.sample(target_sample_size, replace=False)
        else:
            # Upsample with replacement
            resampled_data = subset.sample(target_sample_size, replace=True)

        resampled_df = pandas.concat([resampled_df, resampled_data], axis=0)

    # Shuffle the data
    resampled_df = resampled_df.sample(frac=1).reset_index(drop=True)

    return resampled_df


def forming_train_and_test_data(batch_size, cfg, data):
    train_df = data.sample(frac=0.8,
                           weights=data.groupby(list(cfg.stratefied_sampling_categories))[cfg.response].transform(
                               'count'))
    test_df = data.drop(train_df.index).copy()
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    train_df = balance_dataset(train_df, target_column=cfg.response)
    test_df = balance_dataset(test_df, target_column=cfg.response)
    logging.info("Sizes of training and testing datasets")
    logging.info(f"Training: {len(train_df)} \n {train_df.groupby(list(cfg.stratefied_sampling_categories)).size()}")
    logging.info(f"Testing: {len(test_df)} \n {test_df.groupby(list(cfg.stratefied_sampling_categories)).size()}")
    for stratefied_sampling_category in cfg.stratefied_sampling_categories:
        logging.info(f"Training data value counts on {stratefied_sampling_category}, "
                     f"{train_df[stratefied_sampling_category].value_counts()}")
        logging.info(f"Testing data value counts on {stratefied_sampling_category}, "
                     f"{test_df[stratefied_sampling_category].value_counts()}")
    # composing dataset
    train_dataset = CriketScoreDataSetWithCatAndNum(train_df, cfg.categorical_features, cfg.numerical_features,
                                                    cfg.response)
    test_dataset = CriketScoreDataSetWithCatAndNum(test_df, cfg.categorical_features, cfg.numerical_features,
                                                   cfg.response)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")
    return test_loader, train_loader

