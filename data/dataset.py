"""custom criket score dataset"""
import logging
import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader, Dataset
from models.utils import LabelEncoderExt, Standardizer
from sklearn.model_selection import train_test_split

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


def scale_data(cfg, train_df, test_df):
    # Data scaling
    train_df_original, test_df_original = train_df.copy(), test_df.copy()
    for col in cfg.categorical_features:
        # data[col] = data[col].astype('str')
        le = LabelEncoderExt()
        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        logging.info(
            f"Converting feature {col} into {len(le.classes_)} levels, with {train_df[col].unique().shape[0]} classes.")

        try:
            test_df[col] = le.transform(test_df[col])
        except Exception:
            logging.warning(f"Unseen categorical variable level {col}:, \n"
                            f"{set(test_df[col].unique()) - set(train_df[col].unique())}")

    # le = LabelEncoder()
    # data[cfg.response_binary] = le.fit_transform(data[cfg.response_binary])
    # logging.info(f"Classes of response: {le.classes_}")
    standardizer = Standardizer()
    for col in cfg.numerical_features + [cfg.response]:
        standardizer = Standardizer()
        response_flag = True if col == cfg.response else False
        standardizer.fit(train_df[col])
        train_df[col] = standardizer.transform(train_df[col], response_flag)
        test_df[col] = standardizer.transform(test_df[col], response_flag)
    response_standardizer = standardizer
    return train_df, test_df, train_df_original, test_df_original, response_standardizer


def forming_train_and_test_data(batch_size, cfg, data):
    data = data.copy()
    response = cfg.response if cfg.model_stage == 'regression' else cfg.response_binary
    data['stratify_col'] = data[list(cfg.stratefied_sampling_categories)].apply(lambda x: '_'.join(x.map(str)), axis=1)
    counts = data['stratify_col'].value_counts()
    to_remove = counts[counts < 2].index # remove classes with fewer samples than this
    data = data[~data['stratify_col'].isin(to_remove)]
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['stratify_col'])
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    train_df = balance_dataset(train_df, target_column=response)
    test_df = balance_dataset(test_df, target_column=response)
    train_df, test_df, train_df_original, test_df_original, response_standardizer = scale_data(cfg, train_df, test_df)
    logging.info("Sizes of training and testing datasets")
    stratefied_sampling_categories = list(cfg.stratefied_sampling_categories) + [response]
    logging.info(f"Training: {len(train_df)} \n {train_df.groupby(stratefied_sampling_categories).size()}")
    logging.info(f"Testing: {len(test_df)} \n {test_df.groupby(stratefied_sampling_categories).size()}")
    for stratefied_sampling_category in stratefied_sampling_categories:
        logging.info(f"Training data value counts on {stratefied_sampling_category}, "
                     f"{train_df[stratefied_sampling_category].value_counts()}")
        logging.info(f"Testing data value counts on {stratefied_sampling_category}, "
                     f"{test_df[stratefied_sampling_category].value_counts()}")
    # composing dataset
    train_dataset = CriketScoreDataSetWithCatAndNum(train_df, cfg.categorical_features, cfg.numerical_features, response)
    test_dataset = CriketScoreDataSetWithCatAndNum(test_df, cfg.categorical_features, cfg.numerical_features, response)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")
    return test_loader, train_loader, train_df, test_df, train_df_original, test_df_original, response_standardizer

