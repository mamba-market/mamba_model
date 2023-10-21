"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import FactorizationMachine, EarlyStopping
from models.utils import train, evaluate, plot_losses, LabelEncoderExt, Standardizer

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    epochs = cfg.epoch
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    weight_decay = cfg.weight_decay
    input_fp = cfg.training_input_fp

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU due to GPU unavailable.")
    # grabbing raw data
    if input_fp.endswith('.csv'):
        data = pandas.read_csv(input_fp)
    else:
        data = pandas.read_pickle(input_fp)
    # eliminating any NAs rows
    logging.info(f"Eliminating {data.isna().any(axis=1).sum()} NA rows...")
    data = data.loc[~data.isna().any(axis=1), :].copy()
    # stratified sampling weights
    logging.info(f"Data size before filtering {len(data)}")
    data = data[data[cfg.response] >= cfg.target_lower_limit].copy()
    data = data[data[cfg.response] <= cfg.target_upper_limit].copy()
    logging.info(f"Data size after settting [{cfg.target_lower_limit}, {cfg.target_upper_limit}] "
                 f"as upper and lower limits: {len(data)}")
    for minority_sampling_category in cfg.minority_sampling_categories:
        data['freq'] = data.groupby(minority_sampling_category)[cfg.response].transform('count')
        # eliminating extremely under represented data
        data = data[data['freq'] >= cfg.minority_sample_limit].copy()
    logging.info(f"Data size after removing under represented rows {len(data)}.")
    if cfg.training_sample_size is not None:
        data = data.sample(n=cfg.training_sample_size, replace=False,
                           weights=data.groupby(list(cfg.stratefied_sampling_categories))[cfg.response].transform('count'))
    logging.info(f"Data size after stratified sampling {len(data)}")
    logging.info(f"{data.describe()}")
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)
    data.to_csv(cfg.sampled_training_data, index=False)

    # Data scaling
    for col in cfg.categorical_features:
        # data[col] = data[col].astype('str')
        le = LabelEncoderExt()
        le.fit(data[col])
        data[col] = le.transform(data[col])
        logging.info(f"Converting feature {col} into {len(le.classes_)} levels, with {data[col].unique().shape[0]} classes.")
    # for col in cfg.numerical_features:
    #     standardizer = Standardizer()
    #     standardizer.fit(data[col])
    #     data[col] = standardizer.transform(data[col])

    dims_categorical_vars = [data[col].max() + 1 for col in cfg.categorical_features]
    dim_numerical_vars = len(cfg.numerical_features)
    logging.info(f"Total dims of categorical vars: \n{dims_categorical_vars}")
    logging.info(f"Dim of numerical vars: \n{dim_numerical_vars}")
    logging.info(f"Range of target value: {data[cfg.response].min(), data[cfg.response].max()}")

    # model initialization
    model = FactorizationMachine(cat_dims=dims_categorical_vars, num_dim=dim_numerical_vars,
                                 k=cfg.embedding_dim, attention_dim=cfg.attention_dim).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=3, checkpoint_path=cfg.model_fp,
                                  checkpoint_filename=f"{cfg.best_model_name}_target_val_"
                                                      f"{cfg.target_lower_limit}_{cfg.target_upper_limit}.pth")
    logging.info("Model successfully initialized.")

    train_losses, test_losses = [], []
    # splitting data, still following the stratified rule.
    loss_discrepency = float('inf')
    test_loader, train_loader = forming_train_and_test_data(batch_size, cfg, data)
    while abs(loss_discrepency) > cfg.allowed_initial_loss_diff:
        logging.info(f"Regenerating test and train data to offset initial loss diff {loss_discrepency}...")
        test_loader, train_loader = forming_train_and_test_data(batch_size, cfg, data)
        train_loss, test_loss = evaluate(model, train_loader, criterion, device), \
                                evaluate(model, test_loader, criterion, device)
        loss_discrepency = train_loss - test_loss
        logging.info(f"current loss difference is {loss_discrepency}, train_loss is {train_loss}, test_loss is {test_loss}")
        if abs(loss_discrepency) <= cfg.allowed_initial_loss_diff:
            train_losses.append(train_loss)
            test_losses.append(test_loss)
    # logging.info(f"Pretraining losses for train and test dataset are {train_loss}, {test_loss}, respectively.")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        early_stopper(test_loss, model)
        if early_stopper.early_stop:
            logging.warning("Early stopping")
            break
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip("data/")
    plot_losses(train_losses, test_losses, data_type=data_type, save_path=
    f"{cfg.loss_plot_fp}_target_val_{cfg.target_lower_limit}_{cfg.target_upper_limit}")


def forming_train_and_test_data(batch_size, cfg, data):
    train_df = data.sample(frac=0.8,
                           weights=data.groupby(list(cfg.stratefied_sampling_categories))[cfg.response].transform(
                               'count'))
    test_df = data.drop(train_df.index).copy()
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
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


if __name__ == '__main__':
    main()

