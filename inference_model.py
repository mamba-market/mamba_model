"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import FactorizationMachine
from models.utils import inference

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    batch_size = cfg.batch_size
    input_fp = cfg.input_fp

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # grabbing raw data
    data = pandas.read_pickle(input_fp)
    for col in cfg.categorical_features:
        data[col] = data[col].astype('category').cat.codes
    dims_categorical_vars = [len(data[col].unique()) for col in cfg.categorical_features]
    dim_numerical_vars = len(cfg.numerical_features)

    # splitting data
    train_df = data.sample(frac=0.8, random_state=42)
    train_df.reset_index(inplace=True, drop=True)
    test_df = data.drop(train_df.index)
    test_df.reset_index(inplace=True, drop=True)
    logging.info("Sizes of training and testing datasets")
    logging.info(f"Training: {len(train_df)}")
    logging.info(f"Testing: {len(test_df)}")
    # composing dataset

    test_dataset = CriketScoreDataSetWithCatAndNum(test_df, cfg.categorical_features, cfg.numerical_features,
                                                   cfg.response)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")

    model = FactorizationMachine(dims_categorical_vars, dim_numerical_vars, k=10, attention_dim=50)
    model.load_state_dict(torch.load(os.path.join(cfg.model_fp, cfg.best_model_name)))
    model.eval()

    with torch.no_grad():  # Disable gradient computation
        predictions = inference(model, test_loader, device)

    logging.info(f"Below are the inference results given your input data, shape {len(predictions)}:")
    logging.info(f"Top 100, remaining truncated, \n{predictions[:100]}")

if __name__ == '__main__':
    main()

