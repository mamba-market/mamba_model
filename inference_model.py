"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import FactorizationMachine
from models.utils import inference

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    batch_size = cfg.batch_size
    training_input_fp = cfg.training_input_fp
    inference_input_fp = cfg.inference_input_fp

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # grabbing raw data
    if training_input_fp.endswith('.csv'):
        training_data = pandas.read_csv(training_input_fp)
    else:
        training_data = pandas.read_pickle(training_input_fp)

    if inference_input_fp.endswith('.csv'):
        data = pandas.read_csv(inference_input_fp)
    else:
        data = pandas.read_pickle(inference_input_fp)

    for col in cfg.categorical_features:
        le = LabelEncoder()
        training_data[col] = le.fit_transform(training_data[col])
        try:
            data[col] = le.transform(data[col])
        except Exception:
            logging.warning(f"Unseen categorical variable level {col}:, \n"
                            f"{set(data[col].unique()) - set(training_data[col].unique())}")
    dims_categorical_vars = [data[col].max() + 1 for col in cfg.categorical_features]
    dim_numerical_vars = len(cfg.numerical_features)

    # splitting data
    data = shuffle(data)
    train_df = data.sample(frac=0.8, random_state=42)
    test_df = data.drop(train_df.index)
    train_df.reset_index(inplace=True, drop=True)
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

