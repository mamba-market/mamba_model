"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import FactorizationMachine
from models.utils import inference, LabelEncoderExt, Standardizer

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    batch_size = cfg.batch_size
    training_input_fp = cfg.sampled_training_data
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
    # transform categorical data and numerical data with label encoders and standardizers
    for col in cfg.categorical_features:
        # training_data[col] = training_data[col].astype('str')
        le = LabelEncoderExt()
        le.fit(training_data[col])
        training_data[col] = le.transform(training_data[col])
        try:
            data[col] = le.transform(data[col])
        except Exception:
            logging.warning(f"Unseen categorical variable level {col}:, \n"
                            f"{set(data[col].unique()) - set(training_data[col].unique())}")
    # for col in cfg.numerical_features:
    #     standardizer = Standardizer()
    #     standardizer.fit(training_data[col])
    #     data[col] = standardizer.transform(data[col])

    dims_categorical_vars = list(map(int, [training_data[col].max() + 1 for col in cfg.categorical_features]))
    dim_numerical_vars = len(cfg.numerical_features)

    # splitting data
    data = shuffle(data)
    logging.info(f"Inference: {len(data)}")
    # composing dataset

    test_dataset = CriketScoreDataSetWithCatAndNum(data, cfg.categorical_features, cfg.numerical_features,
                                                   cfg.response)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")

    model = FactorizationMachine(cat_dims=dims_categorical_vars, num_dim=dim_numerical_vars,
                                 k=cfg.embedding_dim, attention_dim=cfg.attention_dim).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.model_fp, f"{cfg.best_model_name}_target_val_"
                                                      f"{cfg.target_lower_limit}_{cfg.target_upper_limit}.pth")))
    model.eval()

    with torch.no_grad():  # Disable gradient computation
        predictions = list(map(lambda x: abs(int(x)), inference(model, test_loader, device)))

    logging.info(f"Below are the inference results given your input data, shape {len(predictions)}:")
    logging.info(f"Top 100, remaining truncated, \n{predictions[:100]}")
    mae = MeanAbsoluteError()
    mae = mae(torch.Tensor(predictions), torch.Tensor(data[cfg.response]))
    logging.info(f"Mean inference score: \nMAE: {mae}")


if __name__ == '__main__':
    main()

