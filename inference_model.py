"""This module implements the model (FM, factorization machine) training process"""
import os
import numpy
import pandas
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
import torch
import xgboost as xgb
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import DeepFactorizationMachineClassification, DeepFactorizationMachineRegression
from models.utils import inference, LabelEncoderExt, Standardizer, assemble_true_labels_and_predictions

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    batch_size = cfg.batch_size
    inference_input_fp = cfg.inference_input_fp
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip(
        "data/")
    data_type = 'ODI'
    training_input_fp = f"data/{data_type}_sampled_cricket_training_data_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.csv"
    inference_output_fp = f"inference_results/game_preds_{datetime.strftime(datetime.utcnow(), '%Y-%m-%d')}.csv"
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
    logging.info(f"Eliminating {data.isna().any(axis=1).sum()} NA rows...")
    data = data.loc[~data.isna().any(axis=1), :].copy()
    data_original = data.copy()
    if cfg.response not in data.columns:
        data[cfg.response] = list(range(len(data)))
    data[cfg.response_binary] = data[cfg.response].apply(lambda x: 0 if x < cfg.classification_target_threshold else 1)

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
    standardizer = Standardizer()
    for col in cfg.numerical_features + [cfg.response]:
        standardizer = Standardizer()
        response_flag = True if (col == cfg.response) or (col in cfg.skewed_features) else False
        standardizer.fit(training_data[col])
        data[col] = standardizer.transform(data[col], response_flag)

    dims_categorical_vars = list(map(int, [training_data[col].max() + 1 for col in cfg.categorical_features]))
    dim_numerical_vars = len(cfg.numerical_features)
    print(dims_categorical_vars)
    logging.info(f"Inference: {len(data)}")
    # composing dataset
    response = cfg.response if cfg.model_stage == 'regression' else cfg.response_binary
    test_dataset = CriketScoreDataSetWithCatAndNum(data, cfg.categorical_features, cfg.numerical_features,
                                                   response)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")

    if cfg.best_model_name == 'xgboost':
        dtest = xgb.DMatrix(data[list(cfg.categorical_features) + list(cfg.numerical_features)])
        model = xgb.Booster()
        model.load_model(os.path.join(cfg.model_fp, f"{cfg.model_stage}_{cfg.best_model_name}_dt_{data_type}_"
                                                f"tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.json"))
        predictions = model.predict(dtest)
        if cfg.model_stage == 'classification':
            predictions = list(map(lambda p: 1 if p > 0.5 else 0, predictions))
    else:
        if cfg.model_stage == 'regression':
            model = DeepFactorizationMachineRegression(field_dims=dims_categorical_vars,
                                         embedding_dim=cfg.embedding_dim,
                                         num_numerical=dim_numerical_vars,
                                         hidden_units=cfg.hidden_layers).to(device)
        else:
            model = DeepFactorizationMachineClassification(field_dims=dims_categorical_vars,
                                                           embedding_dim=cfg.embedding_dim,
                                                           num_numerical=dim_numerical_vars,
                                                           hidden_units=cfg.hidden_layers,
                                                           n_classes=len(data[cfg.response_binary].unique()))

        print(os.path.join(cfg.model_fp, f"{cfg.model_stage}_{cfg.best_model_name}_dt_{data_type}_tz_"
                                                          f"{cfg.target_lower_limit}_{cfg.target_upper_limit}.pth"))
        model.load_state_dict(torch.load(os.path.join(cfg.model_fp, f"{cfg.model_stage}_{cfg.best_model_name}_dt_{data_type}_tz_"
                                                          f"{cfg.target_lower_limit}_{cfg.target_upper_limit}.pth")))
        model.eval()

        with torch.no_grad():  # Disable gradient computation
            if cfg.model_stage == 'regression':
                predictions = list(map(lambda x: float(x), inference(model, test_loader, device)))
                y_trues, predictions = standardizer.inverse_transform(data[response], True), standardizer.inverse_transform(predictions, True)
                predictions = list(map(int, predictions))
                mae = MeanAbsoluteError()
                mae = mae(torch.Tensor(predictions), torch.Tensor(y_trues))
                logging.info(f"Mean inference score: \nMAE: {mae}")
            else:
                predictions = list(map(lambda x: int(x.argmax()), inference(model, test_loader, device)))
                f1_micro = f1_score(data[response], predictions, average='micro')
                f1_macro = f1_score(data[response], predictions, average='macro')
                precision = precision_score(data[response], predictions, average='weighted')
                recall = recall_score(data[response], predictions, average='weighted')
                logging.info(f"Inference scores: \nF1 scores: {f1_micro}, {f1_macro}")
                logging.info(f"Precision and recall {precision}, {recall}")

    logging.info(f"Below are the inference results given your input data, shape {len(predictions)}:")
    logging.info(f"Top 100, remaining truncated, \n{predictions[:100]}, mean {numpy.mean(predictions)}")
    data_original[f'predicted_{cfg.best_model_name}_{cfg.response}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}'] = predictions
    if os.path.exists(inference_output_fp): ## append current models results to existing inference CSV.
        results = pandas.read_csv(inference_output_fp)
        results[f'predicted_{cfg.response}_{cfg.model_stage}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}'] = predictions
        if cfg.model_stage == 'classification': ## correct the regression results by zone
            col_tz_0_20, col_tz_20_70 = list(filter(lambda x: x.endswith('0_20'), results.columns))[0], \
                                        list(filter(lambda x: x.endswith('20_70'), results.columns))[0]
            for i, row in results.iterrows():
                if results.loc[i, f'predicted_{cfg.response}_{cfg.model_stage}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}'] == 0:
                    results.loc[i, col_tz_20_70] = None
                else:
                    results.loc[i, col_tz_0_20] = None
        results.to_csv(inference_output_fp, index=False)
    else:
        data_original.to_csv(inference_output_fp, index=False)


if __name__ == '__main__':
    main()

