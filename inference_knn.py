"""This module implements the model (FM, factorization machine) training process"""
import os
import numpy
import pandas
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
import torch
from models.utils import LabelEncoder, Standardizer

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    inference_input_fp = cfg.inference_input_fp
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip(
        "data/")
    data_type = 'ODI'
    training_input_fp = f"data/{data_type}_sampled_cricket_training_data_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.csv"
    inference_output_fp = f"inference_results/game_preds_{datetime.strftime(datetime.utcnow(), '%Y-%m-%d')}.csv"
    cfg.best_model_name = 'knn'

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
    data = data.loc[~data[list(cfg.categorical_features) + list(cfg.numerical_features)].isna().any(axis=1)].copy()
    data.reset_index(inplace=True, drop=True)
    data_original = data.copy()
    bins = pandas.IntervalIndex.from_breaks(list(cfg.classification_target_threshold), closed='left')
    training_data[cfg.response_binary] = pandas.cut(training_data[cfg.response], bins=bins, labels=list(map(str, bins)))
    training_data[cfg.response_binary] = training_data[cfg.response_binary].apply(str)
    logging.info(f"Len of training data {len(training_data)}")
    all_inference_players = data['player_id'].unique()
    predictions = []
    training_supports = []
    for player in all_inference_players:
        logging.info(f"Now processing player: {player}")
        curr_data = data[data['player_id'] == player].copy()
        if player in training_data['player_id'].unique().tolist():
            curr_training_data = training_data[training_data['player_id'] == player].copy()
            logging.info(f"Found player, training data breaking down from {len(training_data)} to {len(curr_training_data)} records..")
            training_support = len(curr_training_data)
        else:
            curr_training_data = training_data.copy()
            training_support = 0
            logging.warning(f"Unfound player id: {player}, using the whole training data instead..")
        curr_data['training_support'] = training_support
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(cfg.numerical_features)),
                ('cat', OneHotEncoder(handle_unknown='ignore'), list(cfg.categorical_features))
            ])

        if cfg.model_stage == 'regression':
            knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', KNeighborsRegressor(n_neighbors=5))])

            # Fitting the model
            response_col = cfg.response
        else:
            knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', KNeighborsClassifier(n_neighbors=5))])
            response_col = cfg.response_binary
        knn_pipeline.fit(curr_training_data[list(cfg.numerical_features) + list(cfg.categorical_features)], curr_training_data[response_col])
        predictions_for_this_player = knn_pipeline.predict(curr_data[list(cfg.numerical_features) + list(cfg.categorical_features)])
        predictions.extend(predictions_for_this_player)
        training_supports.extend(curr_data['training_support'].tolist())
    # predictions = response_scaler.inverse_transform(predictions)

    logging.info(f"Below are the inference results given your input data, shape {len(predictions)}:")
    logging.info(f"Top 100, remaining truncated, \n{predictions[:100]}.")
    if os.path.exists(inference_output_fp): ## append current models results to existing inference CSV.
        results = pandas.read_csv(inference_output_fp)
    else:
        results = data_original
    results[f'predicted_{cfg.best_model_name}_{cfg.model_stage}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}'] = predictions
    results['training_support'] = training_supports
    results.to_csv(inference_output_fp, index=False)



if __name__ == '__main__':
    main()

