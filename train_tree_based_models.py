"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
from sklearn.utils import shuffle
from data.dataset import form_xgboost_train_and_test_data
import xgboost as xgb
from models.utils import plot_losses, plot_f1_score_and_confusion_matrix, evaluate_regression

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    input_fp = cfg.training_input_fp
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip(
        "data/")
    sampled_training_data_fp = f"data/{data_type}_sampled_cricket_training_data_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.csv"

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
    data[cfg.response_binary] = data[cfg.response].apply(lambda x: 0 if x < cfg.classification_target_threshold else 1)
    data.reset_index(inplace=True, drop=True)
    train_df, test_df, train_df_original, test_df_original, response_standardizer = \
        form_xgboost_train_and_test_data(cfg, data)
    train_df_original.to_csv(sampled_training_data_fp, index=False)

    params, dtrain, dtest = construct_model(cfg, train_df, test_df)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=3,
        evals=[(dtrain, 'Train'), (dtest, "Test")],
        evals_result=evals_result,
    )
    model.save_model(os.path.join(cfg.model_fp, f"{cfg.model_stage}_{cfg.best_model_name}_dt_{data_type}_"
                                                f"tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.json"))
    train_losses, test_losses = [], []
    eval_metrics_fp = f"{cfg.best_model_name}_{cfg.model_stage}_{cfg.eval_metrics_fp}_dt_{data_type}_" \
                      f"tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.txt"
    with open(os.path.join('results', eval_metrics_fp), "w") as file:
        for eval_stage, vals in evals_result.items():
            file.write(f"{eval_stage}\n")
            for key, val in vals.items():
                if eval_stage == 'Train':
                    for iteration, loss in enumerate(val):
                        train_losses.append(loss)
                        file.write(f"{iteration}\t{loss}\n")
                else:
                    for iteration, loss in enumerate(val):
                        test_losses.append(loss)
                        file.write(f"{iteration}\t{loss}\n")
    loss_metrics_fp = f"{cfg.best_model_name}_{cfg.model_stage}_{cfg.loss_plot_fp}_dt_{data_type}_" \
                      f"tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}"
    plot_losses(train_losses, test_losses, data_type=data_type, save_path=loss_metrics_fp)

    predictions = model.predict(dtest)

    if cfg.model_stage == 'regression':
        y_trues, y_preds = test_df_original[cfg.response], response_standardizer.inverse_transform(predictions, True)
        metrics = evaluate_regression(y_trues, y_preds, convert_to_int=True)
        logging.info(f'y_trues: {y_trues[:30]}')
        logging.info(f"y_preds: {y_preds[:30]}")
        eval_metrics_fp = f"{cfg.best_model_name}_{cfg.model_stage}_{cfg.eval_metrics_fp}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.txt"
        with open(os.path.join('results', eval_metrics_fp), 'w') as f:
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
                f.write('%s:%s\n' % (key, value))
    else:
        y_trues, y_preds = test_df_original[cfg.response_binary], list(map(lambda p: 1 if p > 0.5 else 0, predictions))
        logging.info(f'y_trues: {y_trues[:30]}')
        logging.info(f"y_preds: {y_preds[:30]}")
        plot_f1_score_and_confusion_matrix(y_trues, y_preds, sorted(data[cfg.response_binary].unique()),
                                           data_type=data_type, save_path=f"{cfg.best_model_name}_{cfg.model_stage}_{cfg.confusion_matrix_fp}_dt_{data_type}"
                                                            f"_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}")


def construct_model(cfg, train_df, test_df):
    loss_fn = 'reg:squarederror' if cfg.model_stage == 'regression' else 'binary:logistic'
    params = {
        'objective': loss_fn,
        'eval_metric': 'rmse',
        'eta': 5e-3,
        'max_depth': 20,
        'max_leaves': 25,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'alpha': 1.5,  # L1 regularization term on weights. Increasing this value makes models more conservative.
        'lambda': 2.0  # L2 regularization term on weights. Increasing this value makes models more conservative.
    }
    response = cfg.response if cfg.model_stage == 'regression' else cfg.response_binary
    X_train, y_train = train_df[list(cfg.categorical_features) + list(cfg.numerical_features)], train_df[response]
    X_test, y_test = test_df[list(cfg.categorical_features) + list(cfg.numerical_features)], test_df[response]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return params, dtrain, dtest


if __name__ == '__main__':
    main()

