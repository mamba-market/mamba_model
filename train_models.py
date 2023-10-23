"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from sklearn.utils import shuffle
from data.dataset import forming_train_and_test_data
from models.attention_fm import DeepFactorizationMachineRegression, EarlyStopping
from models.utils import WeightedMAELoss
from models.utils import train, evaluate, plot_losses, assemble_true_labels_and_predictions, \
    plot_f1_score_and_confusion_matrix, LabelEncoderExt, Standardizer, evaluate_regression

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    epochs = cfg.epoch
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    weight_decay = cfg.weight_decay
    input_fp = cfg.training_input_fp
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip(
        "data/")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using GPU.")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     logging.info("Using MPS")
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
    # le = LabelEncoder()
    # data[cfg.response] = le.fit_transform(data[cfg.response])
    # logging.info(f"Classes of response: {le.classes_}")
    standardizer = Standardizer()
    for col in cfg.numerical_features + [cfg.response]:
        standardizer = Standardizer()
        standardizer.fit(data[col])
        data[col] = standardizer.transform(data[col])

    dims_categorical_vars = [data[col].max() + 1 for col in cfg.categorical_features]
    dim_numerical_vars = len(cfg.numerical_features)
    logging.info(f"Total dims of categorical vars: \n{dims_categorical_vars}")
    logging.info(f"Dim of numerical vars: \n{dim_numerical_vars}")
    logging.info(f"Range of target value: {data[cfg.response].min(), data[cfg.response].max()}")
    # import ipdb; ipdb.set_trace()

    # model initialization
    model = DeepFactorizationMachineRegression(field_dims=dims_categorical_vars,
                                     embedding_dim=cfg.embedding_dim,
                                     num_numerical=dim_numerical_vars,
                                     hidden_units=cfg.hidden_layers).to(device)

    # class_weights = compute_weights(data[cfg.response].to_numpy(), n_bins=len(data[cfg.response].unique()))
    criterion = WeightedMAELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=3, checkpoint_path=cfg.model_fp,
                                  checkpoint_filename=f"{cfg.best_model_name}_dt_{data_type}_tz_"
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

    plot_losses(train_losses, test_losses, data_type=data_type, save_path=
    f"{cfg.loss_plot_fp}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}")
    y_trues, y_preds = assemble_true_labels_and_predictions(model, test_loader, device)
    y_trues, y_preds = standardizer.inverse_transform(y_trues), standardizer.inverse_transform(y_preds)
    logging.info(f'y_trues: {y_trues[:30]}')
    logging.info(f"y_preds: {y_preds[:30]}")
    metrics = evaluate_regression(y_trues, y_preds, convert_to_int=True)
    eval_metrics_fp = f"{cfg.eval_metrics_fp}_dt_{data_type}_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.txt"
    with open(os.path.join('results', eval_metrics_fp), 'w') as f:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            f.write('%s:%s\n' % (key, value))

    if cfg.plot_confusion_metrics_for_narrow_zone:
        plot_f1_score_and_confusion_matrix(y_trues, y_preds, sorted(data[cfg.response].unique()),
            data_type=data_type, save_path=f"{cfg.loss_plot_fp}_dt_{data_type}"
                                           f"_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}")


if __name__ == '__main__':
    main()

