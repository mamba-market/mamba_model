"""This module implements the model (FM, factorization machine) training process"""
import os
import pandas
import logging
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from data.dataset import CriketScoreDataSetWithCatAndNum
from models.attention_fm import FactorizationMachine, EarlyStopping
from models.utils import train, evaluate, plot_losses

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
    else:
        device = torch.device('cpu')
    # grabbing raw data
    if input_fp.endswith('.csv'):
        data = pandas.read_csv(input_fp)
    else:
        data = pandas.read_pickle(input_fp)
    for col in cfg.categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    dims_categorical_vars = [data[col].max() + 1 for col in cfg.categorical_features]
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
    train_dataset = CriketScoreDataSetWithCatAndNum(train_df, cfg.categorical_features, cfg.numerical_features,
                                                    cfg.response)
    test_dataset = CriketScoreDataSetWithCatAndNum(test_df, cfg.categorical_features, cfg.numerical_features,
                                                   cfg.response)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logging.info(f"Data loader assembled.")
    # model initialization
    model = FactorizationMachine(cat_dims=dims_categorical_vars, num_dim=dim_numerical_vars,
                                 k=10, attention_dim=50).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=3)
    logging.info("Model successfully initialized.")

    train_losses, test_losses = [], []
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

    torch.save(model.state_dict(), os.path.join(cfg.model_fp, cfg.best_model_name))
    plot_losses(train_losses, test_losses, save_path=cfg.loss_plot_fp)


if __name__ == '__main__':
    main()

