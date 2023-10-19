"""modeling utils"""
import os
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.fm import FactorizationMachineModel


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(self.save_path, 'model.pth'))
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False



def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for cat_data, num_data, targets in data_loader:
        cat_data, num_data, targets = cat_data.to(device), num_data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(cat_data, num_data).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for cat_data, num_data, targets in data_loader:
            cat_data, num_data, targets = cat_data.to(device), num_data.to(device), targets.to(device)
            outputs = model(cat_data, num_data).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def inference(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for cat_data, num_data, targets in data_loader:
            cat_data, num_data, targets = cat_data.to(device), num_data.to(device), targets.to(device)
            outputs.extend(model(cat_data, num_data).squeeze())
    outputs = [float(output) for output in outputs]
    return outputs


def plot_losses(train_losses, test_losses, save_path='loss_curve'):
    os.makedirs("results", exist_ok=True)
    # Setting Seaborn style
    sns.set_style("whitegrid")
    # Create a pandas DataFrame for the data
    df_losses = pandas.DataFrame({
        'Epoch': list(range(1, len(train_losses) + 1)) * 2,
        'Loss': train_losses + test_losses,
        'Type': ['Train'] * len(train_losses) + ['Test'] * len(test_losses)
    })
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_losses, x='Epoch', y='Loss', hue='Type', marker='o')
    plt.title('Training and Testing Losses')
    plt.savefig(os.path.join("results", f"{save_path}.png"))

