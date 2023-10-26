"""modeling utils"""
import os
import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.fm import FactorizationMachineModel
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        outputs = model(cat_data, num_data)#.squeeze()
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
            outputs = model(cat_data, num_data)#.squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def assemble_true_labels_and_predictions(model, dataloader, device):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for cat_data, num_data, targets in dataloader:
            cat_data, num_data, targets = cat_data.to(device), num_data.to(device), targets.to(device)
            outputs = model(cat_data, num_data).squeeze()

            true_labels.extend(targets.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    return true_labels, predictions


class WeightedMAELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedMAELoss, self).__init__()

    def forward(self, predictions, targets, weighted=False):
        if weighted:
            weights = compute_weights(targets.detach().cpu().numpy())
            weights = weights.to(predictions.device)  # Ensure weights are on the same device as predictions
            losses = F.l1_loss(predictions, targets, reduction='none')
            return (losses * weights / weights.sum()).mean()
        losses = F.l1_loss(predictions, targets, reduction='none')
        return losses.mean()



def compute_weights(targets, n_bins=10):
    # Bin the continuous target values
    kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    binned_targets = kbins.fit_transform(targets.reshape(-1, 1)).ravel()

    # Compute inverse bin frequencies
    bin_counts = numpy.bincount(binned_targets.astype(int))
    inv_bin_weights = 1. / (bin_counts + 1e-3)  # adding a small constant to avoid division by zero
    weights = inv_bin_weights[binned_targets.astype(int)]

    return torch.tensor(weights, dtype=torch.float32)


def inference(model, data_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for cat_data, num_data, targets in data_loader:
            cat_data, num_data, targets = cat_data.to(device), num_data.to(device), targets.to(device)
            outputs.extend(model(cat_data, num_data).squeeze())
    return outputs


def plot_losses(train_losses, test_losses, data_type = 'ODI', save_path='loss_curve'):
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
    plt.title(f'Training and Testing MAE Losses {data_type}')
    plt.savefig(os.path.join("results", f"{save_path}.png"))


def plot_confusion_matrix(conf_matrix, class_names, data_type = 'ODI', save_path='confusion_matrix'):
    plt.figure(figsize=(12, 9))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix {data_type}')
    plt.savefig(os.path.join("results", f"{save_path}.png"))


def evaluate_regression(true_values, predicted_values, convert_to_int=False):
    if convert_to_int:
        true_values = list(map(int, true_values))
        predicted_values = list(map(int, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2
    }


def plot_f1_score_and_confusion_matrix(true_labels, predicted_labels, class_names,
                                       data_type = 'ODI', save_path='confusion_matrix'):
    # Compute metrics
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(class_names))))

    # Display confusion matrix
    title = f'F1 (Micro): {f1_micro:.4f}, F1 (Macro): {f1_macro:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}'
    title += f"\n{data_type}"
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.savefig(os.path.join("results", f"{save_path}.png"))


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + [-20231029])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = []
        for element in data_list:
            if element in self.classes_:
                new_data_list.append(element)
            else:
                new_data_list.append(-20231029)
        return self.label_encoder.transform(new_data_list).tolist()


class Standardizer(object):
    def __init__(self):
        """
        A unified version of standardizer built on top of the sklearn StandardScaler() module
        """
        self.standardizer = StandardScaler()

    def fit(self, data_list):
        """
        This will fit the standardizer for the given list of data
        :param data_list: A list of numeric values
        :return: self
        """
        self.standardizer.fit(numpy.array(data_list).reshape(-1, 1))
        self.mean = self.standardizer.mean_[0]
        self.variance = self.standardizer.var_[0]
        return self

    def transform(self, data_list):
        """
        This will transform the data_list based on the mean and variance of the standardizer
        :param data_list:
        :return:
        """
        return self.standardizer.transform(numpy.array(data_list).reshape(-1, 1)).squeeze().tolist()

    def inverse_transform(self, data_list):
        """
        Inversely transform data to the original scale
        :param data_list:
        :return:
        """
        return self.standardizer.inverse_transform(numpy.array(data_list).reshape(-1, 1)).squeeze().tolist()


