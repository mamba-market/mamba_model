"""train linear neural network models"""
import os
import sys
sys.path.append('/Users/tianlongxu/mamba_model')
import numpy as np
import pandas
from datetime import datetime
import logging
import re
import hydra
from omegaconf import DictConfig
import torch
from torch.nn import CrossEntropyLoss
from sklearn.utils import shuffle
from models.linear_nn import CricketScorePredictor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm



logging.basicConfig(level=logging.INFO)

responses = {'Batting': ['runs',  'fours',   'sixes'], 'Bowling': ['wickets', 'fours_conceded', 'sixes_conceded']}
raw_features = {'Batting': ['period', 'batted', 'innings',
       'innings_number', 'rain_rule', 'batting_position',
       'away_team_id', 'home_team_id', 'floodlit', 'ground_id',
       'ground_latitude', 'ground_longitude', 'toss_winner_team_id', 'town_id',
       'weather_location_code', 'winner_team_id',
       'short_description', 'date_of_birth', 'position'],
                'Bowling': ['period',
       'innings_number', 'innings_bowled', 'floodlit', 'rain_rule',
       'bowling_position', 'away_team_id', 'home_team_id',
       'ground_id', 'ground_latitude', 'ground_longitude', #'rain_rule_name',
       'toss_winner_team_id', 'town_id', 'weather_location_code',
       'winner_team_id', 'short_description',
       'date_of_birth', 'position']}

categorical_features_raw = {'Batting': ['period', 'batted', 'innings',
       'innings_number', 'rain_rule', 'batting_position',
       'away_team_id', 'home_team_id', 'floodlit', 'ground_id', 'toss_winner_team_id', 'town_id',
       'weather_location_code', 'winner_team_id',
       'short_description', 'position', 'age_category'],
                        'Bowling': ['period',
       'innings_number', 'innings_bowled', 'floodlit', 'rain_rule',
       'bowling_position', 'away_team_id', 'home_team_id',
       'ground_id', #'rain_rule_name',
       'toss_winner_team_id', 'town_id', 'weather_location_code',
       'winner_team_id', 'style_type', 'position', 'age_category']}

original_numerical_features_raw = {'Batting': ['ground_latitude', 'ground_longitude'],
                        'Bowling': ['ground_latitude', 'ground_longitude']}

def categorize_age(date_of_birth):
    # Calculate age
    today = datetime.now()
    age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))

    # Categorize age
    if age <= 27:
        return '<=27'
    elif 27 < age <= 35:
        return '>27< 35'
    else:
        return '>35'


def convert_to_int_if_numeric(s):
    # Check if the string represents a floating-point number or an integer
    s = str(s)
    if re.match(r'^-?\d+\.\d+$', s) or re.match(r'^-?\d+$', s):
        # Convert to float first to handle both integers and floats,
        # then to int to remove any decimals
        return str(int(float(s)))
    # Return the original string if it doesn't represent a numeric value
    return s


read_from_master = False
test_phase = '2023'
game_type = 'Batting'
response = responses[game_type]
sorting_features = ['player_id', 'start_time']

categorical_features = categorical_features_raw[game_type]
original_numerical_features = original_numerical_features_raw[game_type]


def perform_feature_tranformation(data, sorting_features):
    data = data[sorting_features + response + categorical_features + original_numerical_features].copy()
    data = data.loc[~data[categorical_features + original_numerical_features].isna().any(axis=1), :].copy()
    data.sort_values(sorting_features, inplace=True)  # Sort by player_id and match_date
    data.reset_index(inplace=True, drop=True)
    for feature in categorical_features:
        data[feature] = data[feature].apply(convert_to_int_if_numeric)
        # data[feature] = data[feature].astype(str)
    stats_terms = ['rolling_min_', 'rolling_max_', 'cumulative_avg_', 'rolling_avg_']
    # Calculate cumulative averages and rolling averages for the past five matches
    for feature in response:
        data[f'cumulative_avg_{feature}'] = data.groupby('player_id')[feature].expanding().mean().reset_index(level=0,
                                                                                                              drop=True)
        data[f'rolling_avg_{feature}'] = data.groupby('player_id')[feature].rolling(window=5,
                                                                                    min_periods=1).mean().reset_index(
            level=0, drop=True)
        data[f'rolling_min_{feature}'] = data.groupby('player_id')[feature].rolling(window=5,
                                                                                    min_periods=1).min().reset_index(
            level=0, drop=True)
        data[f'rolling_max_{feature}'] = data.groupby('player_id')[feature].rolling(window=5,
                                                                                    min_periods=1).max().reset_index(
            level=0, drop=True)
    hist_response_stats = [stats_term + response_cur for stats_term in stats_terms for response_cur in response]
    organic_numerical_features = original_numerical_features + hist_response_stats
    transform_terms = ['_sqrt', '_log1', '_inverse', '_exp']
    derived_features = [numerical_feature + transform_term for numerical_feature in organic_numerical_features for
                        transform_term in transform_terms]
    for feature in organic_numerical_features:
        data[f'{feature}_sqrt'] = np.sqrt(data[feature])
        data[f'{feature}_log1'] = np.log1p(data[feature])
        data[f'{feature}_inverse'] = 1 / (data[feature] + 1)
        data[f'{feature}_exp'] = np.exp(data[feature])
    all_features = categorical_features + organic_numerical_features + derived_features
    data.reset_index(inplace=True, drop=True)
    return data.copy()


if not read_from_master:
    train_df, test_df = pandas.read_excel('data/IPL_Tianlong.xlsx', sheet_name=f'{game_type} Training Data'), \
        pandas.read_excel('data/IPL_Tianlong.xlsx', sheet_name=f'{game_type} Testing Data')
    train_df['age_category'] = train_df['date_of_birth'].apply(categorize_age)
    test_df['age_category'] = test_df['date_of_birth'].apply(categorize_age)

    train_data, test_data = perform_feature_tranformation(train_df, sorting_features=sorting_features), \
    perform_feature_tranformation(test_df, sorting_features=sorting_features)

else:
    df = pandas.read_excel('data/Final_Data_Sheet_IPL.xlsx', sheet_name = 'IPL Master Data Sheet')
    raw_data = {}
    for k, v in responses.items():
        raw_data[k] = df[df['style_type'] == str(k).lower()].copy()
        raw_data[k]['age_category'] = raw_data[k]['date_of_birth'].apply(categorize_age)

    data = raw_data[game_type]
    data.reset_index(inplace=True, drop=True)
    data, _ = perform_feature_tranformation(data, sorting_features=sorting_features)

    train_data = data[data['start_time'] < f'{test_phase}-01-01 00:00:00'].copy()
    test_data = data[data['start_time'] >= f'{test_phase}-01-01 00:00:00'].copy()


train_players, test_players = set(train_data.player_id), set(test_data.player_id)
common_players = train_players & test_players
test_data = test_data[test_data['player_id'].isin(common_players)]

train_data.reset_index(inplace=True, drop=True)
test_data.reset_index(inplace=True, drop=True)

train_data = pandas.get_dummies(train_data, columns=categorical_features).copy()
test_data = pandas.get_dummies(test_data, columns=categorical_features).copy()

x_cols = []

for col in train_data.columns:
    if col not in response + sorting_features:
        x_cols.append(col)

for col in train_data.columns:
    if col not in test_data.columns:
        print(col)
        test_data[col] = False

test_data = test_data[train_data.columns].copy()
x_cols.sort()
train_data = train_data[response + sorting_features + x_cols].copy()
test_data = test_data[response + sorting_features + x_cols].copy()

# test_data.isna().any(axis=1).sum()

# x_cols = ['a numbe of one-hot encoded categorical features and numerical features']
X_train, X_test, y_train, y_test = train_data[x_cols].values, test_data[x_cols].values, train_data[response].values, test_data[response].values
X_train, X_test, y_train, y_test = X_train.astype(float), X_test.astype(float), y_train.astype(float), y_test.astype(float)

x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)



# Converting data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Creating DataLoader instances
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


learning_rate = 1e-4
batch_size = 64
dropout_rate = 0.25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size=187, output_size=3, dropout_rate=0.5):
        super(NeuralNet, self).__init__()
        # Adjusted layer sizes to start from the input size and gradually reduce to the output size
        self.layer1 = nn.Linear(input_size, 256)  # First layer increased to match closer to input size
        self.norm1 = nn.LayerNorm(256)  # Applying Layer Normalization
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(256, 128)  # Intermediate layer
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(128, 64)  # Intermediate layer
        self.norm3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(64, 32)  # Additional layer for deeper representation
        self.norm4 = nn.LayerNorm(32)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(32, output_size)  # Output layer matching the output size

    def forward(self, x):
        x = F.relu(self.norm1(self.layer1(x)))
        x = self.dropout1(x)
        x = F.relu(self.norm2(self.layer2(x)))
        x = self.dropout2(x)
        x = F.relu(self.norm3(self.layer3(x)))
        x = self.dropout3(x)
        x = F.relu(self.norm4(self.layer4(x)))
        x = self.dropout4(x)
        x = self.output_layer(x)
        return x


# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
output_size = len(response)  # 'runs', 'fours', 'sixes'
model = NeuralNet(input_size, output_size, dropout_rate=dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Example weight decay for L2 regularization


# Training the model

# Early stopping parameters
early_stopping_patience = 3
min_val_loss = np.inf
epochs_no_improve = 0
early_stop = False

# Training loop with early stopping
best_model = None
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(test_loader)

    # Print progress
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # Check for early stopping
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        epochs_no_improve = 0
        best_model = deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            early_stop = True
            model.load_state_dict(best_model)
            break

if not early_stop:
    model.load_state_dict(best_model)

# Evaluate on the test set
model.eval()  # Set the model to evaluation mode

all_predictions = []
all_gts = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)

        # Inverse transform the outputs and targets
        predictions = y_scaler.inverse_transform(outputs.cpu().numpy())
        gts = y_scaler.inverse_transform(targets.cpu().numpy())

        # Append predictions and ground truths to lists
        all_predictions.append(predictions)
        all_gts.append(gts)

# Concatenate lists of arrays into single matrices
all_predictions_matrix = np.concatenate(all_predictions, axis=0)
all_gts_matrix = np.concatenate(all_gts, axis=0)

# Now, you can calculate the MAE for each response variable over the entire test set
for axis in range(len(response)):
    print('MAE:', response[axis], mean_absolute_error(all_gts_matrix[:, axis], all_predictions_matrix[:, axis]))



# per player
def calculate_odds(row, item):
    # Calculate mean and standard deviation of historical scores
    mean_score = sum(row[f'rolling_avg_{item}_hist']) / len(row[f'rolling_avg_{item}_hist'])
    std_dev = (sum((score - mean_score) ** 2 for score in row[f'rolling_avg_{item}_hist']) / len(
        row[f'rolling_avg_{item}_hist'])) ** 0.5

    # Avoid division by zero in case of no variation in historical scores
    if std_dev == 0:
        return float('inf') if mean_score > row[f'{item}_threshold'] else 0

    # Calculate the Z-score for the threshold
    z_score = (row[f'{item}_pred'] - mean_score) / std_dev

    # Calculate probabilities using the CDF
    prob_above_threshold = norm.cdf(z_score)  # Probability of being below the threshold
    prob_below_threshold = 1 - prob_above_threshold  # Probability of being above the threshold

    # Calculate odds
    if prob_below_threshold > 0:
        odds = prob_above_threshold / prob_below_threshold
    else:
        odds = float('inf')

    return odds



inference_data = test_data.copy()
rolling_avg_cols = [f"rolling_avg_{item}" for item in response]
inference_data_with_hist = pandas.merge(inference_data, train_data.groupby('player_id', as_index=False).agg(list)[['player_id'] + rolling_avg_cols],
             on='player_id', suffixes=('', '_hist')).copy()

for i, item in enumerate(response):
    inference_data_with_hist[f'{item}_pred'] = all_predictions_matrix[:, i]
    inference_data_with_hist[f'{item}_threshold'] = inference_data_with_hist[f'rolling_avg_{item}_hist'].apply(np.mean)
    inference_data_with_hist[f'{item}_odd'] = inference_data_with_hist.apply(lambda x: calculate_odds(x, item), axis=1)

inference_data_with_hist_show = inference_data_with_hist.iloc[:, -12:]

