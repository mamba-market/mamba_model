"""Linear neural network model"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CricketScorePredictor(nn.Module):
    def __init__(self, input_size):
        super(CricketScorePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)  # First hidden layer
        self.layer2 = nn.Linear(128, 64)  # Second hidden layer
        self.layer3 = nn.Linear(64, 32)  # Third hidden layer
        self.output_layer = nn.Linear(32, 3)  # Output layer for 'runs', 'fours', 'sixes'

    def forward(self, x):
        x = F.relu(self.layer1(x))  # Activation function for first layer
        x = F.relu(self.layer2(x))  # Activation function for second layer
        x = F.relu(self.layer3(x))  # Activation function for third layer
        x = self.output_layer(x)  # No activation function, as we're doing regression
        return x

