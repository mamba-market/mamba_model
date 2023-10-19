"""This module implements the custom attention-aware factorization machine model"""
import os
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Attention layer
    """
    def __init__(self, embed_size, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.fc = nn.Linear(embed_size, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))

    def forward(self, x):
        # x: tensor of shape (batch_size, num_embeddings, embed_size)
        attention_scores = torch.tanh(self.fc(x))
        attention_scores = torch.matmul(attention_scores, self.context_vector)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(x * attention_weights.unsqueeze(2), dim=1)
        return weighted_sum


class FactorizationMachine(nn.Module):
    """
    Main factorization machine module
    """
    def __init__(self, cat_dims, num_dim, k, attention_dim, dropout_rate=0.5):
        super(FactorizationMachine, self).__init__()

        # Embedding layers for categorical variables
        self.embeddings = nn.ModuleList([nn.Embedding(dim, k) for dim in cat_dims])

        # Attention layer for embeddings
        self.attention = Attention(k, attention_dim)

        # Linear layer for numerical variables
        self.linear = nn.Linear(num_dim, 1)

        # Factor matrix for pairwise interactions
        self.V = nn.Parameter(torch.randn(num_dim, k))

        # Dropout and BatchNorm
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(num_dim)

    def forward(self, cat_inputs, num_inputs):
        # Get embeddings for categorical features
        embeds = [embedding(cat_inputs[:, i]).unsqueeze(1) for i, embedding in enumerate(self.embeddings)]
        cat_embed = torch.cat(embeds, 1)  # Shape: (batch_size, num_embeddings, embed_size)

        # Apply attention
        weighted_embeddings = self.attention(cat_embed)

        # Linear term for numerical inputs
        linear_term = self.linear(num_inputs)

        # Pairwise interactions for numerical features
        inter_term = 0.5 * torch.sum(
            torch.pow(torch.matmul(num_inputs, self.V), 2) -
            torch.matmul(torch.pow(num_inputs, 2), torch.pow(self.V, 2)),
            1
        )

        # Combine everything
        y_pred = linear_term + inter_term + torch.sum(weighted_embeddings, 1)
        y_pred = y_pred.mean(axis=1)

        return y_pred



class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='chkpt', checkpoint_filename='model.pth'):
        """
        :param patience: How many epochs to wait for improvement before stopping.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param checkpoint_path: Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.checkpoint_filename = checkpoint_filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.checkpoint_path, self.checkpoint_filename))
        print(f'Validation loss decreased: best score {self.best_score:.6f}, loss {val_loss:.6f}.  Saving model ...')


