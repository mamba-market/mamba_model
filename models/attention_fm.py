"""This module implements the custom attention-aware factorization machine model"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DeepFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embedding_dim, num_numerical, hidden_units, n_classes=21):
        super(DeepFactorizationMachine, self).__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.n_fields = len(field_dims)

        # Embeddings for categorical features
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = torch.tensor((0, *torch.cumsum(torch.tensor(field_dims, dtype=torch.long), 0)[:-1]),
                                    dtype=torch.long)

        # Numerical feature processing layer
        self.fc_num = nn.Linear(num_numerical, embedding_dim)

        # Hidden layers
        input_size = embedding_dim * (self.n_fields + 1)  # +1 for the numerical features
        layers = []
        for dim in hidden_units:
            layers.append(nn.Linear(input_size, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))  # Dropout for regularization, can adjust as needed
            input_size = dim
        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer
        self.output_layer = nn.Linear(hidden_units[-1], n_classes)

    def forward(self, x_cat, x_num):
        """
        x_cat: Long tensor of size (batch_size, n_fields) for categorical features
        x_num: Float tensor of size (batch_size, num_numerical) for numerical features
        """
        # Embedding for categorical features
        x_cat = x_cat + self.offsets.clone().detach().unsqueeze(0)
        embed_cat = self.embedding(x_cat)

        # Numerical feature processing
        embed_num = F.relu(self.fc_num(x_num))

        # Concatenate processed numerical features and embeddings
        all_features = torch.cat([embed_cat.view(embed_cat.size(0), -1), embed_num], 1)

        # Pass through hidden layers
        features = self.hidden_layers(all_features)

        # Output layer
        logits = self.output_layer(features)

        return F.softmax(logits, dim=1)


class DeepFactorizationMachineRegression(nn.Module):
    def __init__(self, field_dims, embedding_dim, num_numerical, hidden_units):
        super(DeepFactorizationMachineRegression, self).__init__()

        # Categorical embeddings
        self.embedding = nn.Embedding(sum(field_dims), embedding_dim)
        self.embed_output_size = len(field_dims) * embedding_dim

        # Fully connected layers for combined embeddings and numerical fields
        mlp_input_dim = self.embed_output_size + num_numerical
        self.mlp = MLP(mlp_input_dim, hidden_units)

        # Regression output
        self.linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x_categorical, x_numerical):
        """
        x_categorical: Tensor of shape [batch_size, num_fields]
        x_numerical: Tensor of shape [batch_size, num_numerical_features]
        """
        embeds = self.embedding(x_categorical)
        embeds = embeds.view(-1, self.embed_output_size)

        # Concatenate embeddings and numerical features
        x = torch.cat([embeds, x_numerical], dim=1)

        x = self.mlp(x)
        x = self.linear(x)

        return x.squeeze(1)  # Regression output


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        input_units = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(input_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_units = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


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


