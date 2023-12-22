import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanningTokenGenerator(nn.Module):
    def __init__(self, state_dim, planning_token_dim, seq_len, num_layers, state_embedding):
        self.seq_len = seq_len
        super(PlanningTokenGenerator, self).__init__()
        layers = []
        # Define the first layer
        layers.append(nn.Linear(state_dim, planning_token_dim))
        layers.append(nn.ReLU())

        # Define additional layers
        for _ in range(1, num_layers):
            layers.append(nn.Linear(planning_token_dim, planning_token_dim))
            layers.append(nn.ReLU())

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state).unsqueeze(1)
