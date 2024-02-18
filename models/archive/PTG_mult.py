import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanningTokenGenerator(nn.Module):
    def __init__(self, state_dim, planning_token_dim, num_layers, num_planning_tokens):
        super(PlanningTokenGenerator, self).__init__()
        self.num_planning_tokens = num_planning_tokens
        self.planning_token_dim = planning_token_dim

        # Initial layer
        layers = [nn.Linear(state_dim, planning_token_dim), nn.ReLU()]

        # Intermediate layers
        for _ in range(1, num_layers):
            layers += [nn.Linear(planning_token_dim, planning_token_dim), nn.ReLU()]

        # Final layer
        layers.append(nn.Linear(planning_token_dim, num_planning_tokens * planning_token_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        # Reshape the output for batch processing
        batch_size = state.size(0)
        output = self.model(state)
        return output.view(batch_size, self.num_planning_tokens, self.planning_token_dim)