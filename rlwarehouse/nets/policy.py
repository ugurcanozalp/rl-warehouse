import torch as th
from torch import nn
from .heads import *


class ContinuousMLP2Policy(nn.Module):
    independent_actions = True
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLP2Policy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(observation_shape[0], 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.p_fc = nn.Linear(256, action_shape[0])

    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.p_fc(x)
        act = th.tanh(x)
        return act
