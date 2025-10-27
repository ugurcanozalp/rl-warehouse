
import torch as th
from torch import nn
from .heads import *

        
class ContinuousMLP2Reward(nn.Module):
    independent_observations = True
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLP2Reward, self).__init__()
        num_inputs = observation_shape[0] + action_shape[0]
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.r_fc = nn.Linear(256, 1)
        
    def forward(self, observation, action):
        x = th.concat([observation, action], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.r_fc(x)
        return x
