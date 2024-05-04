
import torch as th
from torch import nn
from .heads import *

        
class ContinuousMLPValue(nn.Module):
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLPValue, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(observation_shape[0], 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.v_fc = nn.Linear(256, 1)
        
    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.v_fc(x)
        return x.squeeze(-1)

