
import torch as th
from torch import nn
from .heads import *


def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=1)
        th.nn.init.constant_(m.bias, 0)

        
class ContinuousMLPStochasticValue(nn.Module):
    def __init__(self, observation_shape, dropout=0, **kwargs):
        super(ContinuousMLPStochasticValue, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(observation_shape[0], 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.v_fc = nn.Linear(256, 2)
        self.v_head = GaussianHead(1)
        self.apply(weights_init_)
        
    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.v_fc(x)
        v_dist = self.v_head(x)
        return v_dist
