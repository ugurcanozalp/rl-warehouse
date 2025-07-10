
import torch as th
from torch import nn
from .heads import *

        
class ContinuousMLPStochasticQValue(nn.Module):
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLPStochasticQValue, self).__init__()
        num_inputs = observation_shape[0] + action_shape[0]
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 256), nn.Dropout(dropout),nn.LayerNorm(256), nn.ReLU()) #  
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU()) # 
        self.q_fc = nn.Linear(256, 2)
        self.q_head = GaussianHead(1)
        self.q_fc.bias.data[1].fill_(0)

    def forward(self, observation, action):
        x = th.concat([observation, action], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.q_fc(x)
        q_distr = self.q_head(x)
        return q_distr
