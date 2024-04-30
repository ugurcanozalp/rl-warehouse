
import torch as th
from torch import nn
from .heads import *


def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=1)
        th.nn.init.constant_(m.bias, 0)

        
class ContinuousMLPQValue(nn.Module):
    def __init__(self, observation_shape, action_shape, **kwargs):
        super(ContinuousMLPQValue, self).__init__()
        num_inputs = observation_shape[0] + action_shape[0]
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU())
        self.q_fc = nn.Linear(256, 1)
        self.apply(weights_init_)
        
    def forward(self, observation, action):
        x = th.concat([observation, action], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.q_fc(x)
        return x
