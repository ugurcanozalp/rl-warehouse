
import torch as th
from torch import nn
from .heads import *


class ContinuousMLPEvidentialQValue(nn.Module):
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLPEvidentialQValue, self).__init__()
        num_inputs = observation_shape[0] + action_shape[0]
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.mu_fc = nn.Linear(256, 2)
        self.logvar_fc = nn.Linear(256, 2)
        #self.q_fc.bias.data[0].fill_(0)
        #self.q_fc.bias.data[1].fill_(10)
        self.mu_head = GaussianHead(1)
        self.logvar_head = GaussianHead(1)

    def forward(self, observation, action):
        x = th.concat([observation, action], dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        mu_param = self.mu_fc(x)
        logvar_param = self.logvar_fc(x)
        mu_distr = self.mu_head(mu_param)
        logvar_distr = self.logvar_head(logvar_param)
        return mu_distr, logvar_distr
