
import torch as th
from torch import nn
from .heads import *


class ContinuousMLPStochasticPolicy(nn.Module):
    independent_actions = True
    def __init__(self, observation_shape, action_shape, dropout=0, **kwargs):
        super(ContinuousMLPStochasticPolicy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(observation_shape[0], 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.Dropout(dropout), nn.LayerNorm(256), nn.ReLU())
        self.p_fc = nn.Linear(256, 2*action_shape[0])
        self.p_head = SquashedGaussianHead(action_shape[0])
        # self.p_fc.bias.data.fill_(0)
        # suggested from: https://arxiv.org/pdf/2006.05990.pdf
        # self.p_fc.weight.data = 0.01 * self.p_fc.weight.data

    def forward(self, observation):
        x = observation
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.p_fc(x)
        act_distr = self.p_head(x)
        return act_distr
