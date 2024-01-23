
import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
import torch as th
import  pytorch_lightning as pl
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader

from ..agent import Agent
from ..nets import policy_map, value_map


# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
class PPO(Agent):
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        v_net: str = "continuous_mlp2",
        gamma: float = 0.99,
        lambd: float = 0.90, 
        alpha: float = 5e-4, 
        steps_per_rollout: int = 512, 
        subepochs_per_rollout: int = 20, 
        clip_ratio: int = 0.4, 
        pi_lr: float = 3e-5,
        v_lr: float = 3e-5, 
        vf_coef: float = 0.5, 
        max_grad_norm: float = 0.5, 
        batch_size: int = 128, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._gamma = gamma
        self._lambda = lambd
        self._alpha = alpha
        self._batch_size = batch_size
        self._clip_ratio = clip_ratio 
        self._vf_coef = vf_coef
        self._max_grad_norm = max_grad_norm
        self._steps_per_rollout = steps_per_rollout
        self._subepochs_per_rollout = subepochs_per_rollout
        # networks
        self._pi = policy_map[pi_net](**self.memory.env_info).to(self._device)
        self._v = value_map[v_net](**self.memory.env_info).to(self._device)
        # optimizers
        self._construct_optimizers(pi_lr, v_lr)

    @property
    def extra_fields(self):
        """On policy estimated log_prob and value is necessary for PPO
        """
        return ("log_prob", "value")

    @property
    def derived_fields(self):
        """Cumulative return and GAE should be computed at the end of the episode
        """
        return ("cum_return", "gae")
    
    def forward(self, observation: th.Tensor):
        distr = self._pi(observation)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        if self._pi.independent_actions: 
            log_prob = log_prob.sum(dim=-1)
        value = self._v(observation)
        return action, (log_prob, value)

    @th.no_grad()
    def step(self, observation: np.ndarray):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, (log_prob_, value_) = self.forward(observation_)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        return action, (log_prob, value)

    def reset(self):
        pass

    def learn_on_step(self):
        observation, action, reward, next_observation, \
            done, truncated, log_prob, value, \
            cum_return, gae  = self.memory.sample(self._batch_size)
        distr = self._pi(observation)
        cross_log_prob = distr.log_prob(action)
        entropy = - distr.log_prob(distr.rsample())
        if self._pi.independent_actions: 
            cross_log_prob = cross_log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        value = self._v(observation)
        # policy loss        
        ratio = th.exp(cross_log_prob - log_prob)
        gae_normalized = (gae - gae.mean() ) / (gae.std() + 1e-6)
        clip_adv = th.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * gae_normalized
        pi_loss = -(th.min(ratio * gae_normalized, clip_adv)).mean() - self._alpha * entropy.mean() 
        self._pi_optim.zero_grad()
        pi_loss.backward()
        th.nn.utils.clip_grad_norm_(self._pi.parameters(), self._max_grad_norm)
        self._pi_optim.step()
        # critic loss
        v_loss = self._vf_coef * th.nn.functional.mse_loss(value, cum_return)
        self._v_optim.zero_grad()
        v_loss.backward()
        th.nn.utils.clip_grad_norm_(self._v.parameters(), self._max_grad_norm)
        self._v_optim.step() 
        # useful extra info
        approx_kl = (log_prob - cross_log_prob).mean().item()
        clipped = th.logical_or(ratio.gt(1 + self._clip_ratio), ratio.lt(1 - self._clip_ratio))
        clipfrac = th.as_tensor(clipped, dtype=th.float32).mean().item()
        #
        self.log("pi_loss", pi_loss)
        self.log("v_loss", v_loss)
        self.log("approx_kl", approx_kl)
        self.log("clipfrac", clipfrac)
        self.log("cross_log_prob", cross_log_prob.mean())

    def _construct_optimizers(self, pi_lr, v_lr):
        """Initialize Adam optimizer."""
        self._pi_optim = AdamW(self._pi.parameters(), lr=pi_lr)
        self._v_optim = AdamW(self._v.parameters(), lr=v_lr)

    def compute_function(self,
        observation, 
        action, 
        reward, 
        next_observation, 
        done, 
        truncated, 
        log_prob, 
        value
    ):
        not_done = np.logical_not(done)
        cum_return, gae = np.zeros_like(reward), np.zeros_like(reward)
        _, _, last_value = self.step(next_observation[-1])
        cum_return[-1] = reward[-1] + not_done[-1] * self._gamma * last_value 
        delta = reward[-1] + self._gamma * last_value - value[-1] # delta at last time...
        gae[-1] = delta
        for t in reversed(range(self._not_computed-1)):
            if truncated[t]:
                cum_return[t] = value[t]
            else:
                cum_return[t] = reward[t] + not_done[t] * self._gamma * cum_return[t+1] 
            delta = reward[t] + not_done[t] * self._gamma * value[t+1] - value[t] # one step td error
            gae[t] = delta + not_done[t] * self._gamma * self._lambd * gae[t+1]
        return cum_return, gae
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Agent.add_model_specific_args(parser)
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2") 
        parser.add_argument("--v_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--lambd", type=float, default=0.90) 
        parser.add_argument("--alpha", type=float, default=5e-4)
        parser.add_argument("--steps_per_rollout", type=int, default=512) 
        parser.add_argument("--subepochs_per_rollout", type=int, default=20) 
        parser.add_argument("--clip_ratio", type=int, default=0.4) 
        parser.add_argument("--pi_lr", type=float, default=3e-5)
        parser.add_argument("--v_lr", type=float, default=3e-5)
        parser.add_argument("--vf_coef", type=float, default=0.5)
        parser.add_argument("--max_grad_norm", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=128)
        return parser

if __name__=="__main__":
    pass