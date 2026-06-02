
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os
import time

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_qvalue_map


class STLQN(Agent):
    
    """Stochastic Langevin Q-Learning (STLQN) implementation.
    """
    
    def __init__(self, 
        q_net: str = "continuous_mlp2",
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.2, 
        beta: float = 0.0, 
        autopessimism: bool = False,
        q_dropout: float = 0.0, 
        num_lsteps: int = 5,
        eta: float = 1.0,
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        q_lr: float = 3e-4, 
        beta_lr: float = 3e-4,
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._gamma = gamma
        self._target_entropy = target_entropy
        self._alpha = alpha
        self._beta = beta
        self._autopessimism = autopessimism
        self._q_dropout = q_dropout
        self._num_lsteps = num_lsteps
        self._eta = eta
        # a2 = 1 / self._eta # a2 is average squared score norm. 
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._beta_lr = beta_lr
        # networks
        self._q = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q_target = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        # no grad for target networks
        for param in self._q_target.parameters():
            param.requires_grad = False
        self._hard_update(self._q, self._q_target)
        # optimizers
        self._construct_optimizers()
        # sample fields
        self._sample_fields = ("observation", "action", "reward", "next_observation", "done")
        
    def save_ckpt(self, path: os.PathLike):
        th.save(self._q.state_dict(), os.path.join(path, "q.pth"))

    def load_ckpt(self, path: os.PathLike):
        self._q.load_state_dict(th.load(os.path.join(path, "q.pth"), map_location=self.device))
        self._hard_update(self._q, self._q_target)

    @property
    def derived_fields(self):
        """There is no derived field of STLQN algorithm. 
        """
        return ()
    
    def langevin_policy(self, observation: th.Tensor):
        action_noise = th.tanh(th.randn((observation.shape[0], *self._env.action_space.shape), device=self._device)) # initial noise
        for _ in range(self._num_lsteps):
            action_noise.requires_grad_(True)
            q_distr = self._q(observation, action_noise)
            q_value = q_distr.mean - self._beta * q_distr.stddev
            q_grad = th.autograd.grad(q_value.sum(), action_noise, create_graph=True)[0]
            #logprob_grad = q_grad / self._alpha
            action_noise += self._eta / self._alpha * q_grad + th.sqrt(2 * self._eta) * th.randn_like(action_noise)
            #action_noise.grad.zero_()
            action_noise.detach()
            # self._alpha = (1 - self._tau) * self._alpha + self._tau * self._eta * q_grad.pow(2).mean().item() / 2 # adaptive temperature adjustment
        return action_noise.detach()

    def step_torch(self, observation: th.Tensor, exploit: bool = False):
        action = self.langevin_policy(observation) # iterative action sampling..
        q_distr = self._q(observation, action)
        value = q_distr.mean.squeeze(-1)
        return action, None, value

    @th.no_grad()
    def step(self, observation: np.ndarray, exploit: bool = False):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, _, value_ = self.step_torch(observation_, exploit=exploit)
        action = action_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        if self._total_env_interactions < self._start_steps and not exploit:
            action = None        
        return action, None, value

    def episode_end(self):
        pass
    
    def reset(self):
        pass

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)

    def _hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def learn_on_step(self):
        for i in range(self._batch_per_step): 
            self._total_grad_steps += 1
            observation, action, reward, next_observation, done \
                = self.memory.sample(self._sample_fields, self._batch_size)
            with th.no_grad():
                next_action = self.langevin_policy(next_observation)
                next_q_distr = self._q_target(next_observation, next_action)
                next_value = (next_q_distr.mean - self._beta * next_q_distr.stddev ) * done.logical_not().unsqueeze(-1)
                #next_value_var = next_q_distr.variance * done.logical_not().unsqueeze(-1)
                q_target = reward.unsqueeze(-1) + self._gamma * next_value
                #q_target_var = self._gamma**2 * next_value_var
            # critic learning behavioral policy 
            self._q_optim.zero_grad()
            q_distr = self._q(observation, action)
            q_ce = - q_distr.log_prob(q_target)
            q_loss = q_ce.mean()
            self.log("q_loss", q_loss.item())
            self.log("q_avg", q_distr.mean.mean().item())
            self.log("q_std_avg", q_distr.stddev.mean().item()) 
            q_loss.backward()
            self._q_optim.step()          
            # soft update
            self._soft_update(self._q, self._q_target)

            if self._autopessimism: # EXPERIMENTAL: adjust beta for autopessimism
                pass

    def learn_on_epoch(self):
        pass
    
    @property
    def hparams(self):
        param = {
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "beta": self._beta, 
            "autopessimism": self._autopessimism,
            "q_dropout": self._q_dropout, 
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "beta_lr": self._beta_lr,
            "num_lsteps": self._num_lsteps,
            "eta": self._eta,
        }
        return param
    
    def _construct_optimizers(self):
        """Initialize Adam optimizer."""
        self._q_optim = Adam(self._q.parameters(), lr=self._q_lr)

    def train_mode(self):
        self._q.train()

    def eval_mode(self):
        self._q.eval()

    def compute_function(self,
        observation, 
        action, 
        reward, 
        next_observation, 
        done, 
        truncated, 
        log_prob, 
        value,         
    ):
        return ()

    def experiment_end(self): 
        pass
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Agent.add_model_specific_args(parser)
        parser.add_argument("--q_net", type=str, default="continuous_mlp2")
        parser.add_argument("--target_entropy", type=float, default=-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--beta", type=float, default=0.0)
        parser.add_argument("--autopessimism", action="store_true", default=False)
        parser.add_argument("--q_dropout", type=float, default=0.0)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--beta_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_lsteps", type=int, default=5)
        parser.add_argument("--eta", type=float, default=0.01)
        return parser

if __name__=="__main__":
    pass