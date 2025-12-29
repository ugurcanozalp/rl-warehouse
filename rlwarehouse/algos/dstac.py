
import random
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os
import time

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_policy_map, probabilistic_qvalue_map


class DSTAC(Agent):
    
    """Double Stochastic Actor Critic
    """
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = False, 
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.2, 
        beta: float = 0.0, 
        autopessimism: bool = False,
        pi_dropout: float = 0.0, 
        q_dropout: float = 0.0, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        policy_delay: int = 1, 
        pi_lr: float = 3e-4, 
        q_lr: float = 3e-4, 
        beta_lr: float = 3e-4,
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._gamma = gamma
        self._autotune = autotune
        self._target_entropy = target_entropy
        self._alpha = alpha
        self._beta = beta
        self._autopessimism = autopessimism
        self._pi_dropout = pi_dropout
        self._q_dropout = q_dropout
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._policy_delay = policy_delay
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        self._beta_lr = beta_lr
        # networks
        self._pi = probabilistic_policy_map[pi_net](dropout=self._pi_dropout, **self.env_info).to(self._device)
        self._q1 = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q2 = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q1_target = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q2_target = probabilistic_qvalue_map[q_net](dropout=self._q_dropout, **self.env_info).to(self._device)
        # no grad for target networks
        for param in self._q1_target.parameters():
            param.requires_grad = False
        for param in self._q2_target.parameters():
            param.requires_grad = False
        self._hard_update(self._q1, self._q1_target)
        self._hard_update(self._q2, self._q2_target)
        # optimizers
        self._construct_optimizers()
        # sample fields
        self._sample_fields = ("observation", "action", "reward", "next_observation", "done")
        
    def save_ckpt(self, path: os.PathLike):
        th.save(self._pi.state_dict(), os.path.join(path, "pi.pth"))
        th.save(self._q1.state_dict(), os.path.join(path, "q1.pth"))
        th.save(self._q2.state_dict(), os.path.join(path, "q2.pth"))

    def load_ckpt(self, path: os.PathLike):
        self._pi.load_state_dict(th.load(os.path.join(path, "pi.pth"), map_location=self.device))
        self._q1.load_state_dict(th.load(os.path.join(path, "q1.pth"), map_location=self.device))
        self._q2.load_state_dict(th.load(os.path.join(path, "q2.pth"), map_location=self.device))
        self._hard_update(self._q1, self._q1_target)
        self._hard_update(self._q2, self._q2_target)

    @property
    def derived_fields(self):
        """There is no derived field of DSTAC algorithm. 
        """
        return ()
    
    def step_torch(self, observation: th.Tensor, exploit: bool = False):
        distr = self._pi(observation)
        if exploit:
            action = distr.rsample((10, )).mean(dim=0) # averaged action
        else:
            action = distr.rsample()
        log_prob = distr.log_prob(action)
        if self._pi.independent_actions: 
            log_prob = log_prob.sum(dim=-1)            
        q1_distr = self._q1(observation, action)
        q2_distr = self._q2(observation, action)
        value = 0.5*(q1_distr.mean + q2_distr.mean).squeeze(-1) + self._alpha * (-log_prob)
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation: np.ndarray, exploit: bool = False):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, log_prob_, value_ = self.step_torch(observation_, exploit=exploit)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        if self._total_env_interactions < self._start_steps and not exploit:
            action = None        
        return action, log_prob, value

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
                next_action_distr = self._pi(next_observation)
                next_action = next_action_distr.sample()
                next_entropy = - next_action_distr.log_prob(next_action)
                if self._pi.independent_actions: 
                    next_entropy = next_entropy.sum(dim=-1, keepdim=True)  
                next_q1_distr = self._q1_target(next_observation, next_action)
                next_q2_distr = self._q2_target(next_observation, next_action)
                next_value1 = (next_q1_distr.mean - self._beta * next_q1_distr.stddev + self._alpha * next_entropy) * done.logical_not().unsqueeze(-1)
                next_value2 = (next_q2_distr.mean - self._beta * next_q2_distr.stddev + self._alpha * next_entropy) * done.logical_not().unsqueeze(-1)
                next_value = 0.5*(next_value1 + next_value2)
                #next_value_var = next_q_distr.variance * done.logical_not().unsqueeze(-1)
                q_target = reward.unsqueeze(-1) + self._gamma * next_value
                #q_target_var = self._gamma**2 * next_value_var
            # critic learning behavioral policy 
            self._q_optim.zero_grad()
            q1_distr = self._q1(observation, action)
            q2_distr = self._q2(observation, action)
            q1_ce = - q1_distr.log_prob(q_target)
            q2_ce = - q2_distr.log_prob(q_target)
            q_loss = (q1_ce + q2_ce).mean()
            self.log("q_loss", q_loss.item())
            self.log("q1_avg", q1_distr.mean.mean().item())
            self.log("q1_std_avg", q1_distr.stddev.mean().item())
            self.log("q2_avg", q2_distr.mean.mean().item())
            self.log("q2_std_avg", q2_distr.stddev.mean().item())
            # backward and optimize
            q_loss.backward()
            self._q_optim.step()
            if self._autopessimism: # EXPERIMENTAL: adjust beta for autopessimism
                error_z1_score_ = ( (q1_distr.mean - q_target)/q1_distr.stddev).mean()
                error_z2_score_ = ( (q2_distr.mean - q_target)/q2_distr.stddev).mean()
                error_z_score_ = 0.5 * (error_z1_score_ + error_z2_score_)
                err_term = error_z_score_.cpu().item()
                #self._beta = self._beta - self._beta_lr * err_term
                self._beta = self._beta * math.exp( + self._beta_lr * err_term )
                self.log("beta", self._beta)        
                self.log("error_z_score_", error_z_score_)
            # soft update
            self._soft_update(self._q1, self._q1_target)
            self._soft_update(self._q2, self._q2_target)
            # on-policy updates
            if (self._total_grad_steps+1) % self._policy_delay == 0:
                self._pi_optim.zero_grad()
                action_distr = self._pi(observation)
                action_onpolicy = action_distr.rsample()
                pi_entropy = - action_distr.log_prob(action_onpolicy)
                if self._pi.independent_actions: 
                    pi_entropy = pi_entropy.sum(dim=-1, keepdim=True)
                q1_onpolicy_distr = self._q1(observation, action_onpolicy)
                q2_onpolicy_distr = self._q2(observation, action_onpolicy)
                q1_onpolicy = q1_onpolicy_distr.mean - self._beta * q1_onpolicy_distr.stddev
                q2_onpolicy = q2_onpolicy_distr.mean - self._beta * q2_onpolicy_distr.stddev
                q_onpolicy = 0.5 * (q1_onpolicy + q2_onpolicy)
                pi_obj = - (q_onpolicy + self._alpha * pi_entropy) 
                pi_loss = pi_obj.mean()
                self.log("pi_loss", pi_loss.item())
                self.log("pi_entropy_avg", pi_entropy.mean().item())
                self.log("q1_avg_onpolicy", q1_onpolicy_distr.mean.mean().item())
                self.log("q1_std_avg_onpolicy", q1_onpolicy_distr.stddev.mean().item())
                self.log("q2_avg_onpolicy", q2_onpolicy_distr.mean.mean().item())
                self.log("q2_std_avg_onpolicy", q2_onpolicy_distr.stddev.mean().item())
                pi_loss.backward()
                self._pi_optim.step()
                # if autotune
                if self._autotune:
                    pi_entropy_ = pi_entropy.mean().cpu().item()
                    self._alpha = self._alpha * math.exp(self._q_lr * self._alpha * ( self._target_entropy - pi_entropy_))
                    self.log("alpha", self._alpha)

    def learn_on_epoch(self):
        pass
    
    @property
    def hparams(self):
        param = {
            "autotune": self._autotune, 
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "beta": self._beta, 
            "autopessimism": self._autopessimism,
            "pi_dropout": self._pi_dropout, 
            "q_dropout": self._q_dropout, 
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "policy_delay": self._policy_delay, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param
    
    def _construct_optimizers(self):
        """Initialize Adam optimizer."""
        self._pi_optim = Adam(self._pi.parameters(), lr=self._pi_lr)
        self._q_optim = Adam(
            [{'params': self._q1.parameters()}, {'params': self._q2.parameters()}], 
            lr=self._q_lr
        )

    def train_mode(self):
        self._q1.train()
        self._q2.train()
        self._pi.train()

    def eval_mode(self):
        self._q1.eval()
        self._q2.eval()
        self._pi.eval()

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
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2")
        parser.add_argument("--q_net", type=str, default="continuous_mlp2")
        parser.add_argument("--autotune", action="store_true", default=False)
        parser.add_argument('--no-autotune', dest="autotune", action="store_false")
        parser.add_argument("--target_entropy", type=float, default=-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--beta", type=float, default=0.0)
        parser.add_argument("--autopessimism", action="store_true", default=False)
        parser.add_argument("--pi_dropout", type=float, default=0.0)
        parser.add_argument("--q_dropout", type=float, default=0.0)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--policy_delay", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--beta_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass