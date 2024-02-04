
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import policy_map, probabilistic_qvalue_map

# beta parameter are usually tried from one of them -> 0.1, 0.4, 0.7, 1.0
# beta->            Ant = 1.0, Hopper = 0.7, Walker2d = 0.4, Humanoid = 1.0, HalfCheetah = 0.1


class PRAC(Agent):
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = True, 
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.2, 
        beta: float = 0.4, 
        dropout: float = 0.01, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        pi_lr: float = 3e-4, 
        q_lr: float = 1e-3, 
        max_grad_norm: float = 1.0, 
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
        self._dropout = dropout
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        self._max_grad_norm = max_grad_norm
        # networks
        self._pi = policy_map[pi_net](**self.env_info, dropout=self._dropout).to(self._device)
        self._q = probabilistic_qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
        self._q_target = probabilistic_qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
        #
        # no grad for target networks
        for param in self._q_target.parameters():
            param.requires_grad = False
        self._hard_update(self._q, self._q_target)
        # optimizers
        self._construct_optimizers(pi_lr, q_lr)
        
    @property
    def extra_fields(self):
        """No extra field is required for EAC algorithm.
        """
        return ()

    @property
    def derived_fields(self):
        """There is no derived field for EAC algorithm.
        """
        return ()
    
    def forward(self, observation: th.Tensor):
        distr = self._pi(observation)
        action = distr.rsample()
        return action

    @th.no_grad()
    def step(self, observation: np.ndarray):
        if self._total_env_interactions < self._start_steps:
            action = None
        else:
            observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
            action_ = self.forward(observation_)
            action = action_.squeeze(0).cpu().numpy()
        return action, ()

    def reset(self):
        pass

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)

    def _hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def learn_on_step(self):
        # train critics
        for _ in range(self._batch_per_step): 
            self._total_grad_steps += 1
            observation, action, reward, next_observation, done, truncated = self.memory.sample(self._batch_size)
            with th.no_grad():
                next_action_distr = self._pi(next_observation)
                next_action = next_action_distr.sample()
                next_entropy = - next_action_distr.log_prob(next_action)
                if self._pi.independent_actions: 
                    next_entropy = next_entropy.sum(dim=-1, keepdim=True)  
                # take min of two, like sac
                next_q_distr = self._q_target(next_observation, next_action)
                next_value_target = (next_q_distr.mean - self._beta * next_q_distr.stddev + self._alpha * next_entropy) * done.logical_not().unsqueeze(-1) 
                q_target_sample = reward.unsqueeze(-1) + self._gamma * next_value_target 
            # critic learning behavioral policy 
            self._q_optim.zero_grad()
            q_distr = self._q(observation, action)
            # critic update calculations
            q_crossentropy = - q_distr.log_prob(q_target_sample) # batch, 1
            q_obj = q_crossentropy
            q_loss = q_obj.mean()
            self.log_grad_step("q_loss", q_loss)
            self.log_grad_step("q_entropy_avg", q_distr.entropy().mean())
            self.log_grad_step("q_std_avg", q_distr.stddev.mean())
            q_loss.backward()
            #th.nn.utils.clip_grad_norm_(self._q.parameters(), self._max_grad_norm)
            self._q_optim.step()
            self._soft_update(self._q, self._q_target)
            # on-policy updates
            self._pi_optim.zero_grad()
            action_distr = self._pi(observation)
            action_onpolicy = action_distr.rsample()
            pi_entropy = - action_distr.log_prob(action_onpolicy)
            if self._pi.independent_actions: 
                pi_entropy = pi_entropy.sum(dim=-1, keepdim=True)
            q_onpolicy_distr = self._q(observation, action_onpolicy)
            q_onpolicy = q_onpolicy_distr.mean - self._beta * q_onpolicy_distr.stddev
            pi_obj = - (q_onpolicy + self._alpha * pi_entropy)
            pi_loss = pi_obj.mean()
            self.log_grad_step("pi_loss", pi_loss)
            self.log_grad_step("q_onpolicy_avg", q_onpolicy.mean())
            self.log("q_onpolicy_entropy_avg", q_onpolicy_distr.entropy().mean())
            self.log("q_onpolicy_std_avg", q_onpolicy_distr.stddev.mean())
            self.log_grad_step("pi_entropy_avg", pi_entropy.mean())
            pi_loss.backward()
            #th.nn.utils.clip_grad_norm_(self._pi.parameters(), self._max_grad_norm)
            self._pi_optim.step()
            # if autotune
            if self._autotune:
                pi_entropy_ = pi_entropy.mean().cpu().item()
                self._alpha = self._alpha * math.exp(self._q_lr * self._alpha * ( self._target_entropy - pi_entropy_))
                self.log_grad_step("alpha", self._alpha)

    @property
    def hparams(self):
        param = {
            "autotune": self._autotune, 
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "beta": self._beta, 
            "dropout": self._dropout, 
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param
    
    def _construct_optimizers(self, pi_lr, q_lr):
        """Initialize Adam optimizer."""
        self._pi_optim = AdamW(self._pi.parameters(), lr=pi_lr)
        self._q_optim = AdamW(self._q.parameters(), lr=self._q_lr)

    def compute_function(self,
        observation, 
        action, 
        reward, 
        next_observation, 
        done, 
        truncated, 
    ):
        return ()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Agent.add_model_specific_args(parser)
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2")
        parser.add_argument("--q_net", type=str, default="continuous_mlp2")
        parser.add_argument("--autotune", action="store_true" ,default=True)
        parser.add_argument('--no-autotune', dest="autotune", action="store_false")
        parser.add_argument("--target_entropy", type=float, default=-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--beta", type=float, default=0.4)
        parser.add_argument("--dropout", type=float, default=0.01)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--target_update_interval", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass