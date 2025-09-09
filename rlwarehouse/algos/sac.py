
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os 

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_policy_map, qvalue_map


class SAC(Agent):
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = False, 
        target_entropy: float = -4, 
        gamma: float = 0.99,
        alpha: float = 0.2, 
        dropout: float = 0.0, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        policy_delay: int = 1,
        pi_lr: float = 3e-4,
        q_lr: float = 3e-4, 
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._autotune = autotune
        self._target_entropy = target_entropy # -0.5 * np.prod(self.memory.env_info["action_shape"])
        self._gamma = gamma
        self._alpha = alpha
        self._dropout = dropout        
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._policy_delay = policy_delay
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        # networks
        self._pi = probabilistic_policy_map[pi_net](dropout=self._dropout, **self.env_info).to(self._device)
        self._q1 = qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
        self._q2 = qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
        self._q1_target = qvalue_map[q_net](dropout=self._dropout, **self.env_info).eval().to(self._device)
        self._q2_target = qvalue_map[q_net](dropout=self._dropout, **self.env_info).eval().to(self._device)
        self._hard_update(self._q1, self._q1_target)
        self._hard_update(self._q2, self._q2_target)
        # no grad for target networks
        for param in self._q1_target.parameters():
            param.requires_grad = False
        for param in self._q2_target.parameters():
            param.requires_grad = False
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
    def hparams(self):
        param = {
            "autotune": self._autotune, 
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "dropout": self._dropout,             
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param

    @property
    def derived_fields(self):
        """There is no derived field for SAC algorithm.
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
        q1_ = self._q1(observation, action)
        q2_ = self._q2(observation, action) 
        value = 0.5*( q1_.squeeze(-1) + q2_.squeeze(-1) ) + self._alpha * (-log_prob)
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
                next_qvalue1 = self._q1_target(next_observation, next_action)
                next_qvalue2 = self._q2_target(next_observation, next_action)
                next_value_target = th.min(next_qvalue1, next_qvalue2) + self._alpha * next_entropy
                qvalue = reward.unsqueeze(-1) + self._gamma * next_value_target * done.logical_not().unsqueeze(-1)
            # update critics
            self._q_optim.zero_grad()
            qvalue1_est = self._q1(observation, action)
            qvalue1_loss = 0.5*th.nn.functional.mse_loss(qvalue1_est, qvalue)
            qvalue2_est = self._q2(observation, action)
            qvalue2_loss = 0.5*th.nn.functional.mse_loss(qvalue2_est, qvalue)
            qvalue_loss = qvalue1_loss + qvalue2_loss
            q_std = 0.5*(qvalue1_est - qvalue2_est).abs()
            qvalue_loss.backward()
            self._q_optim.step()
            self._soft_update(self._q1, self._q1_target)
            self._soft_update(self._q2, self._q2_target)
            self.log("q_loss", qvalue_loss.item())
            self.log("q1_loss", qvalue1_loss.item())
            self.log("q2_loss", qvalue2_loss.item())
            self.log("q_avg", 0.5*(qvalue1_est+qvalue2_est).mean().item())
            self.log("q_std_avg", q_std.mean().item())
            # train policy one time
            if (self._total_grad_steps+1) % self._policy_delay == 0:
                self._pi_optim.zero_grad()
                action_distr = self._pi(observation)
                action_imaginary = action_distr.rsample()
                entropy = - action_distr.log_prob(action_imaginary)
                if self._pi.independent_actions: 
                    entropy = entropy.sum(dim=-1, keepdim=True)
                q1_onpolicy = self._q1(observation, action_imaginary)
                q2_onpolicy = self._q2(observation, action_imaginary)
                q_onpolicy = th.min(q1_onpolicy, q2_onpolicy)
                q_std_onpolicy = 0.5*(q1_onpolicy - q2_onpolicy).abs()
                q_mean_onpolicy = 0.5*(q1_onpolicy + q2_onpolicy)
                pi_loss = -(q_onpolicy + self._alpha * entropy).mean()
                pi_loss.backward()
                self._pi_optim.step()
                self.log("pi_loss", pi_loss.item())
                self.log("pi_entropy_avg", entropy.mean().item())
                self.log("q_avg_onpolicy", q_mean_onpolicy.mean().item())
                self.log("q_std_avg_onpolicy", q_std_onpolicy.mean().item())
                # if autotune
                if self._autotune:
                    entropy_ = entropy.mean().cpu().item()
                    self._alpha = self._alpha * math.exp(self._q_lr * ( self._target_entropy - entropy_))
                    self.log("alpha", self._alpha)
                    
    def learn_on_epoch(self):
        pass

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
        parser.add_argument("--autotune", action="store_true")
        parser.add_argument('--no-autotune', dest="autotune", action="store_false")
        parser.add_argument("--target_entropy", type=float, default=-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--dropout", type=float, default=0.0)        
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--policy_delay", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass
