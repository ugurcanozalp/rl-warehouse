
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os 

import random
import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_policy_map, qvalue_map


class DROQ(Agent):
    
    """Dropout Q Functions 
    https://arxiv.org/pdf/2110.02034
    """
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = False, 
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.2, 
        num_ensemble: int = 2, 
        num_subset: int = 2, 
        dropout: float = 0.01, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        policy_delay: int = 1, 
        pi_lr: float = 3e-4, 
        q_lr: float = 3e-4, 
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        assert num_ensemble >= num_subset, "num_ensemble must be greater than or equal to num_subset"
        # hyperparameters
        self._gamma = gamma
        self._autotune = autotune
        self._target_entropy = target_entropy
        self._alpha = alpha
        self._num_ensemble = num_ensemble
        self._num_subset = num_subset
        self._dropout = dropout
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._policy_delay = policy_delay
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        # networks
        self._pi = probabilistic_policy_map[pi_net](**self.env_info).to(self._device)
        self._qs = []
        self._qs_target = []
        for _ in range(self._num_ensemble):
            q = qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
            q_target = qvalue_map[q_net](dropout=self._dropout, **self.env_info).to(self._device)
            # no grad for target networks
            for param in q_target.parameters():
                param.requires_grad = False
            self._hard_update(q, q_target)
            self._qs.append(q)
            self._qs_target.append(q_target)
        # optimizers
        self._construct_optimizers()
        # sample fields
        self._sample_fields = ("observation", "action", "reward", "next_observation", "done")   
        
    def save_ckpt(self, path: os.PathLike):
        th.save(self._pi.state_dict(), os.path.join(path, "pi.pth"))
        for k in range(self._num_ensemble):
            th.save(self._qs[k].state_dict(), os.path.join(path, "q"+str(k)+".pth"))

    def load_ckpt(self, path: os.PathLike):
        self._pi.load_state_dict(th.load(os.path.join(path, "pi.pth"), map_location=self.device))
        for k in range(self._num_ensemble):
            self._qs[k].load_state_dict(th.load(os.path.join(path, "q"+str(k)+".pth"), map_location=self.device))
            self._hard_update(self._qs[k], self._qs_target[k])

    @property
    def derived_fields(self):
        """There is no derived field of DROQ algorithm. 
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
        value = 0
        for q in self._qs:
            qvalues_ = q(observation, action)      
            value += (1/self._num_ensemble) * qvalues_.squeeze(-1) + self._alpha * (-log_prob)      
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
                next_q_values_list = []
                for m in random.sample(range(self._num_ensemble), 2):
                    qtarget = self._qs_target[m]
                    next_qvalue_ = qtarget(next_observation, next_action)
                    next_q_values_list.append(next_qvalue_)
                next_q_values = th.stack(next_q_values_list, dim=0)
                next_qvalue_target = next_q_values.min(dim=0).values + self._alpha * next_entropy
                qvalue = reward.unsqueeze(-1) + self._gamma * next_qvalue_target * done.logical_not().unsqueeze(-1)
            # update critics
            self._q_optim.zero_grad()
            qvalue_loss = 0
            for q in self._qs:
                qvalue_est_ = q(observation, action)
                qvalue_loss += 0.5*th.nn.functional.mse_loss(qvalue_est_, qvalue) 
            qvalue_loss.backward()
            self._q_optim.step()
            self.log("q_loss", qvalue_loss.item())
            for k in range(self._num_ensemble):
                self._soft_update(self._qs[k], self._qs_target[k])
            # train policy one time
            if (self._total_grad_steps+1) % self._policy_delay == 0:
                self._pi_optim.zero_grad()
                action_distr = self._pi(observation)
                action_imaginary = action_distr.rsample()
                entropy = - action_distr.log_prob(action_imaginary)
                if self._pi.independent_actions: 
                    entropy = entropy.sum(dim=-1, keepdim=True)
                qs_onpolicy_list = []
                for q in self._qs:
                    q_onpolicy_ = q(observation, action_imaginary)
                    qs_onpolicy_list.append(q_onpolicy_)
                qs_onpolicy = th.stack(qs_onpolicy_list, dim=0)
                q_mean_onpolicy = qs_onpolicy.mean(dim=0)
                q_std_onpolicy = qs_onpolicy.std(dim=0)
                pi_loss = -(q_mean_onpolicy + self._alpha * entropy).mean()
                pi_loss.backward()
                self._pi_optim.step()
                self.log("pi_loss", pi_loss.item())
                self.log("pi_entropy_avg", entropy.mean().item())
                self.log("q_onpolicy_avg", q_mean_onpolicy.mean().item())
                self.log("q_std_onpolicy_avg", q_std_onpolicy.mean().item())
                # if autotune
                if self._autotune:
                    entropy_ = entropy.mean().cpu().item()
                    self._alpha = self._alpha * math.exp(self._q_lr * ( self._target_entropy - entropy_))
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
            "num_ensemble": self._num_ensemble, 
            "num_subset": self._num_subset, 
            "dropout": self._dropout, 
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param
    
    def _construct_optimizers(self):
        """Initialize Adam optimizer."""
        self._pi_optim = Adam(self._pi.parameters(), lr=self._pi_lr)
        self._q_optim = Adam([{'params': q.parameters()} for q in self._qs], lr=self._q_lr)

    def train_mode(self):
        for q in self._qs:
            q.train()
        self._pi.train()

    def eval_mode(self):
        for q in self._qs:
            q.eval()
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
        parser.add_argument("--num_ensemble", type=int, default=2)
        parser.add_argument("--num_subset", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.01)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--policy_delay", type=int, default=1)
        parser.add_argument("--target_update_interval", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass