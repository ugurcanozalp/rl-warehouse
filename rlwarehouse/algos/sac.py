
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import policy_map, qvalue_map


class SAC(Agent):
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = False, 
        target_entropy: float = -4, 
        gamma: float = 0.99,
        alpha: float = 0.2, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        pi_lr: float = 3e-4,
        q_lr: float = 1e-3, 
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._autotune = autotune
        self._target_entropy = target_entropy # -0.5 * np.prod(self.memory.env_info["action_shape"])
        self._gamma = gamma
        self._alpha = alpha
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        # networks
        self._pi = policy_map[pi_net](**self.env_info).to(self._device)
        self._q1 = qvalue_map[q_net](**self.env_info).to(self._device)
        self._q2 = qvalue_map[q_net](**self.env_info).to(self._device)
        self._q1_target = qvalue_map[q_net](**self.env_info).eval().to(self._device)
        self._q2_target = qvalue_map[q_net](**self.env_info).eval().to(self._device)
        self._hard_update(self._q1, self._q1_target)
        self._hard_update(self._q2, self._q2_target)
        # no grad for target networks
        for param in self._q1_target.parameters():
            param.requires_grad = False
        for param in self._q2_target.parameters():
            param.requires_grad = False
        # optimizers
        self._construct_optimizers(pi_lr, q_lr)
        
    @property
    def hparams(self):
        param = {
            "autotune": self._autotune, 
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "tau": self._tau, 
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param
    
    @property
    def extra_fields(self):
        """On policy estimated log_prob and value is necessary for value bias estimation
        """
        return ("log_prob", "value")

    @property
    def derived_fields(self):
        """There is no derived field for SAC algorithm.
        """
        return ()
    
    def forward(self, observation: th.Tensor):
        distr = self._pi(observation)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        if self._pi.independent_actions: 
            log_prob = log_prob.sum(dim=-1)
        value = th.mean(self._q1(observation, action), self._q2(observation, action)).squeeze(-1) - self._alpha * log_prob
        return action, (log_prob, value)
    
    @th.no_grad()
    def step(self, observation: np.ndarray):
        if self._total_env_interactions < self._start_steps:
            action = None
            log_prob = 0
            value = 0
        else:
            observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
            action_, (log_prob_, value_) = self.forward(observation_)
            action = action_.squeeze(0).cpu().numpy()
            log_prob = log_prob_.squeeze(0).cpu().numpy()
            value = value_.squeeze(0).cpu().numpy()
        return action, (log_prob, value)

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
            observation, action, reward, next_observation, done, truncated, _, _ = self.memory.sample(self._batch_size)
            with th.no_grad():
                next_action_distr = self._pi(next_observation)
                next_action = next_action_distr.sample()
                next_entropy = - next_action_distr.log_prob(next_action)
                if self._pi.independent_actions: 
                    next_entropy = next_entropy.sum(dim=-1, keepdim=True)
                next_qvalue1 = self._q1_target(next_observation, next_action)
                next_qvalue2 = self._q2_target(next_observation, next_action)
                next_qvalue_target = th.min(next_qvalue1, next_qvalue2) + self._alpha * next_entropy
                qvalue = reward.unsqueeze(-1) + self._gamma * next_qvalue_target * done.logical_not().unsqueeze(-1)
            # update critics
            self._q_optim.zero_grad()
            qvalue1_est = self._q1(observation, action)
            qvalue1_loss = 0.5*th.nn.functional.mse_loss(qvalue1_est, qvalue)
            qvalue2_est = self._q2(observation, action)
            qvalue2_loss = 0.5*th.nn.functional.mse_loss(qvalue2_est, qvalue)
            qvalue_loss = qvalue1_loss + qvalue2_loss
            avg_q_var = 0.5*(qvalue1_est - qvalue2_est).square().mean()
            qvalue_loss.backward()
            self._q_optim.step()
            self._soft_update(self._q1, self._q1_target)
            self._soft_update(self._q2, self._q2_target)
            self.log_grad_step("q1_loss", qvalue1_loss)
            self.log_grad_step("q2_loss", qvalue2_loss)
            self.log_grad_step("avg_q_var", avg_q_var)
            # train policy one time
            self._pi_optim.zero_grad()
            action_distr = self._pi(observation)
            action_imaginary = action_distr.rsample()
            entropy = - action_distr.log_prob(action_imaginary)
            if self._pi.independent_actions: 
                entropy = entropy.sum(dim=-1, keepdim=True)
            q_imaginary = th.min(self._q1(observation, action_imaginary), self._q2(observation, action_imaginary))
            pi_loss = -(q_imaginary + self._alpha * entropy).mean()
            pi_loss.backward()
            self._pi_optim.step()
            self.log_grad_step("pi_loss", pi_loss)
            self.log_grad_step("entropy", entropy.mean())
            # if autotune
            if self._autotune:
                entropy_ = entropy.mean().cpu().item()
                self._alpha = self._alpha * math.exp(self._q_lr * ( self._target_entropy - entropy_))
                self.log_grad_step("alpha", self._alpha)

    def _construct_optimizers(self, pi_lr, q_lr):
        """Initialize Adam optimizer."""
        self._pi_optim = AdamW(self._pi.parameters(), lr=pi_lr)
        # q_optim = Adam(self._q1.parameters(), lr=self._q_lr)
        self._q_optim = AdamW(
            [{'params': self._q1.parameters()}, {'params': self._q2.parameters()}], 
            lr=q_lr
        )
    
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
        cum_return = np.zeros_like(reward)
        for t in reversed(range(self.memory._not_computed)):
            # t_global = self._total_env_interactions + t+1 - self.memory._not_computed
            t_global = self.time_noncomputed_to_global(t)
            if t == self.memory._not_computed-1: # first time for calculation
                _, (_, last_value) = self.step(next_observation[-1])
                cum_return_next = not_done[-1] * last_value  
            else:
                cum_return_next = cum_return[t+1] 
            if truncated[t]:
                cum_return[t] = value[t]
            else:
                cum_return[t] = reward[t] - self._alpha * log_prob[t] + not_done[t] * self._gamma * cum_return_next
            if t==0: # if it is first step of rollout
                self.log("cum_return", cum_return[t], t_global)
                self.log("value_estimate", value[t], t_global)
                self.log("return_error", value[t]-cum_return[t], t_global)
        return ()
    
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
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--target_update_interval", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass
