
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os 

import random
import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_policy_map, quantile_qvalue_map


class TQC(Agent):
    
    """Truncated Quantile Critics 
    https://arxiv.org/abs/2005.04269
    """
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        autotune: bool = True, 
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.2, 
        num_ensemble: int = 5, 
        num_quantiles: int = 25, 
        num_drop_quantiles_per_net: float = 2, 
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
        self._gamma = gamma
        self._autotune = autotune
        self._target_entropy = target_entropy
        self._alpha = alpha
        self._num_ensemble = num_ensemble
        self._num_quantiles = num_quantiles
        self._total_quantiles = num_ensemble * num_quantiles
        self._num_drop_quantiles_per_net = num_drop_quantiles_per_net
        self._total_drop_quantiles = num_ensemble * num_drop_quantiles_per_net
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
            q = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, **self.env_info).to(self._device)
            q_target = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, **self.env_info).to(self._device)
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
        """There is no derived field of TQC algorithm. 
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
            q_quantiles_ = q(observation, action)      
            value += (1/self._num_ensemble) * q_quantiles_.mean(dim=-1) + self._alpha * (-log_prob)      
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation: np.ndarray, exploit: bool = False):
        observation_ = th.from_numpy(observation).unsqueeze(0).float().to(self.device)
        action_, log_prob_, value_ = self.step_torch(observation_, exploit=exploit)
        action = action_.squeeze(0).cpu().numpy()
        log_prob = log_prob_.squeeze(0).cpu().numpy()
        value = value_.squeeze(0).cpu().numpy()
        if self._total_env_interactions < self._start_steps:
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
                for m in range(self._num_ensemble):
                    qtarget = self._qs_target[m]
                    next_qvalue_ = qtarget(next_observation, next_action)
                    next_q_values_list.append(next_qvalue_)
                next_q_values = th.stack(next_q_values_list, dim=-2) # batch x num_nets x num_quants 
                next_q_values_sorted, _ = next_q_values.flatten(start_dim=1).sort(dim=-1) # batch x num_nets * num_quants 
                next_q_values_truncated = next_q_values_sorted[:, :self._total_quantiles-self._total_drop_quantiles]
                next_qvalue_target = next_q_values_truncated + self._alpha * next_entropy
                qvalue = reward.unsqueeze(-1) + self._gamma * next_qvalue_target * done.logical_not().unsqueeze(-1) # batch x rem_quants
            # update critics
            self._q_optim.zero_grad()
            qvalue_loss = 0
            qvalue_est_list = []
            for q in self._qs:
                qvalue_est_k = q(observation, action)
                qvalue_est_list.append(qvalue_est_k)
            qvalue_est = th.stack(qvalue_est_list, dim=-2) # batch x num_nets x num_quant 
            qvalue_loss = self.quantile_huber_loss(qvalue_est, qvalue) 
            qvalue_loss.backward()
            self._q_optim.step()
            self.log("q_loss", qvalue_loss.item())
            self.log("q_avg", qvalue_est.mean(dim=-1).mean().item())
            self.log("q_std_avg", qvalue_est.std(dim=-1).mean().item())
            self.log("q_std_epistemic", qvalue_est.mean(dim=-1).std(dim=-1).mean())            
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
                    q_onpolicy = q(observation, action_imaginary)
                    qs_onpolicy_list.append(q_onpolicy)
                qs_onpolicy = th.stack(qs_onpolicy_list, dim=-1).flatten(start_dim=1) # batch x num_quant * num_nets
                q_mean_onpolicy = qs_onpolicy.mean(dim=-1, keepdim=True)
                q_std_onpolicy = qs_onpolicy.std(dim=-1, keepdim=True)
                pi_loss = -(q_mean_onpolicy + self._alpha * entropy).mean()
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
    
    @property
    def hparams(self):
        param = {
            "autotune": self._autotune, 
            "target_entropy": self._target_entropy, 
            "gamma": self._gamma, 
            "alpha": self._alpha, 
            "num_ensemble": self._num_ensemble, 
            "num_quantiles": self._num_quantiles, 
            "num_drop_quantiles_per_net": self._num_drop_quantiles_per_net, 
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
    def quantile_huber_loss(quantiles, samples):
        # samples: # batch x rem_quants
        # quantiles: # batch x num_nets x num_quant
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = th.abs(pairwise_delta)
        huber_loss = th.where(abs_pairwise_delta > 1,
                                abs_pairwise_delta - 0.5,
                                pairwise_delta ** 2 * 0.5)
        n_quantiles = quantiles.shape[2]
        tau = th.arange(n_quantiles, device=pairwise_delta.device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (th.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

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
        parser.add_argument("--num_ensemble", type=int, default=5)
        parser.add_argument("--num_quantiles", type=int, default=25)
        parser.add_argument("--num_drop_quantiles_per_net", type=int, default=2)
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