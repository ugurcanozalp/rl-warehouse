
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os 

import math
import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_policy_map, quantile_qvalue_map


class DSAC(Agent):
    
    """DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning
    https://arxiv.org/abs/2004.14547
    """
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        q_net: str = "continuous_mlp2",
        gamma: float = 0.99,
        autotune: bool = False, 
        target_entropy: float = -4,  
        alpha: float = 0.2, 
        num_quantiles: int = 25, 
        msd: float = 0.0, 
        cvar_qtl: float = 0.0, 
        pi_dropout: float = 0.0, 
        q_dropout: float = 0.0,         
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
        self._target_entropy = target_entropy 
        self._gamma = gamma
        self._alpha = alpha
        self._num_quantiles = num_quantiles
        self._msd = msd # mean semi-deviation weight
        self._cvar_qtl = cvar_qtl # conditional value at risk quantile
        self._pi_dropout = pi_dropout
        self._q_dropout = q_dropout        
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._policy_delay = policy_delay
        self._batch_size = batch_size 
        self._q_lr = q_lr
        self._pi_lr = pi_lr
        # 
        self._prev_episode_score = None
        # networks
        self._pi = probabilistic_policy_map[pi_net](dropout=self._pi_dropout, **self.env_info).to(self._device)
        self._q1 = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q2 = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, dropout=self._q_dropout, **self.env_info).to(self._device)
        self._q1_target = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, dropout=self._q_dropout, **self.env_info).eval().to(self._device)
        self._q2_target = quantile_qvalue_map[q_net](num_quantiles=self._num_quantiles, dropout=self._q_dropout, **self.env_info).eval().to(self._device)
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
            "tau": self._tau, 
            "msd": self._msd, 
            "cvar_qtl": self._cvar_qtl,
            "num_quantiles": self._num_quantiles,
            "batch_per_step": self._batch_per_step, 
            "batch_size": self._batch_size, 
            "q_lr": self._q_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param

    @property
    def derived_fields(self):
        """There is no derived field for TOPSAC algorithm.
        """
        return ()

    def step_torch(self, observation: th.Tensor, exploit: bool = False):
        distr = self._pi(observation)
        if exploit:
            action = distr.rsample((10, )).mean(dim=0) # averaged action
        else:
            action = distr.rsample()
        return action, ()

    def step_torch(self, observation: th.Tensor, exploit: bool = False):
        distr = self._pi(observation)
        if exploit:
            action = distr.rsample((10, )).mean(dim=0) # averaged action
        else:
            action = distr.rsample()
        q1_quantiles = self._q1(observation, action)
        q2_quantiles = self._q2(observation, action) 
        log_prob = distr.log_prob(action)
        if self._pi.independent_actions: 
            log_prob = log_prob.sum(dim=-1)           
        value = 0.5*( q1_quantiles.mean(dim=-1) + q2_quantiles.mean(dim=-1) ) + self._alpha * (-log_prob)
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
                next_qvalue1_qtls = self._q1_target(next_observation, next_action) # batch x num_quants 
                next_qvalue2_qtls = self._q2_target(next_observation, next_action) # batch x num_quants 
                next_value_qtls_target = th.min(next_qvalue1_qtls, next_qvalue2_qtls) + self._alpha * next_entropy
                target_qvalue_qtls = reward.unsqueeze(-1) + self._gamma * next_value_qtls_target * done.logical_not().unsqueeze(-1)
            # update critics
            self._q_optim.zero_grad()
            qvalue1_est_qtls = self._q1(observation, action)
            qvalue1_loss = self.quantile_huber_loss(qvalue1_est_qtls, target_qvalue_qtls) 
            qvalue2_est_qtls = self._q2(observation, action)
            qvalue2_loss = self.quantile_huber_loss(qvalue2_est_qtls, target_qvalue_qtls) 
            qvalue_loss = qvalue1_loss + qvalue2_loss
            qvalue_loss.backward()
            self._q_optim.step()
            self._soft_update(self._q1, self._q1_target)
            self._soft_update(self._q2, self._q2_target)
            wd = self.compute_wd_quantile(qvalue1_est_qtls, qvalue2_est_qtls).mean() # average wassertein-distance between q1 and q2
            self.log("q_loss", qvalue_loss.item())
            self.log("q1_loss", qvalue1_loss.item())
            self.log("q2_loss", qvalue2_loss.item())
            self.log("wd", wd.item())
            # train policy one time
            if (self._total_grad_steps+1) % self._policy_delay == 0:
                self._pi_optim.zero_grad()
                action_distr = self._pi(observation)
                action_imaginary = action_distr.rsample()
                entropy = - action_distr.log_prob(action_imaginary)
                if self._pi.independent_actions: 
                    entropy = entropy.sum(dim=-1, keepdim=True)
                q1_onpolicy_qtls = self._q1(observation, action_imaginary)
                q2_onpolicy_qtls = self._q2(observation, action_imaginary)
                q_onpolicy_qtls_obj = th.min(self.get_policy_obj(q1_onpolicy_qtls), self.get_policy_obj(q2_onpolicy_qtls))
                q_onpolicy = q_onpolicy_qtls_obj.mean(dim=-1)
                pi_loss = -(q_onpolicy + self._alpha * entropy).mean()
                pi_loss.backward()
                self._pi_optim.step()
                self.log("pi_loss", pi_loss.item())
                self.log("pi_entropy_avg", entropy.mean().item())
                self.log("q_onpolicy", q_onpolicy.mean().item())
                # if autotune
                if self._autotune:
                    entropy_ = entropy.mean().cpu().item()
                    self._alpha = self._alpha * math.exp(self._q_lr * ( self._target_entropy - entropy_))
                    self.log("alpha", self._alpha)

    def get_policy_obj(self, q_quantiles):
        reduced_q_quantiles = q_quantiles[:, :self._num_quantiles - self._cvar_qtl]
        q_obj = reduced_q_quantiles.mean(dim=-1, keepdim=True) - self._msd * reduced_q_quantiles.std(dim=-1, keepdim=True)
        return q_obj

    def learn_on_epoch(self):
        pass
    
    def _construct_optimizers(self):
        """Initialize Adam optimizer."""
        self._pi_optim = Adam(self._pi.parameters(), lr=self._pi_lr)
        # q_optim = Adam(self._q1.parameters(), lr=self._q_lr)
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
    def quantile_huber_loss(quantiles, samples):
        # samples: # batch x num_quant_sample
        # quantiles: # batch x num_quant
        pairwise_delta = samples[:, None, :] - quantiles[:, :, None]  # batch x num_quant x num_quant_sample 
        abs_pairwise_delta = th.abs(pairwise_delta)
        huber_loss = th.where(abs_pairwise_delta > 1,
                                abs_pairwise_delta - 0.5,
                                pairwise_delta ** 2 * 0.5)
        n_quantiles = quantiles.shape[-1]
        tau = th.arange(n_quantiles, device=pairwise_delta.device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (th.abs(tau[None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

    @staticmethod
    def compute_wd_quantile(q1_qtls, q2_qtls, wd_gamma = 1.0):
        wd = th.pow(th.sum(th.pow(th.abs(q1_qtls - q2_qtls), wd_gamma)), 1 / wd_gamma)
        wd = wd.mean(dim=-1) # do not mean by batch for now. 
        return wd

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
        parser.add_argument("--num_quantiles", type=int, default=25)
        parser.add_argument("--msd", type=float, default=0.0)
        parser.add_argument("--cvar_qtl", type=float, default=0)
        parser.add_argument("--pi_dropout", type=float, default=0.0)
        parser.add_argument("--q_dropout", type=float, default=0.0)        
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--policy_delay", type=int, default=1)
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--q_lr", type=float, default=3e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass
