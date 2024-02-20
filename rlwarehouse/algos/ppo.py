
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser

import numpy as np
import torch as th
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import policy_map, value_map


# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/reinforce-learning-DQN.html
class PPO(Agent):
    
    def __init__(self, 
        pi_net: str = "continuous_mlp2", 
        v_net: str = "continuous_mlp2",
        gamma: float = 0.99,
        lambd: float = 0.95, 
        alpha: float = 5e-4, 
        epochs_per_rollout: int = 10, 
        steps_per_rollout: int = 2048, 
        clip_ratio: int = 0.2, 
        pi_lr: float = 3e-4,
        v_lr: float = 3e-4, 
        vf_coef: float = 0.5, 
        max_grad_norm: float = 0.5, 
        batch_size: int = 256, 
        **memory_kwargs
    ):
        # override memory_kwargs for ppo
        memory_kwargs["buffer_capacity"] = steps_per_rollout
        super().__init__(**memory_kwargs)
        self._start_steps = 0 # override start_steps
        self._compute_period = steps_per_rollout
        # hyperparameters
        self._gamma = gamma
        self._lambda = lambd
        self._alpha = alpha
        self._batch_size = batch_size
        self._clip_ratio = clip_ratio 
        self._vf_coef = vf_coef
        self._max_grad_norm = max_grad_norm
        self._v_lr = v_lr
        self._pi_lr = pi_lr
        self._steps_per_rollout = steps_per_rollout
        self._batch_per_rollout = self._steps_per_rollout // batch_size 
        self._epochs_per_rollout = epochs_per_rollout
        # networks
        self._pi = policy_map[pi_net](**self.env_info).to(self._device)
        self._v = value_map[v_net](**self.env_info).to(self._device)
        # optimizers
        self._construct_optimizers(pi_lr, v_lr)

    @property
    def hparams(self):
        param = {
            "gamma": self._gamma, 
            "lambda": self._lambda, 
            "alpha": self._alpha, 
            "batch_size": self._batch_size, 
            "clip_ratio": self._clip_ratio, 
            "vf_coef": self._vf_coef, 
            "max_grad_norm": self._max_grad_norm, 
            "steps_per_rollout": self._steps_per_rollout, 
            "batch_per_rollout": self._batch_per_rollout, 
            "epochs_per_rollout": self._epochs_per_rollout, 
            "v_lr": self._v_lr, 
            "pi_lr": self._pi_lr, 
        }
        return param
    
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
        if self.memory._not_computed>0 or self.memory._size==0:
            return # if not ready for computation, return..
        for _ in range(self._epochs_per_rollout):
            for batch_idx in range(self._batch_per_rollout): # get batch..
                self._total_grad_steps += 1
                indices = list(range(batch_idx*self._batch_size, (batch_idx+1)*(self._batch_size)))
                observation, action, reward, next_observation, \
                    done, truncated, log_prob, value, \
                    cum_return, gae  = self.memory._sample_by_indices(indices)
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
                self.log_grad_step("pi_loss", pi_loss)
                self.log_grad_step("v_loss", v_loss)
                self.log_grad_step("approx_kl", approx_kl)
                self.log_grad_step("clipfrac", clipfrac)
                self.log_grad_step("cross_log_prob", cross_log_prob.mean())

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
        _, (_, last_value) = self.step(next_observation[-1])
        for t in reversed(range(self.memory._not_computed)):
            # t_global = self._total_env_interactions + t + 1 - self.memory._not_computed
            t_global = self.time_noncomputed_to_global(t)
            if t == self.memory._not_computed-1: # first time for calculation
                cum_return_next = reward[-1] + not_done[-1] * self._gamma * last_value  
                gae_next = reward[-1] + self._gamma * last_value - value[-1] # delta at last time...
                value_next = last_value
            else:
                cum_return_next = cum_return[t+1] 
                gae_next = gae[t+1]
                value_next = value[t+1]
            if truncated[t]:
                cum_return[t] = value[t]
            else:
                cum_return[t] = reward[t] + not_done[t] * self._gamma * cum_return_next
            delta = reward[t] + not_done[t] * self._gamma * value_next - value[t] # one step td error
            gae[t] = delta + not_done[t] * self._gamma * self._lambda * gae_next
            if t!=0 and (done[t-1] or truncated[t-1]): # if it is first step of rollout
                self.log("cum_return", cum_return[t], t_global)
                self.log("return_error", value[t]-cum_return[t], t_global)
        return cum_return, gae

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Agent.add_model_specific_args(parser)
        parser.add_argument("--pi_net", type=str, default="continuous_mlp2") 
        parser.add_argument("--v_net", type=str, default="continuous_mlp2")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--lambd", type=float, default=0.95) 
        parser.add_argument("--alpha", type=float, default=5e-4)
        parser.add_argument("--epochs_per_rollout", type=int, default=10) 
        parser.add_argument("--steps_per_rollout", type=int, default=2048) 
        parser.add_argument("--clip_ratio", type=int, default=0.4) 
        parser.add_argument("--pi_lr", type=float, default=3e-4)
        parser.add_argument("--v_lr", type=float, default=3e-4)
        parser.add_argument("--vf_coef", type=float, default=0.5)
        parser.add_argument("--max_grad_norm", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser

if __name__=="__main__":
    pass