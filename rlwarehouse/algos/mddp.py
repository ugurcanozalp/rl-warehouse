
from typing import Iterator, List, Tuple, Callable, Any
from argparse import ArgumentParser
import os 

import math
import numpy as np
import torch as th
from torch.distributions import MultivariateNormal
from torch.optim import Adam, AdamW, Optimizer

from ..agent import Agent
from ..nets import probabilistic_reward_map, model_map


class MDDP(Agent):
    
    """Model based DDP
    """
    
    def __init__(self, 
        r_net: str = "continuous_mlp2", 
        model_net: str = "continuous_mlp2", 
        autotune: bool = True, 
        target_entropy: float = -4, 
        gamma: float = 0.99, 
        alpha: float = 0.01, 
        dropout: float = 0.00, 
        tau: float = 0.005, 
        batch_per_step: int = 1, 
        r_lr: float = 3e-4, 
        model_lr: float = 3e-4, 
        plan_horizon: int = 25, 
        control_horizon: int = 1, 
        max_ddp_iters: int = 5, 
        batch_size: int = 256, 
        **memory_kwargs
    ):
        super().__init__(**memory_kwargs)
        # hyperparameters
        self._gamma = gamma
        self._autotune = autotune
        self._target_entropy = target_entropy
        self._alpha = alpha
        self._dropout = dropout
        self._tau = tau
        self._batch_per_step = batch_per_step
        self._batch_size = batch_size 
        self._r_lr = r_lr
        self._model_lr = model_lr
        # planning related 
        self._nx = self._env.observation_space.shape
        self._nu = self._env.action_space.shape
        self._plan_horizon = plan_horizon # MPC horizon
        self._control_horizon = control_horizon # MPC Control horizon
        self._max_ddp_iters = max_ddp_iters # maximum iteration for a DDP step
        # networks
        self._r = probabilistic_reward_map[r_net](**self.env_info, dropout=self._dropout).to(self._device)
        self._model = model_map[model_net](**self.env_info, dropout=self._dropout).to(self._device)
        # jacobian and hessian of networks
        self._r_jacobian = th.vmap(th.func.jacrev(lambda x, u: self._r(x, u).mean, argnums=(0, 1)), in_dims=(0, 0), randomness="same")
        self._r_hessian = th.vmap(th.func.hessian(lambda x, u: self._r(x, u).mean, argnums=(0, 1)), in_dims=(0, 0), randomness="same")
        self._model_jacobian = th.vmap(th.func.jacrev(lambda x, u: self._model(x, u).mean, argnums=(0, 1)), in_dims=(0, 0), randomness="same")
        # self._model_hessian = th.vmap(th.func.hessian(lambda x, u: self._model(x, u).mean, argnums=(0, 1)), in_dims=(0, 0))
        # optimizers
        self._construct_optimizers()
        # sample fields
        self._sample_fields = ("observation", "action", "reward", "next_observation", "done")   

    def save_ckpt(self, path: os.PathLike):
        th.save(self._model.state_dict(), os.path.join(path, "model.pth"))
        th.save(self._r.state_dict(), os.path.join(path, "r.pth"))

    def load_ckpt(self, path: os.PathLike):
        self._model.load_state_dict(th.load(os.path.join(path, "model.pth"), map_location=self.device))
        self._r.load_state_dict(th.load(os.path.join(path, "r.pth"), map_location=self.device))

    @property
    def derived_fields(self):
        """There is no derived field of MDDP algorithm. 
        """
        return ()
    
    def step_torch(self, observation: th.Tensor, exploit: bool = False):
        if (self._horizon_timer % self._control_horizon) == 0: 
            # Time to replan
            # start from previously estimated plan 
            self._u_plan[:-self._control_horizon] = self._u_plan[self._control_horizon:].clone() 
            # Fill remaining points by random sampling
            self._u_plan[-self._control_horizon:] = th.tensor([self._env.action_space.sample() for i in range(self._control_horizon)], dtype=th.float32, device=self._device)
            # self._u_plan[-self._control_horizon:] = torch.zeros(self._control_horizon, *self._nu, dtype=torch.float32, device=self._device)
            self._x_plan[0] = observation
            self._optimize_trajectory(exploit)
            self._horizon_timer = 0
        self._horizon_timer += 1
        action = self._u_plan[self._horizon_timer-1] #.clip(self._u_min, self._u_max) # clip the plan. 
        value = self._v_plan[self._horizon_timer-1]
        log_prob = th.zeros_like(value) # MDDP does not use log_prob, but we need to return it for compatibility
        return action, log_prob, value

    @th.no_grad()
    def step(self, observation: np.ndarray, exploit: bool = False):
        observation_ = th.from_numpy(observation).float().to(self.device)
        action_, log_prob_, value_ = self.step_torch(observation_, exploit=exploit)
        action = action_.cpu().numpy()
        log_prob = log_prob_.cpu().numpy()
        value = value_.cpu().numpy()
        if self._total_env_interactions < self._start_steps:
            action = None    
        return action, log_prob, value

    def episode_end(self):
        pass

    def _optimize_trajectory(self, exploit = False):
        self._initial_forward_pass()
        for j in range(self._max_ddp_iters):
            self._backward_pass()
            updated = self._forward_pass(exploit)
            if not updated:
                break

    def _forward_quadratization(self):
        # Model linear-quadratic
        f_dx_plan, self._f_u_plan = self._model_jacobian(self._x_plan[:-1], self._u_plan)
        self._f_x_plan = f_dx_plan + th.eye(*self._nx, device=f_dx_plan.device).unsqueeze(0).repeat(self._plan_horizon, 1, 1)
        #(self._f_xx_plan, _), (self._f_ux_plan, self._f_uu_plan) = self._model_hessian(self._x_plan[:-1], self._u_plan)
        # Reward linear-quadratic
        self._r_plan = self._r(self._x_plan[:-1], self._u_plan).mean
        self._r_x_plan, self._r_u_plan = self._r_jacobian(self._x_plan[:-1], self._u_plan)
        (self._R_xx_plan, _), (self._R_ux_plan, self._R_uu_plan) = self._r_hessian(self._x_plan[:-1], self._u_plan)

    @th.no_grad()
    def _initial_forward_pass(self):
        for h in range(self._plan_horizon):
            dx_distr = self._model(self._x_plan[h], self._u_plan[h])
            self._x_plan[h+1] = dx_distr.mean + self._x_plan[h]
        self._forward_quadratization()

    def _backward_pass(self, mu=0):
        reg = mu*th.eye(*self._nx, device=self._device)
        for h in range(self._plan_horizon - 1, -1, -1):
            self._q_x_plan[h] = self._r_x_plan[h] + self._f_x_plan[h].mT @ self._v_x_plan[h+1] # 
            self._q_u_plan[h] = self._r_u_plan[h] + self._f_u_plan[h].mT @ self._v_x_plan[h+1] # 
            self._Q_xx_plan[h] = self._R_xx_plan[h] + self._f_x_plan[h].mT @ self._V_xx_plan[h+1] @ self._f_x_plan[h] 
            self._Q_ux_plan[h] = self._R_ux_plan[h] + self._f_u_plan[h].mT @ (self._V_xx_plan[h+1]+reg) @ self._f_x_plan[h] 
            self._Q_uu_plan[h] = self._R_uu_plan[h] + self._f_u_plan[h].mT @ (self._V_xx_plan[h+1]+reg) @ self._f_u_plan[h]
            # Second order dynamics (from iLQR to DDP) / open to switch from iLQR to DDP
            # self._Q_xx_plan[h] += (self._f_xx_plan[h] * self._v_x_plan[h+1].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            # self._Q_ux_plan[h] += (self._f_ux_plan[h] * self._v_x_plan[h+1].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            # self._Q_uu_plan[h] += (self._f_uu_plan[h] * self._v_x_plan[h+1].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            # 
            self._Q_uu_inv_plan[h] = th.linalg.inv(self._Q_uu_plan[h]) # Required on multiple locations
            # Feedback law
            self._k_plan[h] = - self._Q_uu_inv_plan[h] @ self._q_u_plan[h] # - Q_uu_inv @ q_u
            self._K_plan[h] = - self._Q_uu_inv_plan[h] @ self._Q_ux_plan[h] # - Q_uu_inv @ Q_ux
            # Backward Ricatti
            self._v_x_plan[h] = self._q_x_plan[h] + self._Q_ux_plan[h].mT @ self._k_plan[h] 
            self._V_xx_plan[h] = self._Q_xx_plan[h] + self._Q_ux_plan[h].mT @ self._K_plan[h] 

    @th.no_grad()
    def _forward_pass(self, exploit = False): 
        self._x_plan_search[0] = self._x_plan[0]
        v_plan = self._r_plan.sum() 
        for h in range(self._plan_horizon): # Rollout using new policy
            diff_x = self._x_plan_search[h] - self._x_plan[h] 
            diff_u = self._k_plan[h] + (self._K_plan[h] @ diff_x.unsqueeze(-1)).squeeze(-1)
            self._u_plan_search[h] = ( self._u_plan[h] + diff_u ) #.clip(self._u_min, self._u_max)
            dx_distr = self._model(self._x_plan_search[h], self._u_plan_search[h])
            self._x_plan_search[h+1] = self._x_plan_search[h] + dx_distr.mean
        r_plan_search = self._r(self._x_plan_search[:-1], self._u_plan_search).mean
        v_plan_search = r_plan_search.sum() 
        if v_plan_search < v_plan: # success
            self._u_plan = self._u_plan_search.clone()
            self._x_plan = self._x_plan_search.clone()
            self._forward_quadratization()
            return True
        else:
            return False
        
    def reset(self):
        self._horizon_timer = 0
        # DDP variables for each sample in population
        self._u_plan = th.tensor([self._env.action_space.sample() for i in range(self._plan_horizon)], dtype=th.float32, device=self._device) # random actions...
        self._x_plan = th.zeros(self._plan_horizon+1, *self._nx, device=self._device) 
        self._f_x_plan = th.zeros(self._plan_horizon, *self._nx, *self._nx, device=self._device)
        self._f_u_plan = th.zeros(self._plan_horizon, *self._nx, *self._nu, device=self._device)
        #self._f_xx_plan = th.zeros(self._plan_horizon, *self._nx, *self._nx, *self._nx, device=self._device)
        #self._f_ux_plan = th.zeros(self._plan_horizon, *self._nx, *self._nu, *self._nx, device=self._device)
        #self._f_uu_plan = th.zeros(self._plan_horizon, *self._nx, *self._nu, *self._nu, device=self._device)
        self._r_plan = th.zeros(self._plan_horizon, device=self._device)
        self._r_x_plan = th.zeros(self._plan_horizon, *self._nx, device=self._device)
        self._r_u_plan = th.zeros(self._plan_horizon, *self._nu, device=self._device)
        self._R_xx_plan = th.zeros(self._plan_horizon, *self._nx, *self._nx, device=self._device)
        self._R_ux_plan = th.zeros(self._plan_horizon, *self._nu, *self._nx, device=self._device)
        self._R_uu_plan = th.zeros(self._plan_horizon, *self._nu, *self._nu, device=self._device)
        self._v_plan = th.zeros(self._plan_horizon+1, device=self._device)
        self._v_x_plan = th.zeros(self._plan_horizon+1, *self._nx, device=self._device)
        self._V_xx_plan = th.zeros(self._plan_horizon+1, *self._nx, *self._nx, device=self._device)
        self._q_x_plan = th.zeros(self._plan_horizon, *self._nx, device=self._device)
        self._q_u_plan = th.zeros(self._plan_horizon, *self._nu, device=self._device)
        self._Q_xx_plan = th.zeros(self._plan_horizon, *self._nx, *self._nx, device=self._device)
        self._Q_ux_plan = th.zeros(self._plan_horizon, *self._nu, *self._nx, device=self._device)
        self._Q_uu_plan = th.zeros(self._plan_horizon, *self._nu, *self._nu, device=self._device)
        self._Q_uu_inv_plan = th.zeros(self._plan_horizon, *self._nu, *self._nu, device=self._device)
        self._k_plan = th.zeros(self._plan_horizon, *self._nu, device=self._device)
        self._K_plan = th.zeros(self._plan_horizon, *self._nu, *self._nx, device=self._device)
        # candidate plan placeholders. 
        self._u_plan_search = th.zeros(self._plan_horizon, *self._nu, device=self._device) 
        self._x_plan_search = th.zeros(self._plan_horizon+1, *self._nx, device=self._device) 

    def learn_on_step(self):
        for i in range(self._batch_per_step): 
            self._total_grad_steps += 1
            observation, action, reward, next_observation, done \
                = self.memory.sample(self._batch_size)
            # Learn model
            delta_observation = next_observation - observation 
            self._model.zero_grad()
            delta_observation_distr = self._model(observation, action)
            model_loss = - delta_observation_distr.log_prob(delta_observation).mean() # negative log probability is the loss, mean over batches, states etc. 
            model_loss.backward()
            self._model_optim.step()    
            self.log("model_loss", model_loss.item())
            # Learn reward
            self._r.zero_grad()
            r_distr = self._r(observation, action)
            r_loss = - r_distr.log_prob(-reward).mean() # negative log probability is the loss, mean over batches, states etc. 
            r_loss.backward()
            self._r_optim.step()    
            self.log("r_loss", r_loss.item())
            self.log("r_avg", r_distr.mean.mean().item())
            self.log("r_std_avg", r_distr.stddev.mean().item())

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
            "r_lr": self._r_lr, 
            "model_lr": self._model_lr, 
            "plan_horizon": self._plan_horizon, 
            "control_horizon": self._control_horizon, 
            "max_ddp_iters": self._max_ddp_iters, 
        }
        return param
    
    def _construct_optimizers(self):
        """Initialize Adam optimizer."""
        self._r_optim = Adam(self._r.parameters(), lr=self._r_lr)
        self._model_optim = Adam(self._model.parameters(), lr=self._model_lr)

    def train_mode(self):
        self._r.train()
        self._model.train()

    def eval_mode(self):
        self._r.eval()
        self._model.eval()

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
        parser.add_argument("--r_net", type=str, default="continuous_mlp2")
        parser.add_argument("--model_net", type=str, default="continuous_mlp2")
        parser.add_argument("--target_entropy", type=float, default=-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--alpha", type=float, default=0.01)
        parser.add_argument("--dropout", type=float, default=0.00)
        parser.add_argument("--tau", type=float, default=0.005)
        parser.add_argument("--batch_per_step", type=int, default=1)
        parser.add_argument("--r_lr", type=float, default=3e-4)
        parser.add_argument("--model_lr", type=float, default=3e-4)
        parser.add_argument("--plan_horizon", type=int, default=25)
        parser.add_argument("--control_horizon", type=int, default=1)
        parser.add_argument("--max_ddp_iters", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser
        

if __name__=="__main__":
    pass