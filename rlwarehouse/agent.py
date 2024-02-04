"""Generic RL Agent definition
"""

from argparse import ArgumentParser
import os
import math
import numpy as np
from datetime import datetime
from typing import Tuple, Union, Dict, Any

import gymnasium as gym
import torch as th
from tensorboardX import SummaryWriter

from .episode_memory import EpisodeMemory

type_to_torch_dtype_dict = {
    "bool"       : th.bool,
    "uint8"      : th.long,
    "int8"       : th.long,
    "int16"      : th.long,
    "int32"      : th.long,
    "int64"      : th.long,
    "float16"    : th.float,
    "float32"    : th.float,
    "float64"    : th.float,
    "complex64"  : th.cfloat,
    "complex128" : th.cfloat
}


class Agent(object): 

    """Abstract class for all RL agents.
    """
    
    def __init__(self, 
                 env_name: str, 
                 device: str = "cpu", 
                 compute_period: int = -1, 
                 start_steps: int = 10000, 
                 wrapper: gym.ObservationWrapper = None, 
                 env_kwargs: Dict = dict(), 
                 **memory_kwargs
                 ):
        """Summary
        
        Args:
            env_name (str): Name of the environment that agent lives in. 
            device (str, optional): Device in which algorithm works on (cpu, cuda, etc.)
            compute_period (int): How frequenty derived fields are computed. (-1 for off-policy algos)
            start_steps (int): Number of time-steps to act on environment before learning starts. 
            wrapper (gym.ObservationWrapper, optional): Environment wrapper. 
            env_kwargs (Dict, optional): Environment parameters to use during `gym.make` call.  
            **memory_kwargs: Other arguments for episode memory such like environment, size etc. 
        """
        self._device = device
        self._start_steps = start_steps
        self._total_grad_steps = 0 # increment it as you learn from a single batch
        self._env_name = env_name
        self._env = gym.make(env_name, render_mode="human", **env_kwargs)
        if wrapper is not None:
            self._env = wrapper(self._env) # if there is a wrapper.
        self._episode_max_time = self._env._max_episode_steps # environment time limit
        self._episode_time = 0 # episode counter
        self._compute_period = compute_period
        self._total_env_interactions = 0
        self._episode_score = None
        self._logger = None
        self.memory = EpisodeMemory(device=self._device, 
                                    extra_fields=self.extra_fields, 
                                    derived_fields=self.derived_fields, 
                                    **memory_kwargs)

    @property   
    def device(self) -> str:
        """Retrieve device currently being used by minibatch."""
        return self._device

    @property
    def hparams(self): 
        """Hyperparameters of your algorithm. Override it for your algorithm.

        Returns:
            Dict[str, Union[float, int]]: Hyperparameter dictionary 
        """
        return {}

    @property
    def extra_fields(self):
        """Extra fields of the algorithm. Override it for your algorithm.

        Returns:
            Tuple[str]: Names of the extra fields. 
        """
        return ()

    @property
    def derived_fields(self):
        """Derived fields of the algorithm. Override it for your algorithm.

        Returns:
            Tuple[str]: Names of the derived fields. 
        """
        return ()

    @property
    def hparams(self):
        """Hyperparameters of the algorithm. Override it for your algorithm.

        Returns:
            Dict[str, Union[int, float]]: Key-value paris of hyperparameters. 
        """
        return {}
    
    @property
    def env_info(self):
        return {
            "observation_shape": self._env.observation_space.shape,
            "action_shape": self._env.action_space.shape, 
            "observation_space": self._env.observation_space, 
            "action_space": self._env.action_space, 
        }
    
    def _terminate_episode(self):
        self._observation, _ = self._env.reset()
        self.reset()
        self._last_episode_score = self._episode_score
        self._episode_score = 0
        self._episode_time = 0
        self._clear_record()

    def _clear_record(self):
        """The agent saves state-action history, this function clears the records.
        """
        self._obs_history = np.zeros((self._episode_max_time+1, *self._env.observation_space.shape), dtype=self._env.observation_space.dtype)
        self._obs_history[0] = self._observation # clear while adding initial state to record
        self._act_history = np.zeros((self._episode_max_time, *self._env.action_space.shape), dtype=self._env.action_space.dtype)
        self._reward_history = np.zeros(self._episode_max_time, dtype=np.float32)
        self._done_history = np.ones(self._episode_max_time, dtype=np.bool_)

    def _record(self, next_obs: np.ndarray, act: np.ndarray, reward: np.ndarray, done: np.ndarray, time: int):
        """This function is used to record transition tuple onto a time point.
        
        Args:
            next_obs (np.ndarray): Next state
            act (np.ndarray): Action
            reward (np.ndarray): Reward
            done (np.ndarray): Done flag
            time (int): The time index to save transition
        """
        self._obs_history[time+1] = next_obs
        self._act_history[time] = act
        self._reward_history[time] = reward
        self._done_history[time] = done

    @th.no_grad()
    def step(self, obs: np.ndarray, warmup = False, exploit = False):
        """Take action according to your current model. It should return action and corresponding extra fields. 
        
        Args:
            obs (torch.tensor): Observation to take action according to.
            warmup (bool, optional): Flag for warmup steps.
            exploit (bool, optional): Exploit flag.
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError

    def reset(self): 
        """Reset the agent. 
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError

    def train_mode(self):
        """Override this function if you need to do something before training.
        """
        pass

    def test_mode(self):
        """Override this function if you need to do something before training.
        """
        pass

    def save_ckpt(self):
        """Override this function if you need to do save something about your agent.
        """
        pass

    def load_ckpt(self):
        """Override this function if you need to do load saved model.
        """
        pass

    def learn_on_step(self):
        """Define what to do after each time step for learning.
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError

    def compute_function(self, *args):
        """This function is called after episode end or fixed rollout duration
        to populate necessary variables in the experience replay.

        Returns:
            Tuple: Tuple of episodic data, such as return-to-go.
        """
        return ()

    def log(self, log_name: str, log_value: float, step: int = None):
        """Log a parameter during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
            step (int): Step index
        """
        step = self._total_env_interactions if step is None else step
        self._logger.add_scalar(log_name, log_value, step)

    def log_grad_step(self, log_name: str, log_value: float):
        """Log a parameter during training wrt grad step

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
        """
        self._logger.add_scalar(log_name, log_value, self._total_grad_steps)

    def log_hparams(self, dict: Dict[str, Union[bool, str, float, int]]):
        """Log a hyperparameter

        Args:
            dict (Dict): Dictionary to log
        """
        text = " | hyperparam | value | \n | ----------- | ----------- | " + \
            "\n".join([f" | {k} | {v:2.4f} | " for k, v in dict.items()])
        self._logger.add_text("hyperparameters", text, 0)

    def log_text(self, description: str, text: str):
        """Log a textual info

        Args:
            description (str): Text description
            text (str): Text to be logged
        """
        self._logger.add_text(description, text)

    def _adjust_action(self, action):
        """This function takes action output of network bounded in (-1, 1)
        and converts it into environment bounds for action taking. """
        l, h = self._env.action_space.low, self._env.action_space.high
        return (h+l)/2 + (h-l)/2 * action
    
    def one_step_rollout(self, record=False):
        """Rollout in the environment with self and save transitions to memory.

        Args:
            record (bool, optional): Recording history flag. Defaults to False.

        Returns:
            float: Last episode score
        """
        action, extra = self.step(self._observation)
        if action is None: # it means self says randomly act.
            action = np.tanh(np.random.randn(*self._env.action_space.shape)) # random action between (-1, 1)
        next_observation, reward, done, truncated, _ = self._env.step(self._adjust_action(action))
        is_episode_end = done or truncated
        if record:
            self._record(next_observation, action, reward, done, self._episode_time)
        self._total_env_interactions += 1
        self._episode_time += 1
        self._episode_score += reward
        self.memory._insert_transition(
            np.float32(self._observation), 
            np.float32(action), 
            np.float32(reward),
            np.float32(next_observation), 
            done,
            truncated, 
            *extra
        )
        # compute_period=-1 for off-policy algos. compute on episode end
        # compute_period>0 for on-policy algos, compute when the time is ok
        compute_flag = (self._compute_period==-1 and is_episode_end) or (self._total_env_interactions % self._compute_period == 0)
        if compute_flag:
            self.memory._compute(self) 
        if is_episode_end: # end of the episode
            self.log("episode_score", self._episode_score)
            print('\rTime step {}\tScore: {:.2f}'.format(self._total_env_interactions, self._episode_score), end="")
            self._terminate_episode()
        else:
            self._observation = next_observation
        return self._last_episode_score 
    
    def train(self, max_steps: int = int(5e6), 
            test_interval: int = None, 
        ):
        """Main train loop.
        
        Args:
            max_steps (int, optional): Maximum number steps for training
            test_inverval(int, optional): How frequently test episodes are conducted. 
        """
        self._terminate_episode() # initial call for env reset.
        for i in range(max_steps): 
            self.one_step_rollout(self) # step inside episode memory
            self.log_grad_step("total_env_interactions", self._total_env_interactions)
            if i > self._start_steps:
                self.learn_on_step() # learn here.
    
    def experiment(self):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        logdir = os.path.join("runs", self._env_name + "_" + current_time)
        self._logger = SummaryWriter(logdir=logdir)
        # log hyperparameters
        self.log_hparams(self.hparams) # log available variables..
        self.log_text("env_name", self._env_name)
        self.log_text("experiment_time", current_time)
        self.train() # and test sometimes..

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--compute_period", type=int, default=-1)
        #parser.add_argument("--env_kwargs", type=json.loads())
        return parser