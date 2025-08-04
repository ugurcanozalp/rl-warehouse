"""Generic Online RL Agent definition
"""

from argparse import ArgumentParser
import os
import math
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Union, Dict, Any
import pickle 
import json
import jsonlines
import time

import gymnasium as gym
import torch as th
from torch.utils.tensorboard import SummaryWriter
from tbparse import SummaryReader
from .logger import Logger
import pandas as pd

from .memory import ReplayMemory


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
                 seed: int = -1, 
                 num_threads: int = -1, 
                 device: str = "cpu", 
                 max_train_steps = 1000000, 
                 compute_period: int = -1, 
                 start_steps: int = 10000, 
                 eval_interval: int = 10000, 
                 wrapper: gym.ObservationWrapper = None, 
                 render_mode: str = "human", 
                 recording: bool = True, 
                 logging: bool = False, 
                 tblogging: bool = False, 
                 algo_tag: str = "", 
                 env_tag: str = "", 
                 save_memory: bool = False, 
                 env_kwargs: Dict = dict(), 
                 **memory_kwargs
                 ):
        """Summary
        
        Args:
            env_name (str): Name of the environment that agent lives in. 
            seed (int, optional): Overall seed for learning
            num_threads (int, optional): Number of threads used for intraop parallelism on CPU
            device (str, optional): Device in which algorithm works on (cpu, cuda, etc.)
            max_train_steps (int, optional): Maximum number steps for training
            compute_period (int): How frequenty derived fields are computed. (-1 for off-policy algos)
            start_steps (int): Number of time-steps to act on environment before learning starts. 
            eval_interval (int): Interval of training steps required to perform evaluation. 
            wrapper (gym.ObservationWrapper, optional): Environment wrapper. 
            render_mode (str): Render flag during learning. 
            env_kwargs (Dict, optional): Environment parameters to use during `gym.make` call.  
            **memory_kwargs: Other arguments for episode memory such like environment, size etc. 
        """
        self._seed = seed if seed != -1 else random.randint(1, 42)
        self._num_threads = num_threads
        self._device = device
        self._max_train_steps = max_train_steps
        self._start_steps = start_steps
        self._eval_interval = eval_interval
        self._total_grad_steps = 0 # increment it as you learn from a single batch
        self._recording = recording
        self._logging = logging
        self._tblogging = tblogging
        self._algo_tag = algo_tag
        self._env_tag = env_tag
        self._save_memory = save_memory
        # env
        self._env_name = env_name
        render_mode = None if render_mode=="none" else render_mode
        self._env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        self._env.reset(seed=self._seed)
        self._env.action_space.seed(self._seed)
        self._env.observation_space.seed(self._seed)
        # eval env
        self._env_eval = gym.make(env_name, render_mode=None, **env_kwargs) # render_mode="none", 
        self._env_eval.reset(seed=42+self._seed)
        self._env_eval.action_space.seed(42+self._seed)
        self._env_eval.observation_space.seed(42+self._seed)
        # 
        if wrapper is not None:
            self._env = wrapper(self._env) # if there is a wrapper.
        try:
            self._episode_max_time = self._env._max_episode_steps # environment time limit
        except: 
            self._episode_max_time = 1000
        self._episode_time = 0 # episode counter
        self._compute_period = compute_period
        self._total_env_interactions = 0
        self._episode_score = None
        self._episode_discounted_score = None
        self._episode_value_estimate = None
        self._episode_value_error = None
        self._logger = None
        if self._seed != -1:
            random.seed(self._seed)
            np.random.seed(self._seed)
            th.manual_seed(self._seed)
        if self._num_threads != -1:
            th.set_num_threads(self._num_threads)
        self.memory = ReplayMemory(device=self._device, 
                                    derived_fields=self.derived_fields, 
                                    **memory_kwargs)

    @property
    def algo_name(self):
        "Retrieve algorithm name"
        return self.__class__.__name__

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
    
    @property
    def gamma(self):
        """Discount factor of the agent. Default is 1 if not defined. 
        
        Returns:
            float: Discount factor of the agent.
        """
        if hasattr(self, "_gamma"):
            return self._gamma
        else:
            return 1.0
        
    @property
    def alpha(self):
        """Alpha value for the agent. It is used in some algorithms such as SAC.
        
        Returns:
            float: Alpha value of the agent.
        """
        if hasattr(self, "_alpha"):
            return self._alpha
        else:
            return 0.0
        
    def time_noncomputed_to_global(self, t):
        return self._total_env_interactions + t + 1 - self.memory._not_computed

    def _terminate_episode(self):
        self.episode_end()
        self.reset()
        self._observation, _ = self._env.reset()
        self._episode_score = 0
        self._episode_discounted_score = 0
        self._episode_value_estimate = 0
        self._episode_value_error = 0
        self._episode_time = 0

    def _clear_record(self, obs0):
        """The agent saves state-action history, this function clears the records.
        """
        self._obs_history = np.nan * np.zeros((self._episode_max_time+1, *self._env.observation_space.shape), dtype=self._env.observation_space.dtype)
        self._obs_history[0] = obs0 # clear while adding initial state to record
        self._act_history = np.nan * np.zeros((self._episode_max_time, *self._env.action_space.shape), dtype=self._env.action_space.dtype)
        self._reward_history = np.nan * np.zeros(self._episode_max_time, dtype=np.float32)
        self._done_history = np.nan * np.ones(self._episode_max_time, dtype=np.bool_)
        self._truncated_history = np.nan * np.ones(self._episode_max_time, dtype=np.bool_)

    def _record(self, next_obs: np.ndarray, act: np.ndarray, reward: np.ndarray, done: np.ndarray, truncated: np.ndarray, time: int):
        """This function is used to record transition tuple onto a time point.
        
        Args:
            next_obs (np.ndarray): Next state
            act (np.ndarray): Action
            reward (np.ndarray): Reward
            done (np.ndarray): Done flag
            truncated (np.ndarray): Truncation flag
            time (int): The time index to save transition
        """
        self._obs_history[time+1] = next_obs
        self._act_history[time] = act
        self._reward_history[time] = reward
        self._done_history[time] = done
        self._truncated_history[time] = truncated

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

    '''
    @th.no_grad()
    def value(self, obs: np.ndarray):
        """Estimate value of the policy given the observation. 
        
        Args:
            obs (torch.tensor): Observation to take action according to.
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError
    '''
    
    def reset(self): 
        """Reset the agent. 
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError

    def episode_end(self):
        """Call this function at the end of episode.  
        
        Raises:
            NotImplementedError: This function must be overriden. 
        """
        raise NotImplementedError
            
    def train_mode(self):
        """Override this function if you need to do something before training.
        """
        pass

    def eval_mode(self):
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

    def learn_on_epoch(self):
        """Define what to do after memory compoutations are done. Used by 
        on-policy algorithms in general. 

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

    def log(self, log_name: str, log_value: float, step: int = None, bypass: bool = False):
        """Log a parameter during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
            step (int): Step index
            bypass (bool): Bypass no logging flag (Always log)
        """
        step = self._total_env_interactions if step is None else step
        if self._logging or bypass:
            self._logger.log(log_name, log_value, step)
        if self._tblogging:
            self._tbloggingger.add_scalar(log_name, log_value, step)
            
    def log_histogram(self, log_name: str, log_value: np.ndarray, step: int = None):
        """Log a vector during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (np.ndarray): Value of the logged parameter
            step (int): Step index
        """
        if self._logging:
            step = self._total_env_interactions if step is None else step
            self._tbloggingger.add_histogram(log_name, log_value, step)

    def log_hparams(self, hparams: Dict[str, Union[bool, str, float, int]]):
        """Log a hyperparameters of the run

        Args:
            hparams (Dict): Hyperparameter dictionary
        """
        self._logger.log_hparams(hparams)

    def log_text(self, description: str, text: str): 
        """Log a textual info

        Args:
            description (str): Text description
            text (str): Text to be logged
        """
        self._logger.log_text(description, text)
        if self._tblogging:
            self._tbloggingger.add_text(description, text)

    def _adjust_action(self, action):
        """This function takes action output of network bounded in (-1, 1)
        and converts it into environment bounds for action taking. """
        l, h = self._env.action_space.low, self._env.action_space.high
        return (h+l)/2 + (h-l)/2 * action.astype(self._env.action_space.dtype)
    
    def train_step_rollout(self):
        """Rollout in the environment with self and save transitions to memory.
        """
        action, log_prob, value = self.step(self._observation, exploit=False)
        if action is None: # it means self says randomly act.
            action = np.tanh(np.random.randn(*self._env.action_space.shape).astype(self._env.action_space.dtype)) # random action between (-1, 1)
        if self._episode_time == 0:
            self._episode_value_estimate = value
        next_observation, reward, done, truncated, _ = self._env.step(self._adjust_action(action))
        is_episode_end = done or truncated
        self._episode_score += reward
        self._episode_discounted_score += pow(self.gamma, self._episode_time) * ( reward + self.alpha * (-log_prob))
        self._total_env_interactions += 1
        self._episode_time += 1
        self.memory._insert_transition(
            np.float32(self._observation), 
            np.float32(action), 
            np.float32(reward),
            np.float32(next_observation), 
            done,
            truncated, 
            log_prob, 
            value, 
        )
        # compute_period = -1 for off-policy algos. compute on episode end
        # compute_period > 0 for on-policy algos, compute when the time is ok
        # compute_flag = (self._compute_period==-1 and is_episode_end) or (self._total_env_interactions % self._compute_period == 0)
        compute_flag = is_episode_end if self._compute_period==-1 else (self._total_env_interactions % self._compute_period == 0)
        # self.log("obs", self._observation) # TODO: Open this? but when?
        if compute_flag:
            self.memory._compute(self) 
        if is_episode_end: # end of the episode
            self._episode_value_error = self._episode_value_estimate - self._episode_discounted_score
            self.log("episode_score", self._episode_score)
            self.log("episode_discounted_score", float(self._episode_discounted_score))
            self.log("episode_value_estimate", float(self._episode_value_estimate))
            self.log("episode_value_error", float(self._episode_value_error))
            print('\rTime step {}\tScore: {:.2f}'.format(self._total_env_interactions, self._episode_score), end="")
            self._terminate_episode()
        else:
            self._observation = next_observation
    
    def train(self):
        """Main train loop.
        """
        self.train_mode() # start with training mode
        self._terminate_episode() # initial call for env reset.
        while self._total_env_interactions < self._max_train_steps:
            self.train_step_rollout() # step inside episode memory
            self.log("total_env_interactions", self._total_env_interactions) 
            if self.memory._not_computed==0 and self.memory._size!=0:
                self.learn_on_epoch()
            if self._total_env_interactions > self._start_steps:
                self.learn_on_step() # learn here.
            if self._total_env_interactions % self._eval_interval == 0 and self._total_env_interactions != 0:
                self.eval()
                self.train_mode() # go back to train mode
    
    def experiment_end(self): 
        """The function to be called at the end of experiment, for logging etc. 
        """
        raise NotImplementedError
    
    def eval(self): 
        """Evaluate the agent for fixed number of episodes
        """
        self.eval_mode()
        score = 0
        discounted_score = 0
        time = 0
        self.reset()
        observation, _ = self._env_eval.reset()
        values = []
        rewards = []        
        is_episode_end = False
        while not is_episode_end:
            action, log_prob, value = self.step(observation, exploit=True)
            values.append(value)
            if action is None: # it means self says randomly act.
                action = np.tanh(np.random.randn(*self._env.action_space.shape)) # random action between (-1, 1)
            next_observation, reward, done, truncated, _ = self._env_eval.step(self._adjust_action(action))
            is_episode_end = done or truncated
            rewards.append(reward)
            score += reward
            discounted_score += pow(self.gamma, time) * ( reward + self.alpha * (-log_prob) ) 
            time += 1
            # update observation
            observation = next_observation
        # end of the episode
        rewards_np = np.stack(rewards, axis=0)
        values_np = np.stack(values, axis=0)
        returns_np = np.zeros_like(rewards_np)
        return_next = 0
        for t in reversed(range(values_np.shape[0])):
            returns_np[t] = rewards[t] + self.gamma * return_next
            return_next = returns_np[t]
        value_errors = values_np[:-1] - returns_np[:-1]
        # logging 
        self.log("eval_score", score, bypass=True)
        self.log("eval_value_error", value_errors.mean(), bypass=True)

    #def eval_rollout(self): 
    #    self.eval_mode()
    #    score = 0
    #    discounted_score = 0
    #    time = 0
    #    observation, _ = self._env_eval.reset()
    #    if self._recording:
    #        self._clear_record(observation)
    #    self.reset()
    #    is_episode_end = False
    #    while not is_episode_end:
    #        action, _ = self.step(observation, exploit=True)
    #        if action is None: # it means self says randomly act.
    #            action = np.tanh(np.random.randn(*self._env.action_space.shape)) # random action between (-1, 1)
    #        next_observation, reward, done, truncated, _ = self._env_eval.step(self._adjust_action(action))
    #        is_episode_end = done or truncated
    #        if self._recording:
    #            self._record(next_observation, action, reward, done, truncated, time)
    #        score += reward
    #        discounted_score += pow(self.gamma, time) * reward
    #        time += 1
    #        # update observation
    #        observation = next_observation
    #    print(f"Undiscounted Score: {score}")
    #    print(f"Discounted Score: {discounted_score}")
    
    def experiment(self):
        t0 = time.perf_counter()
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        logname = os.path.join(self._env_name+self._env_tag, self.algo_name+self._algo_tag, "seed"+str(self._seed)+current_time)
        logpath = os.path.join("logs", logname)
        self._logger = Logger(path=logpath)
        self._logger.open()
        if self._tblogging:
            log_dir = os.path.join("runs", logname)
            self._tbloggingger = SummaryWriter(log_dir=log_dir)
        # log hyperparameters
        self.log_hparams(self.hparams) # log available variables..
        self.log_text("env_name", self._env_name)
        self.log_text("experiment_time", current_time)
        self.log_text("seed", str(self._seed))
        self.train() # and test sometimes..
        t1 = time.perf_counter()
        print(f"Elapsed Time: {t1-t0}")
        self.log("elapsed_time", t1-t0)        
        self._logger.close()
        self.experiment_end()
        self.save_ckpt(logpath) # save model
        if self._save_memory:
            self.memory.save_memory(logpath)
            print("Experience Memory saved. ")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--seed", type=int, default=-1)
        parser.add_argument("--num_threads", type=int, default=-1)
        parser.add_argument("--max_train_steps", type=int, default=1000000)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--eval_interval", type=int, default=1000)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--compute_period", type=int, default=-1)
        parser.add_argument("--render_mode", type=str, default="human")
        parser.add_argument("--recording", action="store_true", default=True)
        parser.add_argument("--logging", action="store_true", default=False)
        parser.add_argument("--tblogging", action="store_true", default=False)
        parser.add_argument("--algo_tag", type=str, default="")
        parser.add_argument("--env_tag", type=str, default="")        
        parser.add_argument("--save_memory", action="store_true", default=False)        
        parser.add_argument("--env_kwargs", type=json.loads, default="{}")
        #https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
        return parser