"""Generic RL Agent definition
"""

from argparse import ArgumentParser
import json
import os
import time
import math
import numpy as np
from datetime import datetime
from typing import Tuple, Union, Dict, Any

import torch as th
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
    
    def __init__(self, device: str = "cpu", start_steps: int = 10000, **memory_kwargs):
        """Summary
        
        Args:
            device (str, optional): Device in which algorithm works on (cpu, cuda, etc.)
            **memory_kwargs: Other arguments for episode memory such like environment, size etc. 
        """
        self._device = device
        self._start_steps = start_steps
        self.memory = EpisodeMemory(device=self._device,
                                    extra_fields=self.extra_fields, 
                                     derived_fields=self.derived_fields, 
                                     **memory_kwargs)

    @property   
    def device(self) -> str:
        """Retrieve device currently being used by minibatch."""
        return self._device
    
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
    def total_env_interactions(self):
        return self.memory._total_env_interactions
    
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

    def episode_function(self, *args):
        """This function is called after episode ending to populate necessary variables 
        in the experience replay.

        Returns:
            Tuple: Tuple of episodic data, such as return-to-go.
        """
        return ()

    def log(self, log_name: str, log_value: float):
        """Log a parameter during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
        """
        self.memory.log(log_name, log_value)

    def log_hparams(self, dict: Dict[str, Union[bool, str, float, int]]):
        """Log set of hyperparameters

        Args:
            dict (Dict): Dictionary to log
        """
        self.memory.log_hparams(dict)

    def train(self, max_steps: int = int(5e5), 
            test_interval: int = None, 
        ):
        """Main train loop.
        
        Args:
            max_steps (int, optional): Maximum number steps for training
        
        Returns:
            TYPE: Description
        """
        for i in range(max_steps): 
            self.memory.one_step_rollout(self) # step inside episode memory
            if i > self._start_steps:
                self.learn_on_step() # learn here.
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
        parser.add_argument("--buffer_capacity", type=int, default=1000000)
        parser.add_argument("--start_steps", type=int, default=10000)
        parser.add_argument("--device", type=str, default="cuda")
        #parser.add_argument("--env_kwargs", type=json.loads())
        return parser