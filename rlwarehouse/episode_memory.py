
from collections import OrderedDict, deque
import datetime
import random
import os
from typing import Iterator, List, Tuple, Callable, Any, Dict, Union

import numpy as np
import torch as th


class EpisodeMemory(object):

    _main_fields = ("observation", "action", "reward", "next_observation", "done", "truncated")

    def __init__(self, 
                 buffer_capacity: int, 
                 device: str = "cuda", 
                 extra_fields: Tuple[str] = tuple(), 
                 derived_fields: Tuple[str] = tuple(), 
                 **kwargs
        ):
        """Memory initializer

        Args:
            env_name (str): Name of the environment that agent lives in.
            buffer_capacity (int): Number of timesteps that agent can remember.
            device (str): Device where the tensors are put on. 
            env_kwargs (Dict, optional): Extra environment parameteres. 
                Defaults to dict().
            extra_fields (Tuple[str], optional): Extra parameters that agent may 
                calculate such as value, hidden state etc. Defaults to tuple().
            derived_fields (Tuple[str], optional): Derived parameters at the end
                of the episode such as cumulative return. Defaults to tuple().
        """
        self._device = device
        self._extra_fields = extra_fields
        self._derived_fields = derived_fields # derived on episode end
        self._insert_fields = self._main_fields + self._extra_fields # added to memory at each step
        self._fields = self._insert_fields + self._derived_fields
        #
        self._buffer_capacity = buffer_capacity
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: deque(maxlen=self._buffer_capacity) for field in self._fields}
        self.clear() # clears everything 

    def clear(self):
        """Clear the buffer and all calculated stuff. Also, reset the environment.
        """
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: deque(maxlen=self._buffer_capacity) for field in self._fields}     

    def log(self, log_name: str, log_value: float, step: int = None):
        """Log a parameter during training

        Args:
            log_name (str): Name of the logged parameter
            log_value (float): Value of the logged parameter
            step (int): Step index
        """
        step = self._total_env_interactions if step is None else step
        self._logger.add_scalar(log_name, log_value, step)
    
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
        
    def __getitem__(self, key):
        if key in self._fields:
            return self._buffer[key]
        else:
            raise KeyError("Given key is not recorded!")

    def _insert_transition(self, *data):
        """Insert one time step transition to memory
        """
        for key, value in zip(self._insert_fields, data):
            self._buffer[key].append(value)
        self._not_computed += 1
        if self._size < self._buffer_capacity:
            self._size += 1 

    def _sample_by_indices(self, indices: List[int]):
        """Sample data given the indices

        Args:
            indices (List[int]): Indices of sampled data

        Returns:
            Tuple[np.ndarray]: Sampled data
        """
        output = []
        for field in self._fields:
            sampled = [self._buffer[field][i] for i in indices]
            # stacked = th.from_numpy(np.stack(sampled, axis=0)).to(self._device)
            stacked = th.as_tensor(np.stack(sampled, axis=0), device=self._device)
            output.append(stacked)
        return tuple(output)

    def _get_last_n(self, deq, n):
        idxs = range(self._size - n, self._size)
        elements = [deq[i] for i in idxs]
        return np.stack(elements, axis=0)
        
    def _compute(self, agent: Callable):
        """Compute derived values and save to memory. Call it at the end of 
        episode.

        Args:
            agent (Agent): Agent object

        """
        if self._not_computed == 0:
            return None
        episode_args = (self._get_last_n(self._buffer[key], self._not_computed) for key in self._insert_fields)
        episode_results = agent.compute_function(*episode_args)
        for key, value in zip(self._derived_fields, episode_results):
            self._buffer[key].extend(value)
        self._not_computed = 0

    def sample(self, sample_size: int):
        """Sample data randomly from memory with a given size

        Args:
            sample_size (int): Size of the sampled data

        Raises:
            AssertionError: Raised if necessary derived fields are not computed.

        Returns:
            Tuple[np.ndarray]: Sampled data
        """
        if self._not_computed != 0 and bool(self._derived_fields): # do we have things to compute before sampling?
            raise AssertionError("Please call compute function to compute remaining features!")
        if self._size > sample_size:
            indices = random.sample(range(self._size), sample_size)
        else:
            multiple, remainder = sample_size // self._size, sample_size % self._size
            indices = random.sample(range(self._size), remainder) + multiple*list(range(self._size))
            random.shuffle(indices)
        return self._sample_by_indices(indices)

    def sample_last(self, sample_size: int):
        """Sample latest data from memory with a given size

        Args:
            sample_size (int): Size of the sampled data

        Raises:
            AssertionError: Raised if necessary derived fields are not computed.

        Returns:
            Tuple[np.ndarray]: Sampled data
        """
        if self._not_computed != 0 and bool(self._derived_fields): # do we have things to compute before sampling?
            raise AssertionError("Please call compute function to compute remaining features!")
        if self._size > sample_size:
            indices = list(range(self._size-1, self._size-1-sample_size, -1))
        else:
            multiple, remainder = sample_size // self._size, sample_size % self._size
            indices = indices = list(range(self._size-1, self._size-1-remainder, -1)) + multiple*list(range(self._size))
        return self._sample_by_indices(indices)

