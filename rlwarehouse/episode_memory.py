
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
        self._buffer_capacity = buffer_capacity
        self.clear() # clears everything 

    def clear(self):
        """Clear the buffer and all calculated stuff. Also, reset the environment.
        """
        self._ptr = 0 
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: np.empty(self._buffer_capacity, dtype=object) for field in self._fields} 
        
    def __getitem__(self, key):
        if key in self._fields:
            return self._buffer[key]
        else:
            raise KeyError("Given key is not recorded!")

    def _insert_transition(self, *data):
        """Insert one time step transition to memory
        """
        for key, value in zip(self._insert_fields, data):
            self._buffer[key][self._ptr] = value
        self._not_computed += 1
        if self._size < self._buffer_capacity:
            self._size += 1 
        self._ptr = (self._ptr + 1) % self._buffer_capacity
        #if self._ptr == self._buffer_capacity-1:
        #    self._ptr = self._buffer_capacity # increase by 1 do not make 0.
        #else:
        #    self._ptr = (self._ptr + 1) % self._buffer_capacity
        #print(self._ptr)

    def _sample_by_indices(self, indices: List[int]):
        """Sample data given the indices

        Args:
            indices (List[int]): Indices of sampled data

        Returns:
            Tuple[np.ndarray]: Sampled data
        """
        output = []
        for field in self._fields:
            sampled = np.stack(self._buffer[field][indices], axis=0)
            stacked = th.as_tensor(np.stack(sampled, axis=0), device=self._device)
            output.append(stacked)
        return tuple(output)

    def _get_last_n(self, deq, n):
        idxs = list(map(lambda i: i%self._buffer_capacity, range(self._ptr - n, self._ptr))) # ptr points to empty first field
        return np.stack(deq[idxs], axis=0)
        
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
            ptr = self._ptr if self._ptr != self._buffer_capacity else self._buffer_capacity
            for i, j in enumerate(range(ptr-len(value), ptr)):
                self._buffer[key][j] = value[i]
            #self._buffer[key][self._ptr-len(value):self._ptr] = value
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
            return self._sample_by_indices(indices)
        else:
            raise AssertionError("You cannot get a sample bigger than memory!")    
    
    def sample_batch(self, batch_size: int, batch_idx: int):
        """Sample batch data from memory with order. 

        Args:
            batch_size (int): Size of the sampled data
            batch_idx (int): Order of the batch. 

        Returns:
            Tuple[np.ndarray]: Sampled data
        """
        if self._not_computed != 0 and bool(self._derived_fields): # do we have things to compute before sampling?
            raise AssertionError("Please call compute function to compute remaining features!")
        if self._size > batch_size:
            indices = list(range(batch_idx * batch_size, (batch_idx+1)*(batch_size)))
            return self._sample_by_indices(indices)
        else:
            raise AssertionError("You cannot get a batch bigger than memory!")            

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
    
    def save_memory(self, path: os.PathLike):
        for field in self._fields:
            with open(os.path.join(path, field+".npy"), "wb") as f:
                np.save(f, self._buffer[field])
    
    def load_memory(self, path: os.PathLike):
        for field in self._fields:
            with open(os.path.join(path, field+".npy"), "rb") as f:
                self._buffer[field] = np.load(f)
