
from collections import OrderedDict, deque
import datetime
import random
import os
from typing import Iterator, List, Tuple, Callable, Any, Dict, Union

import numpy as np
import gymnasium
import torch as th

from tensorboardX import SummaryWriter

class EpisodeMemory(object):

    _main_fields = ("observation", "action", "reward", "next_observation", "done", "truncated")

    def __init__(self, 
                 env_name: str,
                 buffer_capacity: int, 
                 compute_period: int, 
                 device: str = "cuda", 
                 env_kwargs: Dict = dict(), 
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
        self._compute_period = compute_period
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: deque(maxlen=self._buffer_capacity) for field in self._fields}
        self._env = gymnasium.make(env_name, render_mode="human", **env_kwargs)
        self._episode_max_time = self._env._max_episode_steps # environment time limit
        self._episode_time = 0 # episode counter
        self._total_env_interactions = 0
        self._clear_record()
        self.clear() # clears everything and resets the environment
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        logdir = os.path.join("runs", env_name + "_" + current_time)
        self._logger = SummaryWriter(logdir=logdir)
        self.log_text("envname", env_name)
        self.log_text("experiment_time", current_time)

    @property
    def env_info(self):
        return {
            "observation_shape": self._env.observation_space.shape,
            "action_shape": self._env.action_space.shape, 
            "observation_space": self._env.observation_space, 
            "action_space": self._env.action_space, 
        }
    
    def _clear_record(self, obs: Union[np.ndarray, None] = None):
        """The agent saves state-action history, this function clears the records.

        Args:
            obs (Union[np.array, None], optional): Initial state after reset call.
        """
        self._obs_history = np.zeros((self._episode_max_time+1, *self._env.observation_space.shape), dtype=self._env.observation_space.dtype)
        if obs is not None:
            self._obs[0] = obs # clear while adding initial state to record
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

    def clear(self):
        """Clear the buffer and all calculated stuff. Also, reset the environment.
        """
        self.last_episode_score = 0
        self.episode_score = 0
        self.observation, _ = self._env.reset()   
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

    def _adjust_action(self, action):
        """This function takes action output of network bounded in (-1, 1)
        and converts it into environment bounds for action taking. """
        l, h = self._env.action_space.low, self._env.action_space.high
        return (h+l)/2 + (h-l)/2 * action

    def one_step_rollout(self, agent: Callable, record=False):
        """Rollout in the environment with agent and save transitions to memory.

        Args:
            agent (Agent): Agent object with step method. 
            record (bool, optional): Recording history flag. Defaults to False.

        Returns:
            float: Last episode score
        """
        action, extra = agent.step(self.observation)
        if action is None: # it means agent says randomly act.
            action = self._env.action_space.sample()
        # next_observation, reward, done, truncated, _ = self._env.step(action)
        next_observation, reward, done, truncated, _ = self._env.step(self._adjust_action(action))
        is_episode_end = done or truncated
        if record:
            if self._episode_time == 0:
                self._clear_record()
            self._record(next_observation, action, reward, done, self._episode_time)
        self._total_env_interactions += 1
        self._episode_time += 1
        self.episode_score += reward
        self._insert_transition(
            np.float32(self.observation), 
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
            self._compute(agent) 
        if is_episode_end: # end of the episode
            self.observation, _ = self._env.reset()
            agent.reset()
            self.log("episode_score", self.episode_score)
            print('\rTime step {}\tScore: {:.2f}'.format(self._total_env_interactions, self.episode_score), end="")
            self.last_episode_score = self.episode_score
            self.episode_score = 0
        else:
            self.observation = next_observation
        return self.last_episode_score 
