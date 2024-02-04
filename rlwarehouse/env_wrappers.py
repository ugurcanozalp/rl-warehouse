
import gymnasium as gym
from collections import deque
import numpy as np


class BoxToHistoryBox(gym.ObservationWrapper):
    '''This wrapper converts the environment which returns last h observations.
    First h-1 observations are made NaN as environment is reset. 
    '''
    def __init__(self, env, history_length=8):
        super().__init__(env)
        self._history_length = history_length
        self._obs_memory = deque(maxlen=self._history_length)
        shape = (self._history_length,) + self.observation_space.shape
        low = np.repeat(np.expand_dims(self.observation_space.low, 0), self._history_length, axis=0)
        high = np.repeat(np.expand_dims(self.observation_space.high, 0), self._history_length, axis=0)    
        self.observation_space = gym.spaces.Box(low, high, shape)

    def add_to_memory(self, obs):
        self._obs_memory.append(np.expand_dims(obs, axis=0))

    def observation(self, obs):
        self.add_to_memory(obs)
        return np.concatenate(self._obs_memory)

    def reset(self):
        reset_state = self.env.reset()
        for i in range(self.h-1):
            self.add_to_memory(np.full_like(reset_state, float("nan"))) # unavailable observations
        return self.observation(reset_state)