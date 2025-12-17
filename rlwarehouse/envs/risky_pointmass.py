
import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

# Modified from the source: https://github.com/JasonMa2016/CODAC/
# Paper: https://arxiv.org/pdf/2107.06106
#
# The state space of the PointMass agent 4-dimensional, including
# the agent’s position as well as the goal position, which is fixed to [0.1, 0.1]. The state space constraint
# is [0, 1]. Hence, the agent cannot enter a location outside of this unit square. The risky red region
# is centered at [0.5, 0.5] with radius of 0.3. The agent’s initial state is randomly chosen inside the
# [0.1, 0.9]2 box outside the risky red region. The agent dynamics is holomorphic, allowing the agent
# to move freely in any direction with its x-axis and y-axis displacement capped at 0.1. The reward
# the agent receives at each step is its negative Euclidean distance to the goal plus a constant −0.1,
# which encourages the agent to reach the goal as fast as possible. When the agent is inside the risky
# red region, with probability 0.1, an additional −50 reward is incurred. The episode terminates when
# the agent is within 0.1 distance to the goal. An episode may last up to 100 steps.

class RiskyPointMass(gym.Env):
    def __init__(self, N=1, risk_prob=0.1, risk_penalty=10, eval=False, render_mode=None, max_episode_steps=100):
        # Car parameterss
        self.v_max = 0.1
        # Environment parameters
        self.d_goal = 0.05
        self.init_pos = np.array([0.95, 0.95])
        self.risk_prob = risk_prob
        self.risk_penalty = risk_penalty
        self.eval = eval
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.low_state = 0
        self.high_state= 1

        self.min_actions = np.array(
            [-self.v_max, -self.v_max], dtype=np.float32
        )
        self.max_actions = np.array(
            [self.v_max, self.v_max], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(2+2, ),
            dtype=np.float32
        )

        self.goal = np.array([0.05, 0.05])
        self.r = 0.3 # obstacle radius
        self.centers = np.array([0.5, 0.5])

        # Step 4: Rendering parameters
        self.screen_size = [600, 600]
        self.screen_scale = 600
        self.background_color = [255, 255, 255]
        self.wall_color = [0, 0, 0]
        self.circle_color = [255, 0, 0]
        self.safe_circle_color = [200,0,0]
        self.lidar_color = [0, 0, 255]
        self.goal_color = [0, 255, 0]
        self.robot_color = [0, 0, 0]
        self.safety_color = [255, 0, 0]
        self.goal_size = 15
        self.radius = 9
        self.width = 3
        self.pygame_init = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        return [seed]

    def reset(self,
        seed: int | None = None,
        options: dict | None = None,
        ):
        
        self.t = 0
        if self.eval:
            self.init_pos = np.array([0.95, 0.95])
        else:        
            sampled = False
            while not sampled:
                # uniform state space initial state distribution
                self.init_pos = self.np_random.uniform(0.05, 0.95, size=(2,))
                if self.is_safe(self.init_pos):
                    sampled = True

        self.state = np.array(list(self.init_pos) + list(self.goal))
        return np.array(self.state), {'cost': 0}


    def get_dist_to_goal(self, state):
        return np.linalg.norm(state[-2:]-state[:2])

    # Check if the state is safe.
    def is_safe(self, state):
        if len(state.shape) == 1:
            safe = True
            d_circle = (state[0]-self.centers[0])**2 + (state[1]-self.centers[1])**2
            if d_circle <= (self.r ** 2):
                safe = False
            return safe

    # calculate failure risk
    def calc_risk(self, state):
        safe = True
        d_circle = (state[0]-self.centers[0])**2 + (state[1]-self.centers[1])**2
        return self.risk_prob*np.exp(-4*d_circle**2/self.r**2)

    def step(self, action):
        action = np.clip(action, -self.v_max, self.v_max)
        assert self.action_space.contains(action)

        d_goal = self.get_dist_to_goal(self.state)
        reward = - d_goal - 0.1
        cost = 0
        u = np.random.uniform(0, 1)
        if u > self.calc_risk(self.state):
            cost = 1
            reward -= self.risk_penalty

        done = False
        if d_goal < self.d_goal:
            done = True

        self.state[:2] = self.state[:2] + action
        self.state = np.clip(self.state, self.low_state, self.high_state)

        if self.render_mode == "human":
            self.render()
        
        self.t += 1
        truncated = self.t>=self.max_episode_steps
        
        return np.array(self.state), reward, done, truncated, {'cost': cost}

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.pygame_init = True
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.background_color)

        p_car = self.state[:2]

        p = (self.screen_scale * p_car).astype(int).tolist()
        pygame.draw.circle(self.screen, self.robot_color, p, self.radius, self.width)

        c, r = (self.screen_scale*self.centers[:2]).astype(int), int(self.screen_scale*self.r)
        pygame.draw.circle(self.screen, self.circle_color, c, r)

        pygame.draw.circle(self.screen, self.goal_color, (self.screen_scale * self.goal).astype(int), self.goal_size)
        pygame.display.flip()

        self.clock.tick(20)