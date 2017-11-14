"""
Simulation of trading under the assumption that risky asset price follows an OU process.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import exp, sqrt

logger = logging.getLogger(__name__)

class SimTradingEnv(gym.Env):
    def __init__(self):
        self.T = 1
        self.mu = .9
        self.sigma = .4
        self.gamma = 2
        self.lmda = 1
        self.r = .05
        self.N = 100
        self.Xinit = 1
        self.Winit = 10
        self.Xlow = 0.1
        self.Wlow = 1
        self.Xhigh = 3
        self.Whigh = 30
        self.max_shares = 1000
        self.num_actions = 50
        self.dt = self.T / self.N


        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds

        self.action_space = spaces.Discrete(self.num_actions * 2 + 1)
        self.observation_space = spaces.Box(np.array([0.0,self.Xlow,self.Wlow,self.max_shares]),
                                            np.array([self.T,self.Xhigh,self.Whigh,self.max_shares]))

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        dt = self.dt
        tau, X, W, prev_shares = state
        Ntdelta = np.random.normal(size=[1]) * self.sigma \
                  * sqrt((1 - exp(-2 * self.lmda * dt)) / (2 * self.lmda))
        dX = (exp(-self.lmda * dt) - 1) * X + self.mu * (1 - exp(-self.lmda * dt)) + Ntdelta
        at = (action - self.num_actions ) / self.num_actions * W / X
        dW = at * (self.mu - X) * (1 - exp(-self.lmda * dt)) + (W - at * X) * (exp(self.r * dt) - 1) + at * Ntdelta
        X = X + dX
        W = W + dW
        tau = tau - dt
        prev_shares = at

        self.state = (tau, X, W, prev_shares)
        done = (tau == 0)

        if not done:
            reward = 0.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.T,self.Xinit,self.Winit,0.0])
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        return None
