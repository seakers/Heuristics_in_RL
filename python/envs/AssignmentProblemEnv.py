# -*- coding: utf-8 -*-
"""
Gymnasium environment for Assignment Problem

@author: roshan94
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AssignmentProblemEnv(gym.Env):
    def __init__(self, n_actions, n_states):
        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.MultiBinary(n_states)
        