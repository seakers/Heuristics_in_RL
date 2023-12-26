# -*- coding: utf-8 -*-
"""
Gymnasium environment for Artery Problem

@author: roshan94
"""
import gymnasium as gym
from gymnasium import spaces
from support.MetamaterialSupport import MetamaterialSupport
import matplotlib.pyplot as plt
import numpy as np

class ArteryProblemEnv(gym.Env):
    def __init__(self, n_actions, n_states, sel, sidenum, rad, E, c_target, obj_names, constr_names, heur_names):

        super(ArteryProblemEnv, self).__init__()

        # Define problem parameters
        self.prob = 'Artery'
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E
        self.target_stiffrat = c_target

        self.metamat_support = MetamaterialSupport.__init__(sel, sidenum, rad, E, c_target, obj_names, constr_names, heur_names)

        # Action space defined by n_actions possible actions
        self.action_space = spaces.MultiBinary(n_actions)

        # State space defined by n_states design decisions representing complete designs
        self.observation_space = spaces.MultiBinary(n_states)

        # Initial state
        self.start_pos = self.observation_space.sample()
        self.current_pos = self.start_pos

    def reset(self):
        # reset position to intial position
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        # Modify design based on selected action (use method that calls Java Gateway)
        new_pos = self.metamat_support.modify_by_action(self.current_pos, action)
        self.current_pos = new_pos

        # Compute Reward Function
        reward = self.metamat_support.compute_reward(self.current_pos, self.current_PF)

        return self.current_pos, reward, {}
    
    def render(self):

        ## Plot current design
        fig = plt.figure()

        # Plot nodes first
        plt.scatter()

        
    