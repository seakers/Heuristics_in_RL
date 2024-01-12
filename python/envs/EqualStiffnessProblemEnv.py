# -*- coding: utf-8 -*-
"""
Gymnasium environment for Equal Stiffness Problem

@author: roshan94
"""
import gymnasium as gym
from gymnasium import spaces
from support.MetamaterialSupport import MetamaterialSupport
import matplotlib.pyplot as plt
import numpy as np

class EqualStiffnessProblemEnv(gym.Env):
    def __init__(self, n_actions, n_states, max_steps, model_sel, sel, sidenum, rad, E, c_target, nuc_fac, save_path, obj_names, constr_names, heur_names, heurs_used):

        super(EqualStiffnessProblemEnv, self).__init__()

        # Define problem parameters
        self.prob = 'Equal Stiffness'
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E
        self.target_stiffrat = c_target

        self.metamat_support = MetamaterialSupport.__init__(sel, sidenum, rad, E, c_target, nuc_fac, n_vars=n_states, model_sel=model_sel, artery_prob=False, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used)

        # Action space: defined by n_actions possible actions 
        self.action_space = spaces.Discrete(n_actions, start=0)

        # State space: defined by n_states design decisions representing complete designs
        self.observation_space = spaces.MultiBinary(n_states)

        # Initial state
        self.start_pos = self.observation_space.sample()
        self.current_pos = self.start_pos
        self.action_members = []

        # Counting number of steps
        self.step_number = 0
        self.max_steps = max_steps

    def reset(self, seed=None):
        super().reset(seed=seed)

        # reset position to intial position
        self.current_pos = self.start_pos
        self.action_members = []
        return self.current_pos
    
    def step(self, action):
        # Modify design based on selected action (use method that calls Java Gateway)
        new_pos = self.metamat_support.modify_by_action(self.current_pos, action)
        
        # Get action members
        self.action_members = self.metamat_support.obtain_action_members()

        self.current_pos = new_pos

        # Compute Reward Function
        reward = self.metamat_support.compute_reward(self.current_pos, self.current_PF)

        self.step_number += 1

        terminated = False # None of the test problems have terminal states, given that they are to be optimized

        truncated = False
        if self.step_number == self.max_steps:
            truncated = True

        return self.current_pos, reward, terminated, truncated, {}
    
    def render(self, action):

        # Get node labels
        labels = [str(i) for i in range(self.side_node_number**2)]

        # Get nodal connectivity/position array
        nodal_conn_array = self.metamat_support.get_nodal_position_array() 

        # Create figure
        fig = plt.figure()

        # Plot nodes and labels first
        plt.scatter(nodal_conn_array[:,0], nodal_conn_array[:,1], marker='*', color='#3776ab') # color - blue
        for i in range(len(labels)):
            plt.text(nodal_conn_array[i,0]-0.01*self.side_elem_length, nodal_conn_array[i,1]-0.01*self.side_elem_length, labels[i])

        # Get connectivity array for the current design
        design_CA = self.metamat_support.obtain_design_CA()

        # Plot current design members
        for j in range(design_CA[0]): # CURRENT POS IS ONLY BOOLEAN, GET FULL CONNECTIVITY ARRAY AND PLOT MEMBERS
            # Current member
            current_member = design_CA[j,:]

            # Position of first node in member
            x1 = nodal_conn_array[current_member[0],0]
            y1 = nodal_conn_array[current_member[0],1]

            # Position of second node in member
            x2 = nodal_conn_array[current_member[1],0]
            y2 = nodal_conn_array[current_member[1],1]

            plt.plot([x1,x2], [y1,y2], color='#000000') # color - black

        # Plot actions (green - added member, red - removed member)
        if action < nodal_conn_array.shape[0]: # member addition
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[member[0],0]
                y1 = nodal_conn_array[member[0],1]

                # Position of second node in member
                x2 = nodal_conn_array[member[1],0]
                y2 = nodal_conn_array[member[1],1]

                plt.plot([x1,x2], [y1,y2], color='#52a736') # color - green
            plt.title('Member added')
        elif action > nodal_conn_array.shape[0] : # member removal
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[member[0],0]
                y1 = nodal_conn_array[member[0],1]

                # Position of second node in member
                x2 = nodal_conn_array[member[1],0]
                y2 = nodal_conn_array[member[1],1]

                plt.plot([x1,x2], [y1,y2], color='#FF0000') # color - red      
            plt.title('Member removed')
        else:
            plt.title('No change')  

        