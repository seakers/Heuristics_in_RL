# -*- coding: utf-8 -*-
"""
Gymnasium environment for EOSS Assignment Problem

@author: roshan94
"""
import gymnasium as gym
from gymnasium import spaces
#import gym
#from gym import spaces
from support.eosssupport import EOSSSupport
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np

class AssignmentProblemEnv(gym.Env):
    def __init__(self, operations_instance, resources_path, obj_names, heur_names, heurs_used, consider_feas, dc_thresh, mass_thresh, pe_thresh, ic_thresh, render_steps, new_reward, obj_max, include_wts_in_state):

        super(AssignmentProblemEnv, self).__init__()

        # Define problem parameters
        self.prob = 'Assigning'
        self.render_steps = render_steps

        self.is_done = False
        self.include_weights_in_state = include_wts_in_state
        
        self.new_reward = new_reward # Boolean representing the use of compute_reward() (dominance and diversity-based) or compute_reward2() (individual state based)     

        self.eoss_support = EOSSSupport(operations_instance=operations_instance, assign_prob=True, consider_feas=consider_feas, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, new_reward=new_reward, obj_max=obj_max, include_weights=include_wts_in_state)

        self.instr_names = self.eoss_support.get_instr_names()
        self.orbit_names = self.eoss_support.get_orbit_names()

        n_states = len(self.instr_names)*len(self.orbit_names)

        # Action space: defined by n_actions (= n_states for this problem) possible actions + number of heuristics used
        n_heurs_used = heurs_used.count(True)
        self.action_space = spaces.Discrete(n_states+n_heurs_used, start=0) 

        # State space: defined by n_states design decisions representing complete designs
        if new_reward:
            if self.include_weights_in_state:
                self.observation_space = spaces.Dict(
                    {
                        "design": spaces.MultiBinary(n_states),
                        "objective weight0": spaces.Box(low=0.0, high=1.0, shape=(len(obj_names)-1,), dtype=np.float32) # This problem has only 2 objectives so the other weight is just 1 - this weight
                    }
                )
            else:
                self.observation_space = spaces.MultiBinary(n_states)
        else:
            self.observation_space = spaces.MultiBinary(n_states)

        self.n_states = n_states

        # Initial state
        self.start_pos = self.observation_space.sample()
        self.current_pos = self.start_pos
        self.action_members = []

        # Counting number of steps
        self.step_number = 0
        self.current_nfe_val = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # reset position to random intial position
        self.current_pos = self.observation_space.sample()
        self.action_members = []
        self.step_number = 0
        self.eoss_support.current_PF_objs = []
        self.eoss_support.current_PF_cds = []
        self.eoss_support.current_design_hashset = set()
        return self.current_pos, {}
    
    def step(self, action):
        # Modify design based on selected action (use method that calls Java Gateway)
        new_pos = self.eoss_support.modify_by_action(self.current_pos, [action], self.observation_space)

        # Get instrument modified (added/removed) by action
        instr_modified, orbit_modified = self.eoss_support.obtain_action_modified_instrument(actions=[action], one_dec=False, assign_idx=0) # assign_idx not used for this environment

        # Compute Reward Function
        if self.new_reward:
            reward, mod_nfe, current_eoss_des, new_eoss_des = self.eoss_support.compute_reward2(prev_state=self.current_pos, state=new_pos, nfe_val=self.current_nfe_val, assign_prob=True)
        else:
            reward = self.eoss_support.compute_reward(prev_state=self.current_pos, state=new_pos, step=self.step_number)
            new_eoss_des = None

        # Render if needed
        if self.render_steps:
            self.render(action=action, instr_modified=instr_modified, orb_modified=orbit_modified, new_state=new_pos, new_des=new_eoss_des)
                
        self.current_pos = new_pos
        self.step_number += 1

        terminated = False # None of the test problems have terminal states, given that they are to be optimized

        truncated = False

        self.is_done = terminated or truncated

        kw_arg = {}
        if self.new_reward:    
            self.current_nfe_val = mod_nfe
            kw_arg['Current NFE'] = mod_nfe
            kw_arg['Current EOSS design'] = current_eoss_des
            kw_arg['New EOSS design'] = new_eoss_des

        return self.current_pos, reward, terminated, truncated, kw_arg
    
    def get_step_counter(self):
        return self.step_number
    
    def get_isdone(self):
        return self.is_done
    
    def get_n_states(self):
        return self.n_states
    
    def render_state(self, axes, orbit_instr_map, state):

        #TODO

        for i in range(len(orbit_instr_map)):
            current_orbit_instr_map = orbit_instr_map[self.orbit_names[i]]
            for j in range(len(current_orbit_instr_map)):
                current_instr_pos = current_orbit_instr_map[self.instr_names[j]]


    def render(self, action, instr_modified, orbit_modified, new_state, new_des): # TODO: Change to matrix of rectangles for each instrument and orbit that change color and border based on presence and modification
        n_orbits = len(self.orbit_names)
        n_instrs = len(self.instr_names)
        height = 1.0/n_orbits
        width = 1.0/n_orbits

        # Create figure
        if self.step_number == 0:
            fig, ax = plt.subplots()

            # Create orbit row map (starting locations for the instruments of each orbit to be listed)
            # as instruments are added to each orbit, the value for that orbit is updated
            current_orbit_row_map = {}

            for i in range(n_orbits):
                orbit_instrument_cell_map = {}
                for j in range(n_instrs):
                    orbit_instrument_cell_map[self.instr_names[j]] = (float(i/n_orbits) + j*width + 0.01, (i*height) + float(i/n_orbits) + 0.01) # lower left positions for each instrument in the orbit
                current_orbit_row_map[self.orbit_names[i]] = orbit_instrument_cell_map 

        else:
            plt.clf()

        # Get new design objectives and constraints
        if self.new_reward:
            new_objs = new_des.get_objs()

            new_instrs = new_des.get_instruments() # should be a dict with orbits as keys and list of instruments in each orbit as values
            new_orbs = new_des.get_orbits()
        else:
            new_norm_objs, new_heurs, new_objs = self.eoss_support.evaluate_design(new_state)

        # plot instruments and orbits in the new state
        # for orbit in new_orbs:
        #     ax.add_artist()
        #     instrs_orbit = new_instrs[orbit]

        #     for instr in instrs_orbit:

        #         # Plot current instrument
        #         current_instr_map = current_orbit_row_map[orbit]
        #         current_orbit_lower_left_width = current_orbit_pos[0]
        #         current_orbit_lower_left_height = current_orbit_pos[1]

        #         p = plt.Rectangle(current_orbit_pos, width=width, height=height, fill=False)
        #         ax.add_patch(p)

        #         text_pos = (current_orbit_pos[0] + 0.5*height, current_orbit_pos[1] + 0.5*width)
        #         ax.text(text_pos[0], text_pos[1], instr)

        #         # Modify current orbit position map
        #         current_orbit_row_map[orbit] = (current_orbit_lower_left_width + width, current_orbit_lower_left_height)

    def get_eoss_support(self):
        return self.eoss_support
