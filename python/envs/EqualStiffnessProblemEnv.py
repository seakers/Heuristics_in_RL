# -*- coding: utf-8 -*-
"""
Gymnasium environment for Equal Stiffness Problem

@author: roshan94
"""
#import gymnasium as gym
#from gymnasium import spaces
import gym
from gym import spaces
from support.MetamaterialSupport import MetamaterialSupport
import matplotlib.pyplot as plt
import numpy as np

class EqualStiffnessProblemEnv(gym.Env):
    def __init__(self, operations_instance, n_actions, n_states, max_steps, model_sel, sel, sidenum, rad, E_mod, c_target, c_target_delta, nuc_fac, save_path, obj_names, constr_names, heur_names, heurs_used, render_steps, new_reward, obj_max, include_wts_in_state):

        super(EqualStiffnessProblemEnv, self).__init__()

        # Define problem parameters
        self.prob = 'Equal Stiffness'
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E_mod
        self.target_stiffrat = c_target
        self.render_steps = render_steps

        self.is_done = False
        self.include_weights_in_state = include_wts_in_state
        self.n_states = n_states

        self.new_reward = new_reward # Boolean representing the use of compute_reward() (dominance and diversity-based) or compute_reward2() (individual state based)         

        # Action space: defined by n_actions possible actions 
        self.action_space = spaces.Discrete(n_actions, start=0)

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

        self.metamat_support = MetamaterialSupport(sel=sel, operations_instance=operations_instance, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=c_target_delta, nuc_fac=nuc_fac, n_vars=n_states, model_sel=model_sel, artery_prob=False, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, new_reward=new_reward, obj_max=obj_max, obs_space=self.observation_space, include_weights=include_wts_in_state)

        # Initial state
        self.start_pos = self.observation_space.sample()
        self.current_pos = self.start_pos
        self.action_members = []

        # Counting number of steps
        self.step_number = 0
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # reset position to random intial position
        self.current_pos = self.observation_space.sample()
        self.action_members = []
        self.step_number = 0
        self.metamat_support.current_PF_objs = []
        self.metamat_support.current_PF_constrs = []
        self.metamat_support.current_PF_cds = []
        self.metamat_support.current_design_hashset = set()
        return self.current_pos
    
    def step(self, action, nfe_val, traj_start):
        # Modify design based on selected action (use method that calls Java Gateway)
        new_pos = self.metamat_support.modify_by_action(self.current_pos, action)
        
        # Get action members
        self.action_members, member_added = self.metamat_support.obtain_action_members()

        # Render if needed
        if self.render_steps:
            self.render(action=action, member_added=member_added)

        # Compute Reward Function
        if self.new_reward:
            reward, mod_nfe, current_truss_des, new_truss_des = self.metamat_support.compute_reward2(prev_state=self.current_pos, state=new_pos, nfe_val=nfe_val, start_of_traj=traj_start)
        else:
            reward = self.metamat_support.compute_reward(prev_state=self.current_pos, state=new_pos, step=self.step_number)

        self.current_pos = new_pos
        self.step_number += 1

        terminated = False # None of the test problems have terminal states, given that they are to be optimized

        truncated = False
        if self.step_number >= self.max_steps: # in case more than {max_steps} designs are evaluated due to batch size
            truncated = True

        done = terminated or truncated
        self.is_done = done

        kw_arg = {}
        if self.new_reward:    
            kw_arg['Current NFE'] = mod_nfe
            kw_arg['Current truss design'] = current_truss_des
            kw_arg['New truss design'] = new_truss_des

        return self.current_pos, reward, done, kw_arg
    
    def get_step_counter(self):
        return self.step_number
    
    def get_isdone(self):
        return self.is_done
    
    def render(self, action, member_added):

        # Create figure
        if self.step_number == 0:
            fig = plt.figure()
        else:
            plt.clf()

        # Get node labels
        labels = [str(i+1) for i in range(self.side_node_number**2)]

        # Get nodal connectivity/position array
        nodal_conn_array = self.metamat_support.get_nodal_position_array() 

        # Plot nodes and labels first
        plt.scatter(nodal_conn_array[:,0], nodal_conn_array[:,1], marker='*', color='#3776ab') # color - blue
        for i in range(len(labels)):
            plt.text(nodal_conn_array[i,0]-0.04*self.side_elem_length, nodal_conn_array[i,1]-0.04*self.side_elem_length, labels[i])

        # Get connectivity array for the current design
        design_CA = self.metamat_support.obtain_current_design_CA()

        # Plot current design members
        for j in range(design_CA.shape[0]): 
            # Current member
            current_member = design_CA[j,:]

            # Position of first node in member
            x1 = nodal_conn_array[int(current_member[0]-1),0]
            y1 = nodal_conn_array[int(current_member[0]-1),1]

            # Position of second node in member
            x2 = nodal_conn_array[int(current_member[1]-1),0]
            y2 = nodal_conn_array[int(current_member[1]-1),1]

            plt.plot([x1,x2], [y1,y2], color='#000000') # color - black

        # Plot actions (green - added member, red - removed member)
        #if action < self.n_states: # member addition
        if member_added: # member addition
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[int(member[0]-1),0]
                y1 = nodal_conn_array[int(member[0]-1),1]

                # Position of second node in member
                x2 = nodal_conn_array[int(member[1]-1),0]
                y2 = nodal_conn_array[int(member[1]-1),1]

                plt.plot([x1,x2], [y1,y2], color='#52a736') # color - green
            
        else: # member removal
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[int(member[0]-1),0]
                y1 = nodal_conn_array[int(member[0]-1),1]

                # Position of second node in member
                x2 = nodal_conn_array[int(member[1]-1),0]
                y2 = nodal_conn_array[int(member[1]-1),1]

                plt.plot([x1,x2], [y1,y2], color='#FF0000') # color - red       

        plt.title('Step number: ' + str(self.step_number))
        plt.show(block=False)
        plt.pause(1)

    def get_metamaterial_support(self):
        return self.metamat_support