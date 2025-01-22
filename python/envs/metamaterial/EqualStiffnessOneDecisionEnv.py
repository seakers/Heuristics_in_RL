# -*- coding: utf-8 -*-
"""
Gymnasium environment for Metamaterial Equal Stiffness Problem 
(but where each step is assigning one design decision, i.e. one episode is generating one complete design)

@author: roshan94
"""
import gymnasium as gym
from gymnasium import spaces
#import gym
#from gym import spaces
from support.MetamaterialSupport import MetamaterialSupport
import matplotlib.pyplot as plt
import numpy as np

class EqualStiffnessOneDecisionEnv(gym.Env):
    def __init__(self, operations_instance, n_states, model_sel, sel, sidenum, rad, E_mod, c_target, c_target_delta, nuc_fac, save_path, obj_names, constr_names, heur_names, heurs_used, render_steps, new_reward, obj_max, include_wts_in_state):

        super(EqualStiffnessOneDecisionEnv, self).__init__()

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

        self.new_reward = new_reward # Boolean representing the use of compute_reward()

        # Action space: defined by either 1 or 0
        self.action_space = spaces.Discrete(2)

        # State space: defined by n_states design decisions representing complete designs
        nvec = np.zeros(n_states)
        nvec.fill(3) # Total n_states number of decisions, each with 3 possibilities (0, 1, 2)
        if self.include_weights_in_state:
            self.observation_space = spaces.Dict(
                {
                    "design": spaces.MultiDiscrete(nvec),
                    "objective weight0": spaces.Box(low=0.0, high=1.0, shape=(len(obj_names)-1,), dtype=np.float32) # This problem has only 2 objectives so the other weight is just 1 - this weight
                }
            )
        else:
            self.observation_space = spaces.MultiDiscrete(nvec)

        self.metamat_support = MetamaterialSupport(sel=sel, operations_instance=operations_instance, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=c_target_delta, nuc_fac=nuc_fac, n_vars=n_states, model_sel=model_sel, artery_prob=False, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, new_reward=True, obj_max=obj_max, obs_space=self.observation_space, include_weights=include_wts_in_state)

        # Initial state
        self.start_pos = self.observation_space.sample()
        if self.include_weights_in_state:
            self.start_pos['design'].fill(2)
        else:
            self.start_pos.fill(2) # unassigned decisions are assigned 2
        self.current_pos = self.start_pos
        self.current_truss_des = None
        self.action_members = []

        # Counting number of steps
        self.step_number = 0
        self.max_steps = n_states
        self.current_nfe_val = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # reset position to random intial position
        self.current_pos = self.observation_space.sample()
        if self.include_weights_in_state:
            self.current_pos['design'].fill(2)
        else:
            self.current_pos.fill(2) # unassigned decisions are assigned 2
        self.action_members = []
        self.step_number = 0
        self.metamat_support.current_PF_objs = []
        self.metamat_support.current_PF_constrs = []
        self.metamat_support.current_PF_cds = []
        self.metamat_support.current_design_hashset = set()
        return self.current_pos, {}
    
    def step(self, action):
        # Assign design decision based on selected action (use method that calls Java Gateway)
        assign_idx, new_pos = self.metamat_support.assign_dec_by_action(self.current_pos, action)
        prev_pos = self.current_pos

        # Setting the current design based on the current state
        self.metamat_support.set_current_design(current_state=self.current_pos)

        # Setting the new design based on the new state
        self.metamat_support.set_new_design(new_state=new_pos)

        # Get action members
        self.action_members, self.member_added = self.metamat_support.obtain_action_members()
        
        # Compute Reward Function
        reward, mod_nfe, new_truss_des = self.metamat_support.compute_reward_one_dec(state=new_pos, nfe_val=self.current_nfe_val)
                
        self.current_pos = new_pos
        self.current_truss_des = new_truss_des
        self.step_number += 1

        terminated = False # None of the test problems have terminal states, given that they are to be optimized
        if self.step_number >= self.max_steps: # in case more than {max_steps} designs are evaluated due to batch size
            terminated = True

        truncated = False

        self.is_done = terminated or truncated

        # Render if needed
        if self.render_steps:
            self.render(action=action, dec_assigned=assign_idx, prev_state=prev_pos, new_state=new_pos, new_des=new_truss_des)

        kw_arg = {}
        if self.new_reward:    
            self.current_nfe_val = mod_nfe
            kw_arg['Current NFE'] = mod_nfe
            kw_arg['New truss design'] = new_truss_des

        return self.current_pos, reward, terminated, truncated, kw_arg
    
    def get_step_counter(self):
        return self.step_number
    
    def get_isdone(self):
        return self.is_done
    
    def render(self, action):

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
        design_CA = self.metamat_support.obtain_current_design_CA2()

        # Get new design objectives and constraints
        if self.is_done:
            if self.new_reward:
                new_objs = self.current_truss_des.get_objs()
                new_constrs = self.current_truss_des.get_constrs()
            else:
                new_norm_objs, new_constrs, new_heurs, new_objs = self.metamat_support.evaluate_design(self.current_pos)

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

        # Plot actions (solid green line - member added, light red dashed line - member not added)
        if action: # member added
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[int(member[0]-1),0]
                y1 = nodal_conn_array[int(member[0]-1),1]

                # Position of second node in member
                x2 = nodal_conn_array[int(member[1]-1),0]
                y2 = nodal_conn_array[int(member[1]-1),1]

                plt.plot([x1,x2], [y1,y2], color='#52a736') # color - green
            
        else: # member not added
            for member in self.action_members:
                # Position of first node in member
                x1 = nodal_conn_array[int(member[0]-1),0]
                y1 = nodal_conn_array[int(member[0]-1),1]

                # Position of second node in member
                x2 = nodal_conn_array[int(member[1]-1),0]
                y2 = nodal_conn_array[int(member[1]-1),1]

                plt.plot([x1,x2], [y1,y2], color='#FF6347', linestyle='dashed') # color - light tomato

        if self.is_done:
            plt.title('Step number: ' + str(self.step_number) + '\n New Design Objectives: ' + str(new_objs) + '\n New Design Constraints: ' + str(new_constrs))
        else:
            plt.title('Step number: ' + str(self.step_number))

        plt.show(block=False)
        plt.pause(1)

    def get_metamaterial_support(self):
        return self.metamat_support
