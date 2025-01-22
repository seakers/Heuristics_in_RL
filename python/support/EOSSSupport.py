# -*- coding: utf-8 -*-
"""
Support class for the EOSS design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import numpy as np
import torch
from support.EOSSDesign import EOSSDesign
from collections import OrderedDict
from itertools import compress
from copy import deepcopy

class EOSSSupport:
    def __init__(self, operations_instance, assign_prob, consider_feas, resources_path, obj_names, heur_names, heurs_used, dc_thresh, mass_thresh, pe_thresh, ic_thresh, new_reward, obj_max, include_weights):
        
        # Define class parameters
        self.obj_names = obj_names
        self.heur_names = heur_names
        self.heurs_used = heurs_used

        self.obj_max = obj_max

        self.assign_prob = assign_prob

        self.new_reward = new_reward

        self.explored_design_objectives = {}
        self.explored_design_true_objectives = {}
        self.explored_design_heuristics = {}

        self.include_weights = include_weights

        self.current_PF_objs = []

        self.current_PF_cds = [] # Crowing distances

        self.current_design_hashset = set()

        self.dc_thresh = dc_thresh
        self.mass_thresh = mass_thresh
        self.pe_thresh = pe_thresh
        self.ic_thresh = ic_thresh

        # Get operations instance (the class instance will be different depending on problem, artery or equal stiffness)
        self.operations_instance = operations_instance

        self.operations_instance.setAssigningProblem(assign_prob)
        self.operations_instance.setConsiderFeasibility(consider_feas)
        self.operations_instance.setResourcesPath(resources_path)
        self.operations_instance.setObjectiveNames(obj_names)
        self.operations_instance.setHeuristicNames(heur_names)
        self.operations_instance.setHeuristicsDeployed(heurs_used)
        self.operations_instance.setHeuristicThresholds(dc_thresh, mass_thresh, pe_thresh, ic_thresh)

        # Initialize problem instance and heuristic operators (if any) in java
        self.operations_instance.setProblem()

        self.instrument_names = list(self.operations_instance.getInstrumentNames())
        self.orbit_names = list(self.operations_instance.getOrbitNames())

    def get_instr_names(self):
        return self.instrument_names
    
    def get_orbit_names(self):
        return self.orbit_names

    ## Internal method to check Pareto objective domination
    def dominates(self, objectives, current_PF_objectives):
        # Assuming both objectives must be minimized
        dominates = False
        non_dominating = False

        domination_counter = 0
        non_domination_counter = 0
        obj_num = len(objectives)

        for i in range(len(current_PF_objectives)):
            dominate = [0] * obj_num
            for k in range(obj_num):
                if objectives[k] > current_PF_objectives[i][k]:
                    dominate[k] = 1
                elif objectives[k] < current_PF_objectives[i][k]:
                    dominate[k] = -1
            if -1 not in dominate and 1 in dominate:
                domination_counter += 1
            elif -1 in dominate and 1 in dominate:
                non_domination_counter += 1
            
        if domination_counter == 0:
            dominates = True
        
        if non_domination_counter == len(current_PF_objectives):
            non_dominating = True

        return dominates, non_dominating
    
    ## Internal method only to check non-dominance (incorporated into dominates method)
    def is_non_dominating(self, objs_current, objs_last):

        is_non_dominating = False

        dominate = [0] * len(objs_current)
        for k in range(len(objs_current)):
            if objs_current[k] > objs_last[k]:
                dominate[k] = 1
            elif objs_current[k] < objs_last[k]:
                dominate[k] = -1
        
        if -1 in dominate and 1 in dominate:
            is_non_dominating = True

        return is_non_dominating
    
    ## Method to assign current state based on the One Decision Environments (Note: Has to be modified for the Partitioning environment)
    def set_current_design(self, current_state):
        if self.new_reward:
            if self.include_weights:
                state_decisions_one_hot = current_state['design']
                #state_obj_weight0 = current_state['objective weight0']
            else:
                state_decisions_one_hot = current_state
        else:
            state_decisions_one_hot = current_state

        state_decisions = state_decisions_one_hot.argmax(dim=1) # Convert one-hot encoding to decisions array
        state_design = np.zeros(len(state_decisions), dtype=int)
        for i in range(len(state_decisions)):
            if not state_decisions[i] == 2:
                state_design[i] = state_decisions[i].numpy()
            else:
                break

        self.operations_instance.setCurrentDesign(state_design.tolist())

    ## Method to assign next design decision based on action (used for One Decision Environments) (Note: Has to be modified for the Partitioning environment)
    def assign_dec_by_action(self, state, action, obs_space):

        if self.new_reward:
            if self.include_weights:
                state_design = state['design']
                state_obj_weight0 = state['objective weight0']
            else:
                state_design = state
        else:
            state_design = state

        # Find decision to assign
        state_decisions = state_design.argmax(dim=1) # Convert one-hot encoding to decisions array
        assign_idx = len(state_decisions)
        for i in range(len(state_decisions)):
            if state_decisions[i] == 2:
                assign_idx = i
                break

        try:
            new_state_design = deepcopy(state_decisions)
            new_state_design[assign_idx] = torch.as_tensor(action)
            if self.include_weights:
                new_state = obs_space.sample()
                new_state['design'] = new_state_design
                new_state['objective weight0'] = state_obj_weight0
                #new_state = OrderedDict(('design', new_state_design), ('objective weights', state_obj_weights))
            else:
                new_state = new_state_design
        except:
            current_state = state
            current_action = action
            print("Current state: " + str(current_state))
            print("Current action: " + str(current_action))
            print("Modify by action exception")

        return assign_idx, new_state
    
    ## Method to modify state based on action
    def modify_by_action(self, state, action, obs_space):

        # Pass state and action to java operator class
        if self.new_reward:
            if self.include_weights:
                state_design = state['design']
                state_obj_weight0 = state['objective weight0']
            else:
                state_design = state
        else:
            state_design = state

        try:
            self.operations_instance.setCurrentDesign(state_design.tolist())
            self.operations_instance.setAction(np.int64(action).tolist())

            # Take action and obtain new state
            self.operations_instance.operate()
            new_state_design = np.array(self.operations_instance.getNewDesign()) # possible ways to speed up: convert to byte[] in java, import and convert to python list
            if self.new_reward:
                if self.include_weights:
                    new_state = obs_space.sample()
                    new_state['design'] = new_state_design
                    new_state['objective weight0'] = state_obj_weight0
                    #new_state = OrderedDict(('design', new_state_design), ('objective weights', state_obj_weights))
                else:
                    new_state = new_state_design
            else:
                new_state = new_state_design
        except:
            current_state = state
            current_action = action
            print("Current state: " + str(current_state))
            print("Current action: " + str(current_action))
            print("Modify by action exception")

        return new_state
    
    ## Method to determine the modified instrument and orbit (added or removed) by enacting the action(s) (Note: Has to be modified for the Partitioning environment)
    def obtain_action_modified_instrument(self, actions, one_dec, assign_idx):

        if self.assign_prob:
            if one_dec:
                dec_idx = assign_idx
            else:
                dec_idx = actions[0]
            orbit_mod = int(np.floor(dec_idx/len(self.instrument_names))) # actions (k-1)*n_instr to k*n_instr-1 -> orbit k-1
            instr_mod = dec_idx - (len(self.instrument_names)*orbit_mod)

        return self.instrument_names[instr_mod], self.orbit_names[orbit_mod]

    ## Method to compute reward based on dominance and diversity of new state compared to current PF (assuming deterministic action outcome from previous state)
    ## NOTE: NOT USING THIS SINCE THIS REWARD IS NOT MARKOVIAN
    def compute_reward(self, prev_state, state, step):
        r = 0

        # Assign rewards only if a new design is created
        new_des_bitstring = self.get_bitstring(state)
        if not new_des_bitstring in self.current_design_hashset:
            # Pass new state to evaluator and evaluate
            objs, heurs, true_objs = self.evaluate_design(state)

            if len(self.current_PF_objs) == 0: # Add evaluated objectives if Pareto Front is empty
                self.current_PF_objs.append(objs)

            self.current_design_hashset.add(new_des_bitstring)

            objs_PF = self.current_PF_objs

            # Check if current state is present in the Pareto Front
            objs_present = np.any([np.array_equal(objs, objs_test) for objs_test in objs_PF])

            is_PF = False
            r_cd = 0
            dominates, non_dominating = self.dominates(objs, objs_PF)
            if (dominates or non_dominating) and (not objs_present): # Add evaluated objectives and constraints if current design dominates the Pareto Front designs
                
                ### Compute crowding distance based reward before adding current design objectives and constraints to current Pareto Front (only if new design is non-dominating)
                if non_dominating:
                    ## Approach 1 - Reward = decrease in average crowding distance 
                    #current_PF_cds = self.compute_crowding_distances(self.current_PF_objs)
                    #PF_objs_with_design = copy.deepcopy(self.current_PF_objs)
                    #PF_objs_with_design.append(objs)
                    #PF_with_design_cds = self.compute_crowding_distances(PF_objs_with_design)

                    #PF_with_design_cds_mask = np.ma.masked_invalid(PF_with_design_cds)
                    #current_PF_cds_mask = np.ma.masked_invalid(current_PF_cds)
                    #np.ma.set_fill_value(PF_with_design_cds_mask, 10000)
                    #np.ma.set_fill_value(current_PF_cds_mask, 10000)

                    #PF_with_design_cds_fixed = np.ma.fix_invalid(PF_with_design_cds_mask).data
                    #current_PF_cds_fixed = np.ma.fix_invalid(current_PF_cds_mask).data

                    #r_cd = (np.sum(current_PF_cds_fixed)/len(current_PF_cds)) - (np.sum(PF_with_design_cds_fixed)/len(PF_with_design_cds))

                    ## Approach 2 - Reward = local crowding distance of new design
                    if not objs_present: # To account for case where domination is due to constraint violation but objectives are the same
                        r_cd = self.compute_design_crowding_distance(objs)

                self.current_PF_objs.append(objs)

                # Remove dominated designs
                #objs_PF_copy = copy.deepcopy(self.current_PF_objs)
                #constrs_PF_copy = copy.deepcopy(self.current_PF_constrs)
                remove_inds = []
                for count in range(len(self.current_PF_objs)):
                    dominates, non_dominating = self.dominates(self.current_PF_objs[count], self.current_PF_objs)
                    if not (dominates or non_dominating):
                        #self.current_PF_objs.remove(count)
                        remove_inds.append(count)

                PF_objs_array = np.array(self.current_PF_objs)
                PF_objs_array = np.delete(PF_objs_array, remove_inds, axis=0)
                
                #for index_val in remove_inds:
                    #del self.current_PF_objs[index_val]
                
                self.current_PF_objs = PF_objs_array.tolist()

                is_PF = True

            # Evaluate previous state 
            objs_prev, heurs_prev, true_objs_prev = self.evaluate_design(prev_state)

            #self.operations_instance.resetDesignGoals()
            #self.operations_instance.setCurrentDesign(prev_state.tolist())
            #self.operations_instance.evaluate()

            # Obtain objectives
            #objs_prev = list(self.operations_instance.getObjectives())
            #heurs_prev = list(self.operations_instance.getHeuristics())

            #true_objs_prev = list(self.operations_instance.getTrueObjectives())

            # Compute reward
            #r = r_cd
            #if np.any(constrs): # one or more constraints not satisfied (not equal to zero)
                #r = -100
            #else:
                #if is_PF:
                    #r = 10
                #else:
                    #r = 1

            # Reward contribution by offspring-parent dominance
            r_dom = 0
            dominates, non_dominating = self.dominates(objs, [objs_prev])
            if dominates:
                r_dom = 1
            elif non_dominating:
                r_dom = 0
            else:
                r_dom = -1

            #r += r_dom
            # Reward contribution by Pareto Front entry
            r_PF = 0
            if is_PF:
                r_PF = 100

            # Penalize reward based on time step
            r = np.min([np.exp(r_cd + r_dom), 50]) + r_PF
            r /= (step+1)
            
            self.operations_instance.resetDesignGoals()

        return r
    
    # Alternative reward computation solely based on the current state's objectives and constraints
    def compute_reward2(self, prev_state, state, nfe_val, assign_prob):

        if self.include_weights:
            prev_design = prev_state['design']
            prev_obj_weight0 = prev_state['objective weight0'][0]
            if torch.is_tensor(prev_design):
                prev_design = prev_design.detach().cpu().numpy()
                prev_obj_weight0 = prev_obj_weight0.detach().cpu().numpy()
        else:
            prev_design = prev_state
            if torch.is_tensor(prev_design):
                prev_design = prev_design.detach().cpu().numpy()
        
        if self.include_weights:
            current_design = state['design']
            obj_weight0 = state['objective weight0'][0]
            if torch.is_tensor(current_design):
                current_design = current_design.detach().cpu().numpy()
                obj_weight0 = obj_weight0.detach().cpu().numpy()

            ## Objective weights are same for old and new states (weights are changed only when the environment is reset)
            current_truss_des = EOSSDesign(design_array=current_design, weight=obj_weight0)
            current_truss_des.obtain_instruments_and_orbits(assigning_prob=assign_prob, all_instr_names=self.instrument_names, all_orbit_names=self.orbit_names)

            #obj_weights_normalized = np.divide(obj_weights, np.sum(obj_weights))
            obj_weights = [obj_weight0, (1.0 - obj_weight0)]
        else:
            current_design = state
            if torch.is_tensor(current_design):
                current_design = current_design.detach().cpu().numpy()

            current_truss_des = EOSSDesign(design_array=current_design, weight=0.0)
            current_truss_des.obtain_instruments_and_orbits(assigning_prob=assign_prob, all_instr_names=self.instrument_names, all_orbit_names=self.orbit_names)

        r = 0

        # Evaluate previous design
        prev_des_bitstring = self.get_bitstring(prev_design)
        
        if not prev_des_bitstring in list(self.explored_design_objectives.keys()):
            if self.include_weights:
                prev_truss_des = EOSSDesign(design_array=prev_design, weight=prev_obj_weight0)
            else:
                prev_truss_des = EOSSDesign(design_array=prev_design, weight=-1)

            prev_truss_des.obtain_instruments_and_orbits(assigning_prob=assign_prob, all_instr_names=self.instrument_names, all_orbit_names=self.orbit_names)

            prev_objs, prev_heurs, prev_true_objs = self.evaluate_design(prev_design) # objs are normalized with no constraint penalties added
            self.explored_design_true_objectives[prev_des_bitstring] = prev_true_objs
            self.explored_design_objectives[prev_des_bitstring] = prev_objs
            self.explored_design_heuristics[prev_des_bitstring] = prev_heurs

            nfe_val += 1
            prev_truss_des.set_objs(prev_true_objs)
            prev_truss_des.set_heur_vals(prev_heurs)
            prev_truss_des.set_nfe(nfe_val)
        else:
            if self.include_weights:
                prev_truss_des = EOSSDesign(design_array=prev_design, weight=prev_obj_weight0) 
            else:
                prev_truss_des = EOSSDesign(design_array=prev_design, weight=-1) 

            prev_truss_des.obtain_instruments_and_orbits(assigning_prob=assign_prob, all_instr_names=self.instrument_names, all_orbit_names=self.orbit_names)
    
            prev_objs = self.explored_design_objectives[prev_des_bitstring]
            prev_true_objs = self.explored_design_true_objectives[prev_des_bitstring]
            prev_heurs = self.explored_design_heuristics[prev_des_bitstring]

            prev_truss_des.set_objs(prev_true_objs)
            prev_truss_des.set_heur_vals(prev_heurs)
            prev_truss_des.set_nfe(nfe_val)
                
        # Evaluate current design
        new_des_bitstring = self.get_bitstring(current_design)
        if not new_des_bitstring in list(self.explored_design_objectives.keys()):
            objs, heurs, true_objs = self.evaluate_design(current_design) # objs are normalized with no constraint penalties added
            self.explored_design_objectives[new_des_bitstring] = objs
            self.explored_design_heuristics[new_des_bitstring] = heurs
            self.explored_design_true_objectives[new_des_bitstring] = true_objs

            nfe_val += 1
            current_truss_des.set_objs(true_objs)
            current_truss_des.set_heur_vals(heurs)
        else:
            objs = self.explored_design_objectives[new_des_bitstring]
            heurs = self.explored_design_heuristics[new_des_bitstring] 
            true_objs = self.explored_design_true_objectives[new_des_bitstring]

            current_truss_des.set_objs(true_objs)
            current_truss_des.set_heur_vals(heurs)

        current_truss_des.set_nfe(nfe_val)

        # Clear rete object every 100 NFEs
        if nfe_val % 100:
            self.operations_instance.clearRete()
            print("Rete object cleared")

        # High objective values are reduced to be within normalized range (in order to prevent gradient explosion)
        prev_objs = [prev_objs[i]/prev_objs[i] if np.abs(prev_objs[i]) > 1 else prev_objs[i] for i in range(len(prev_objs))]
        objs = [objs[i]/objs[i] if np.abs(objs[i]) > 1 else objs[i] for i in range(len(objs))]

        # If objectives are to be maximized, reverse sign of objectives (since self.dominates() assumes objectives are to be minimized)
        prev_objs = [-prev_objs[i] if self.obj_max[i] else prev_objs[i] for i in range(len(prev_objs))]
        objs = [-objs[i] if self.obj_max[i] else objs[i] for i in range(len(objs))]

        if self.include_weights:
            for obj, weight in zip(objs, obj_weights): # Assuming objectives are to be minimized, larger objective value implies lower reward (more negative)
                r += -weight*obj
                    
        else:
            dominates, non_dominating = self.dominates(objs, [prev_objs])
            if dominates:
                r_dom = 1
            elif non_dominating:
                r_dom = 0
            else:
                r_dom = -1

            #r = 100*r_dom + np.max(np.abs(objs - prev_objs))
            r = r_dom

        return r, nfe_val, prev_truss_des, current_truss_des
    
    ## Method to compute reward for the One Decision Environments
    def compute_reward_one_dec(self, state, nfe_val, assign_prob):

        if self.include_weights:
            design = state['design']
            obj_weight0 = state['objective weight0'][0]

            obj_weights = [obj_weight0, (1.0 - obj_weight0)]

            if torch.is_tensor(design):
                design = design.detach().cpu().numpy()

            current_truss_des = EOSSDesign(design_array=design, weight=obj_weight0)
        else:
            design = state
            if torch.is_tensor(design):
                design = design.detach().cpu().numpy()
            current_truss_des = EOSSDesign(design_array=design, weight=0.0)

        r = 0

        if not 2 in design: # All decisions have been assigned
            current_truss_des.obtain_instruments_and_orbits(assigning_prob=assign_prob, all_instr_names=self.instrument_names, all_orbit_names=self.orbit_names)
            new_des_bitstring = self.get_bitstring(design)
            if not new_des_bitstring in list(self.explored_design_objectives.keys()):
                objs, heurs, true_objs = self.evaluate_design(design) # objs are normalized with no constraint penalties added
                self.explored_design_objectives[new_des_bitstring] = objs
                self.explored_design_heuristics[new_des_bitstring] = heurs
                self.explored_design_true_objectives[new_des_bitstring] = true_objs
                nfe_val += 1
                current_truss_des.set_objs(true_objs)
                current_truss_des.set_heur_vals(heurs)
            else:
                objs = self.explored_design_objectives[new_des_bitstring]
                heurs = self.explored_design_heuristics[new_des_bitstring] 
                true_objs = self.explored_design_true_objectives[new_des_bitstring]
                current_truss_des.set_objs(true_objs)
                current_truss_des.set_heur_vals(heurs)

            current_truss_des.set_nfe(nfe_val)

            # Clear rete object every 100 NFEs
            if nfe_val % 100:
                self.operations_instance.clearRete()
                print("Rete object cleared")

            # If objectives are to be maximized, reverse sign of objectives (since self.dominates() assumes objectives are to be minimized)
            objs = [-objs[i] if self.obj_max[i] else objs[i] for i in range(len(objs))]

            if self.include_weights:
                for obj, weight, max_obj in zip(objs, obj_weights, self.obj_max):
                    if max_obj:
                        r += weight*obj
                    else:
                        r += -weight*obj

        return r, nfe_val, current_truss_des

    ## Method to compute the crowding distance for each design in the objectives list
    def compute_crowding_distances(self, objs_list):
        current_cds = np.zeros((len(objs_list), len(objs_list[0])))

        for i in range(len(objs_list[0])):
            current_list_obj = np.array([x[i] for x in objs_list])

            # Sort current objectives in descending order
            ascend_inds = np.argsort(current_list_obj)
            descend_inds = np.flip(ascend_inds)

            current_list_obj_sorted = current_list_obj[descend_inds]
            current_obj_max = np.max(current_list_obj_sorted)
            current_obj_min = np.min(current_list_obj_sorted)

            current_obj_cds = np.zeros((len(current_list_obj)))
            current_obj_cds[0] = np.inf
            current_obj_cds[-1] = np.inf
            current_cds[descend_inds[0],i] = current_obj_cds[0] # setting CD for max objective design
            current_cds[descend_inds[-1],i] = current_obj_cds[-1] # setting CD for min objective design

            if len(current_obj_cds) > 2:
                for j in range(1,len(current_list_obj)-1):
                    current_obj_cds[j] = (current_list_obj_sorted[j-1] - current_list_obj_sorted[j+1])/(current_obj_max - current_obj_min)
                    current_cds[descend_inds[j],i] = current_obj_cds[j]

        current_cds_aggr = np.sum(current_cds, axis=1)

        return current_cds_aggr
    
    # Method to compute the crowding distance for new Pareto design
    def compute_design_crowding_distance(self, objs_new):
        n_objs = len(objs_new)
        new_state_cds = np.zeros((n_objs))

        # Compute crowding distances for each objective
        objs_PF = deepcopy(self.current_PF_objs)
        objs_PF.append(objs_new)

        for i in range(n_objs):
            objs_current = np.array([x[i] for x in objs_PF])

            # Sort current objectives in descending order
            ascend_inds = np.argsort(objs_current)
            descend_inds = np.flip(ascend_inds)

            objs_sorted = objs_current[descend_inds]

            # Find where new objective is present and compute crowding distance accordingly
            new_obj_ind = np.where(objs_sorted == objs_new[i])[0][0]

            obj_max = np.max(objs_current)
            obj_min = np.min(objs_current)

            if (new_obj_ind == 0) or (new_obj_ind == len(objs_current)-1): # New design is outside the current Pareto Front
                new_state_cds[i] = 1
            else:
                new_state_cds[i] = (objs_sorted[new_obj_ind-1] - objs_sorted[new_obj_ind+1])/(obj_max - obj_min)

        return np.sum(new_state_cds)
    
    ## Method to change design array to bitstring to save to the hashset
    def get_bitstring(self, design_array):
        des_str = ''
        for dec in design_array:
            des_str += str(dec)
        return des_str
        
    def evaluate_design(self, design):
        if (1 in design) and (not 2 in design):
            self.operations_instance.resetDesignGoals()
            self.operations_instance.setCurrentDesign(design.tolist())
            self.operations_instance.evaluate()

            # Obtain objectives and constraints
            objs = list(self.operations_instance.getObjectives())
            heurs = list(self.operations_instance.getHeuristics())
            true_objs = list(self.operations_instance.getTrueObjectives())

        else:
            objs = np.zeros(len(self.obj_names))
            objs.fill(1)

            used_heur_names = list(compress(self.heur_names, self.heurs_used))
            heurs = np.zeros(len(used_heur_names))
            heurs.fill(10)

            true_objs = [1, 1]
            
        return objs, heurs, true_objs
