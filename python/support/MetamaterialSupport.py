# -*- coding: utf-8 -*-
"""
Support class for the metamaterial design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import sys
import os
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

import numpy as np
from support.trussdesign import TrussDesign
import torch
from models.truss.TrussFeatures import TrussFeatures
from models.truss.TrussModel import TrussModel
from itertools import compress
from copy import deepcopy
from repairoperators.metamaterial.adddiagonalmember import AddDiagonalMember
from repairoperators.metamaterial.addmember import AddMember
from repairoperators.metamaterial.improveorientation import ImproveOrientation
from repairoperators.metamaterial.removeintersection import RemoveIntersection

class MetamaterialSupport:
    def __init__(self, operations_instance, use_python_model, sel, sidenum, rad, E_mod, c_target, c_target_delta, nuc_fac, n_vars, model_sel, artery_prob, save_path, obj_names, constr_names, heur_names, heurs_used, new_reward, obj_max, obs_space, include_weights):

        # Define class parameters
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E_mod
        self.target_stiffrat = c_target
        self.target_stiffrat_delta = c_target_delta
        self.heurs_used = heurs_used
        self.nuc_fac = nuc_fac

        self.obj_names = obj_names
        self.constr_names = constr_names
        self.heur_names = heur_names
        self.heur_incorp = list(compress(heur_names, heurs_used))

        self.artery_prob = artery_prob

        self.obj_max = obj_max

        self.new_reward = new_reward
        self.obs_space = obs_space

        self.explored_design_objectives = {}
        self.explored_design_true_objectives = {}
        self.explored_design_constraints = {}
        self.explored_design_heuristics = {}

        self.include_weights = include_weights

        self.current_PF_objs = []
        self.current_PF_constrs = []

        self.current_PF_cds = [] # Crowing distances

        self.current_design_hashset = set()

        # Get operations instance (the class instance will be different depending on problem, artery or equal stiffness)
        self.heur_ops = []
        self.use_python_model = use_python_model
        self.operations_instance = operations_instance
        if not self.use_python_model:
            self.operations_instance.setSideElementLength(sel) 
            self.operations_instance.setSideNodeNumber(float(sidenum)) 
            self.operations_instance.setRadius(rad) 
            self.operations_instance.setYoungsModulus(E_mod) 
            self.operations_instance.setTargetStiffnessRatio(float(c_target)) 
            self.operations_instance.setNucFac(float(nuc_fac))

            self.operations_instance.setArteryProblem(artery_prob)
            self.operations_instance.setSavePath(save_path) 
            self.operations_instance.setModelSelection(model_sel)
            self.operations_instance.setNumberOfVariables(n_vars)
            self.operations_instance.setObjectiveNames(obj_names)
            self.operations_instance.setConstraintNames(constr_names)
            self.operations_instance.setHeuristicNames(heur_names)
            self.operations_instance.setHeuristicsDeployed(heurs_used)

            # Initialize problem instance and heuristic operators (if any) in java
            self.operations_instance.setProblem()
        else:
            if artery_prob:
                problem = "Artery"
            else:
                problem = "Equal Stiffness"

            heur_ops = []

            for heur_name in self.heur_incorp:
                match heur_name:
                    case "Partial Collapsibility":
                        operator = AddDiagonalMember(sidenum=sidenum, problem=problem, sel=sel)

                    case "Nodal Properties":
                        operator = AddMember(sidenum=sidenum, problem=problem, sel=sel)

                    case "Orientation":
                        operator = ImproveOrientation(sidenum=sidenum, problem=problem, sel=sel, target_c_ratio=c_target)

                    case "Intersection":
                        operator = RemoveIntersection(sidenum=sidenum, problem=problem, sel=sel)

                    case _:
                        print("Invalid operator selection")
                        sys.exit()
                    
                heur_ops.append(operator)

            self.heur_ops = heur_ops

    ## Internal method to check constrained domination
    def dominates(self, objectives, constraints, current_PF_objectives, current_PF_constraints):
        # Assuming both objectives must be minimized
        dominates = False
        non_dominating = False

        domination_counter = 0
        non_domination_counter = 0
        obj_num = len(objectives)

        # Compute aggregate constraint violation
        aggr_constraint = np.mean(constraints)

        current_PF_aggr_constraints = np.zeros((len(current_PF_constraints)))
        for j in range(len(current_PF_constraints)):
            current_PF_aggr_constraints[j] = np.mean(current_PF_constraints[j])

        for i in range(len(current_PF_objectives)):
            # First check for aggregate constraint dominance
            if aggr_constraint > current_PF_aggr_constraints[i]:
                domination_counter += 1
            elif aggr_constraint == current_PF_aggr_constraints[i]:
                # For equal constraint satisfaction, check each objective for dominance
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
    def is_non_dominating(self, objs_current, constrs_current, objs_last, constrs_last):

        is_non_dominating = False

        if np.mean(constrs_current) == np.mean(constrs_last):
            dominate = [0] * len(objs_current)
            for k in range(len(objs_current)):
                if objs_current[k] > objs_last[k]:
                    dominate[k] = 1
                elif objs_current[k] < objs_last[k]:
                    dominate[k] = -1
            
            if -1 in dominate and 1 in dominate:
                is_non_dominating = True

        return is_non_dominating
    
    ## Method to obtain the nodal position array from sidenum and sel
    def get_nodal_position_array(self):
        nodal_position_array = np.zeros((self.side_node_number**2, 2))

        for i in range(nodal_position_array.shape[0]):
            nodal_position_array[i][0] = ((np.floor(i/self.side_node_number))/(self.side_node_number - 1)) * self.side_elem_length

        for j in range(nodal_position_array.shape[0]):
            if np.remainder(j, self.side_node_number) == 0:
                nodal_position_array[j][1] = 0
            else:
                remain = np.remainder(j, self.side_node_number)
                nodal_position_array[j][1] = (remain/(self.side_node_number - 1)) * self.side_elem_length

        return nodal_position_array
    
    ## Method to obtain connectivity array for the current design (note that the linked java class already has the current design as one of its parameters)
    def obtain_current_design_CA(self, current_design):
        if not self.use_python_model:
            return np.array(self.operations_instance.getFullConnectivityArray())
        else:
            if self.new_reward:
                if self.include_weights:
                    state_decisions_one_hot = current_design['design']
                    #state_obj_weight0 = current_design['objective weight0']
                else:
                    state_decisions_one_hot = current_design
            else:
                state_decisions_one_hot = current_design

            state_decisions = state_decisions_one_hot.argmax(dim=1) # Convert one-hot encoding to decisions array
            state_design = np.zeros(len(state_decisions), dtype=int)
            for i in range(len(state_decisions)):
                if not state_decisions[i] == 2:
                    state_design[i] = state_decisions[i].numpy()
                else:
                    break
            
            des_truss_features = TrussFeatures(bit_list=state_design.tolist(), sidenum=self.side_node_number, problem=None)
            return np.array(des_truss_features.design_conn_array)

    ## Method to obtain connectivity array for the new design (note that the linked java class already has the current design as one of its parameters)
    def obtain_new_design_CA(self, new_design):
        if not self.use_python_model:
            return np.array(self.operations_instance.getNewDesignConnectivityArray())
        else: 
            if self.new_reward:
                if self.include_weights:
                    state_decisions = new_design['design'] # State decisions are directly stored instead of the one-hot encoding
                    #state_obj_weight0 = new_design['objective weight0']
                else:
                    state_decisions = new_design
            else:
                state_decisions = new_design 

            state_design = np.zeros(len(state_decisions), dtype=int)
            for i in range(len(state_decisions)):
                if not state_decisions[i] == 2:
                    state_design[i] = state_decisions[i].numpy()
                else:
                    break

            des_truss_features = TrussFeatures(bit_list=state_design.tolist(), sidenum=self.side_node_number, problem=None)
            return np.array(des_truss_features.design_conn_array)
    
    ## Method to obtain the member added or removed based on the action (call method only after modify_by_action)
    def obtain_action_members(self, current_design, new_design):
        current_CA = self.obtain_current_design_CA(current_design=current_design)
        new_CA = self.obtain_new_design_CA(new_design=new_design)

        action_members = []

        # Find action members by comparing the members the new CA with the old CA (unique members are action members)
        # More than one action member is possible if the added/removed member is an edge member
        no_member_change = True
        member_addition = True
        if new_CA.shape[0] > current_CA.shape[0]: # member addition
            larger_CA = new_CA
            smaller_CA = current_CA
            no_member_change = False
        elif new_CA.shape[0] < current_CA.shape[0]: # member removal
            larger_CA = current_CA
            smaller_CA = new_CA
            no_member_change = False
            member_addition = False

        if not no_member_change:
            for member in larger_CA:
                member_present = False
                for or_member in smaller_CA:
                    if np.array_equal(member, or_member):
                        member_present = True
                        break
                if not member_present:
                    action_members.append(member)

        return action_members, member_addition
    
    ## Method to assign current state based on the One Decision Environments (only if operations instance is used)
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

    ## Method to assign next design decision based on action (used for One Decision Environments)
    def assign_dec_by_action(self, state, action):

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
                new_state = self.obs_space.sample()
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
    
    ## Method to set new design for One Decision environments (only if operations instance is used)
    def set_new_design(self, new_state):
        if self.new_reward:
            if self.include_weights:
                state_decisions = new_state['design']
            else:
                state_decisions = new_state
        else:
            state_decisions = new_state

        state_design = np.zeros(len(state_decisions), dtype=int)
        for i in range(len(state_decisions)):
            if not state_decisions[i] == 2:
                state_design[i] = state_decisions[i].numpy()
            else:
                break

        self.operations_instance.setNewDesign(state_design.tolist())

    ## Method to modify state based on action
    def modify_by_action(self, state, action):

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
            if not self.use_python_model:
                self.operations_instance.setCurrentDesign(state_design.tolist())
                self.operations_instance.setAction(np.int64(action).tolist())

                # Take action and obtain new state
                self.operations_instance.operate()
                new_state_design = np.array(self.operations_instance.getNewDesign()) # possible ways to speed up: convert to byte[] in java, import and convert to python list
            else: 
                if action < len(state_design):
                    new_state_design = state_design.copy()
                    new_state_design[action] = 1 - state_design[action]
                else: # Modify current design based on repair operators
                    chosen_op_idx = action - len(state_design)
                    chosen_op = self.heur_ops[chosen_op_idx]
                    chosen_op.set_design(state_design.tolist())
                    new_state_design = chosen_op.evolve()
            
            if self.new_reward:
                if self.include_weights:
                    new_state = self.obs_space.sample()
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

    ## Method to change binary design array to bitstring to save to the hashset
    def get_bitstring(self, design_array):
        des_str = ''
        for dec in design_array:
            des_str += str(dec)
        return des_str

    ## Method to compute reward based on dominance and diversity of new state compared to current PF (assuming deterministic action outcome from previous state)
    ## NOTE: NOT USING THIS SINCE THIS REWARD IS NOT MARKOVIAN
    def compute_reward(self, prev_state, state, step):
        r = 0

        # Assign rewards only if a new design is created
        new_des_bitstring = self.get_bitstring(state)
        if not new_des_bitstring in self.current_design_hashset:
            # Pass new state to evaluator and evaluate
            objs, constrs, heurs, true_objs = self.evaluate_design(state)

            if len(self.current_PF_objs) == 0: # Add evaluated objectives and constraints if Pareto Front is empty
                self.current_PF_objs.append(objs)
                self.current_PF_constrs.append(constrs)

            self.current_design_hashset.add(new_des_bitstring)

            objs_PF = self.current_PF_objs
            constrs_PF = self.current_PF_constrs

            # Check if current state is present in the Pareto Front
            objs_present = np.any([np.array_equal(objs, objs_test) for objs_test in objs_PF])
            constrs_present = np.any([np.array_equal(constrs, constrs_test) for constrs_test in constrs_PF])

            is_PF = False
            r_cd = 0
            dominates, non_dominating = self.dominates(objs, constrs, objs_PF, constrs_PF)
            if (dominates or non_dominating) and (not (objs_present and constrs_present)): # Add evaluated objectives and constraints if current design dominates the Pareto Front designs
                
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
                self.current_PF_constrs.append(constrs)

                # Remove dominated designs
                #objs_PF_copy = copy.deepcopy(self.current_PF_objs)
                #constrs_PF_copy = copy.deepcopy(self.current_PF_constrs)
                remove_inds = []
                for count in range(len(self.current_PF_objs)):
                    dominates, non_dominating = self.dominates(self.current_PF_objs[count], self.current_PF_constrs[count], self.current_PF_objs, self.current_PF_constrs)
                    if not (dominates or non_dominating):
                        #self.current_PF_objs.remove(count)
                        #self.current_PF_constrs.remove(count)
                        remove_inds.append(count)

                PF_objs_array = np.array(self.current_PF_objs)
                PF_constrs_array = np.array(self.current_PF_constrs)
                
                PF_objs_array = np.delete(PF_objs_array, remove_inds, axis=0)
                PF_constrs_array = np.delete(PF_constrs_array, remove_inds, axis=0)
                
                #for index_val in remove_inds:
                    #del self.current_PF_objs[index_val]
                    #del self.current_PF_constrs[index_val]
                
                self.current_PF_objs = PF_objs_array.tolist()
                self.current_PF_constrs = PF_constrs_array.tolist()

                is_PF = True

            # Evaluate previous state 
            objs_prev, constrs_prev, heurs_prev, true_objs_prev = self.evaluate_design(prev_state)

            #self.operations_instance.resetDesignGoals()
            #self.operations_instance.setCurrentDesign(prev_state.tolist())
            #self.operations_instance.evaluate()

            # Obtain objectives and constraints
            #objs_prev = list(self.operations_instance.getObjectives())
            #constrs_prev = list(self.operations_instance.getConstraints())
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

            # Reward contribution by constraint violation
            r_constr = (np.mean(constrs_prev) - np.mean(constrs))

            # Reward contribution by offspring-parent dominance
            r_dom = 0
            dominates, non_dominating = self.dominates(objs, constrs, [objs_prev], [constrs_prev])
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
            r = np.min([np.exp(r_cd + r_constr + r_dom), 50]) + r_PF
            r /= (step+1)
            
            if not self.use_python_model:
                self.operations_instance.resetDesignGoals()

        return r
    
    # Alternative reward computation solely based on the current state's objectives and constraints
    def compute_reward2(self, prev_state, state, nfe_val):

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
            current_truss_des = TrussDesign(design_array=current_design, weight=obj_weight0)

            #obj_weights_normalized = np.divide(obj_weights, np.sum(obj_weights))
            obj_weights = [obj_weight0, (1.0 - obj_weight0)]
        else:
            current_design = state
            if torch.is_tensor(current_design):
                current_design = current_design.detach().cpu().numpy()

            current_truss_des = TrussDesign(design_array=current_design, weight=0.0)

        r = 0

        # Evaluate previous design
        prev_des_bitstring = self.get_bitstring(prev_design)
        
        if not prev_des_bitstring in list(self.explored_design_objectives.keys()):
            if self.include_weights:
                prev_truss_des = TrussDesign(design_array=prev_design, weight=prev_obj_weight0)
            else:
                prev_truss_des = TrussDesign(design_array=prev_design, weight=-1)

            prev_objs, prev_constrs, prev_heurs, prev_true_objs = self.evaluate_design(prev_design) # objs are normalized with no constraint penalties added
            self.explored_design_true_objectives[prev_des_bitstring] = prev_true_objs
            self.explored_design_objectives[prev_des_bitstring] = prev_objs
            self.explored_design_constraints[prev_des_bitstring] = prev_constrs
            self.explored_design_heuristics[prev_des_bitstring] = prev_heurs

            nfe_val += 1
            prev_truss_des.set_objs(prev_true_objs)
            prev_truss_des.set_constr_vals(prev_constrs)
            prev_truss_des.set_heur_vals(prev_heurs)
            prev_truss_des.set_nfe(nfe_val)
        else:
            if self.include_weights:
                prev_truss_des = TrussDesign(design_array=prev_design, weight=prev_obj_weight0) 
            else:
                prev_truss_des = TrussDesign(design_array=prev_design, weight=-1) 
    
            prev_objs = self.explored_design_objectives[prev_des_bitstring]
            prev_true_objs = self.explored_design_true_objectives[prev_des_bitstring]
            prev_constrs = self.explored_design_constraints[prev_des_bitstring]
            prev_heurs = self.explored_design_heuristics[prev_des_bitstring]

            prev_truss_des.set_objs(prev_true_objs)
            prev_truss_des.set_constr_vals(prev_constrs)
            prev_truss_des.set_heur_vals(prev_heurs)
            prev_truss_des.set_nfe(nfe_val)
                
        # Evaluate current design
        new_des_bitstring = self.get_bitstring(current_design)
        if not new_des_bitstring in list(self.explored_design_objectives.keys()):
            objs, constrs, heurs, true_objs = self.evaluate_design(current_design) # objs are normalized with no constraint penalties added
            self.explored_design_objectives[new_des_bitstring] = objs
            self.explored_design_constraints[new_des_bitstring] = constrs
            self.explored_design_heuristics[new_des_bitstring] = heurs
            self.explored_design_true_objectives[new_des_bitstring] = true_objs

            nfe_val += 1
            current_truss_des.set_objs(true_objs)
            current_truss_des.set_constr_vals(constrs)
            current_truss_des.set_heur_vals(heurs)
        else:
            objs = self.explored_design_objectives[new_des_bitstring]
            constrs = self.explored_design_constraints[new_des_bitstring]
            heurs = self.explored_design_heuristics[new_des_bitstring] 
            true_objs = self.explored_design_true_objectives[new_des_bitstring]

            current_truss_des.set_objs(true_objs)
            current_truss_des.set_constr_vals(constrs)
            current_truss_des.set_heur_vals(heurs)

        current_truss_des.set_nfe(nfe_val)

        # High objective values are reduced to be within normalized range (in order to prevent gradient explosion)
        prev_objs = [prev_objs[i]/prev_objs[i] if np.abs(prev_objs[i]) > 1 else prev_objs[i] for i in range(len(prev_objs))]
        objs = [objs[i]/objs[i] if np.abs(objs[i]) > 1 else objs[i] for i in range(len(objs))]

        # High stiffness ratio constraint violations reduced to be the same magnitude as other violations (generally stiffness ratio violations for designs that cannot be evaluated by the model is 999)
        constrs_array = np.array(constrs)
        constrs_array[constrs_array >= 5] = 5
        constrs = list(constrs_array)

        prev_constrs_array = np.array(prev_constrs)
        prev_constrs_array[prev_constrs_array >= 5] = 5
        prev_constrs = list(prev_constrs_array)

        # If objectives are to be maximized, reverse sign of objectives (since self.dominates() assumes objectives are to be minimized)
        prev_objs = [-prev_objs[i] if self.obj_max[i] else prev_objs[i] for i in range(len(prev_objs))]
        objs = [-objs[i] if self.obj_max[i] else objs[i] for i in range(len(objs))]

        if self.include_weights:
            for obj, weight in zip(objs, obj_weights): # Assuming objectives are to be minimized, larger objective value implies lower reward (more negative)
                r += -weight*obj
                    
            r -= 100*np.mean(constrs) # Higher constraint violation greatly reduces reward
        else:
            dominates, non_dominating = self.dominates(objs, constrs, [prev_objs], [prev_constrs])
            if dominates:
                r_dom = 1
            elif non_dominating:
                r_dom = 0
            else:
                r_dom = -1

            #r = 100*r_dom + np.max(np.abs(objs - prev_objs))
            r = r_dom
            
            # if any(c > 1 for c in constrs):
            #     r = -10
            # else:
            #     r = 1 - np.mean(constrs)

        return r, nfe_val, prev_truss_des, current_truss_des
    
    ## Method to compute reward for the One Decision Environments
    def compute_reward_one_dec(self, state, nfe_val):

        if self.include_weights:
            design = state['design']
            obj_weight0 = state['objective weight0'][0]

            obj_weights = [obj_weight0, (1.0 - obj_weight0)]

            current_truss_des = TrussDesign(design_array=design, weight=obj_weight0)
        else:
            design = state
            current_truss_des = TrussDesign(design_array=design, weight=0.0)

        r = 0

        if not 2 in design: # All decisions have been assigned
            new_des_bitstring = self.get_bitstring(design)
            if not new_des_bitstring in list(self.explored_design_objectives.keys()):
                objs, constrs, heurs, true_objs = self.evaluate_design(design) # objs are normalized with no constraint penalties added
                self.explored_design_objectives[new_des_bitstring] = objs
                self.explored_design_constraints[new_des_bitstring] = constrs
                self.explored_design_heuristics[new_des_bitstring] = heurs
                self.explored_design_true_objectives[new_des_bitstring] = true_objs
                nfe_val += 1
                current_truss_des.set_objs(true_objs)
                current_truss_des.set_constr_vals(constrs)
                current_truss_des.set_heur_vals(heurs)
            else:
                objs = self.explored_design_objectives[new_des_bitstring]
                constrs = self.explored_design_constraints[new_des_bitstring]
                heurs = self.explored_design_heuristics[new_des_bitstring] 
                true_objs = self.explored_design_true_objectives[new_des_bitstring]
                current_truss_des.set_objs(true_objs)
                current_truss_des.set_constr_vals(constrs)
                current_truss_des.set_heur_vals(heurs)

            current_truss_des.set_nfe(nfe_val)

            # If objectives are to be maximized, reverse sign of objectives (since self.dominates() assumes objectives are to be minimized)
            objs = [-objs[i] if self.obj_max[i] else objs[i] for i in range(len(objs))]

            if self.include_weights:
                for obj, weight, max_obj in zip(objs, obj_weights, self.obj_max):
                    if max_obj:
                        r += weight*obj
                    else:
                        r += -weight*obj

                r -= 100*np.mean(constrs)

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
        
    def evaluate_design(self, design):
        if (1 in design) and (not 2 in design):
            if not self.use_python_model:
                self.operations_instance.resetDesignGoals()
                self.operations_instance.setCurrentDesign(design.tolist())
                self.operations_instance.evaluate()

                # Obtain objectives and constraints
                objs = list(self.operations_instance.getObjectives())
                constrs = list(self.operations_instance.getConstraints())
            else:
                objs, constrs, heurs, true_objs = self.compute_objectives_constraints_python(design=design)

            # Modify stiffness ratio constraint based on target delta
            if not self.artery_prob:
                stiffrat_index = self.constr_names.index('StiffnessRatioViolation')
                if np.abs(constrs[stiffrat_index]) <= self.target_stiffrat_delta:
                    constrs[stiffrat_index] = 0

            if not self.use_python_model:
                heurs = list(self.operations_instance.getHeuristics())
                true_objs = list(self.operations_instance.getTrueObjectives())

        else:
            objs = np.zeros(len(self.obj_names))
            objs.fill(1)

            constrs = np.zeros(len(self.constr_names))
            constrs.fill(10)

            heurs = np.zeros(len(self.heur_incorp))
            heurs.fill(10)

            true_objs = [1, 1]
            
        return objs, constrs, heurs, true_objs

    ### Method to compute objectives from the python model based on the problem the same way as the original java problem class (only if use_python_model is True)
    ### NOTE: The python model does not compute the connectivity constraint and heuristic soft constraints
    def compute_objectives_constraints_python(self, design):
        truss_model = TrussModel(self.side_node_number)
        stiffness_tensor, vol_frac, feas = truss_model.evaluate(design_array=design, y_modulus=self.Youngs_modulus, member_radii=self.radius, side_length=self.side_elem_length)

        stiff_11 = stiffness_tensor[0][0]
        stiff_12 = stiffness_tensor[0][1]
        stiff_21 = stiffness_tensor[1][0]
        stiff_22 = stiffness_tensor[1][1]
        stiff_16 = stiffness_tensor[0][2]
        stiff_26 = stiffness_tensor[1][2]
        stiff_61 = stiffness_tensor[2][0]
        stiff_62 = stiffness_tensor[2][1]
        stiff_66 = stiffness_tensor[2][2]
            
        objs = np.zeros(2)
        true_objs = np.zeros(2)
        heurs = np.zeros(4)
        if self.artery_prob:
            true_objs[0] = stiff_11/vol_frac
            true_objs[1] = (np.absolute((stiff_22/stiff_11) - self.target_stiffrat) + np.absolute((stiff_12/stiff_11) - 0.0745) + np.absolute((stiff_21/stiff_11) - 0.0745) + np.absolute(stiff_61/self.Youngs_modulus) + np.absolute(stiff_16/self.Youngs_modulus) + np.absolute(stiff_26/self.Youngs_modulus) + np.absolute(stiff_62/self.Youngs_modulus) + np.absolute((stiff_66/stiff_11) - 5.038))/8
                             
            objs[1] = -(true_objs[0] - 2e5)/1e6
            objs[2] = ((np.absolute((stiff_22/stiff_11) - self.target_stiffrat))/6 + np.absolute((stiff_12/stiff_11) - 0.0745) + np.absolute((stiff_21/stiff_11) - 0.0745) + np.absolute(stiff_61/1.5e5) + np.absolute(stiff_16/9e4) + np.absolute(stiff_26/9.5e4) + np.absolute(stiff_62/1.5e5) + ((np.absolute((stiff_66/stiff_11) - 5.038) - 4.5)/0.5))/8

            constrs = np.array([feas])
        else:
            true_objs[0] = stiff_22
            true_objs[1] = vol_frac

            objs[0] = -true_objs[0]/self.Youngs_modulus
            objs[1] = true_objs[1]

            constrs = np.zeros(2) # Make sure constraints are in the order in problem-config.json
            constrs[0] = feas
            stiffrat = np.absolute((stiff_22/stiff_11) - self.target_stiffrat)
            if stiffrat < self.target_stiffrat_delta:
                stiffrat = 0
            constrs[1] = stiffrat

        return objs, constrs, heurs, true_objs
