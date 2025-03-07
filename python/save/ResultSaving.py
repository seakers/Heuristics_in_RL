# -*- coding: utf-8 -*-
"""
Class to save results into csv files

@author: roshan94
"""
import csv
import numpy as np
from itertools import compress
from models.truss.TrussModel import TrussModel

class ResultSaver:
    def __init__(self, save_path, operations_instance, artery_prob, use_python_model, obj_names, constr_names, heur_names, new_reward, include_weights, radius, side_elem_length, side_node_number, Youngs_modulus, target_stiffrat, c_target_delta):
        
        # Define class parameters
        self.save_path = save_path # must include {filename}.csv
        self.operations_instance = operations_instance # use operations instance from 
        self.artery_prob = artery_prob
        self.objective_names = obj_names
        self.constraint_names = constr_names
        self.heuristic_names = heur_names
        self.new_reward = new_reward
        self.include_weights = include_weights
        self.target_stiffrat_delta = c_target_delta
        self.radius = radius
        self.side_elem_length = side_elem_length
        self.side_node_number = side_node_number
        self.Youngs_modulus = Youngs_modulus
        self.target_stiffrat = target_stiffrat
        self.use_python_model = use_python_model

        self.explored_design_true_objectives = {}
        self.explored_design_objectives = {}
        self.explored_design_constraints = {}
        self.explored_design_heuristics = {}
        self.step_logger = {}

    def save_to_logger2(self, step_number, action, truss_design, reward):
        current_design = truss_design.get_design()
        current_state = "".join([str(dec) for dec in current_design])
        step_info = {}
        step_info['Observation'] =  current_state # bitstring or array as string

        if self.new_reward:
            # Hard coded for two objectives for now
            if self.include_weights:
                step_info['Objective Weight 0'] = truss_design.get_weight()
                step_info['Objective Weight 1'] = 1 - truss_design.get_weight()

        # Save step results with step number as key
        step_info['Step Number'] = step_number
        step_info['Action'] = action
        step_info['Reward'] = reward
        step_info['NFE'] = truss_design.get_nfe()
        
        true_objectives = truss_design.get_objs()
        constraints = truss_design.get_constrs()
        heuristics = truss_design.get_heurs()

        for i in range(len(self.objective_names)):
            step_info[self.objective_names[i]] = true_objectives[i]

        for j in range(len(self.constraint_names)):
            step_info[self.constraint_names[j]] = constraints[j]

        for k in range(len(self.heuristic_names)):
            step_info[self.heuristic_names[k]] = heuristics[k]
        
        self.step_logger[step_number] = step_info

    def save_to_logger(self, artery_prob, step_number, action, prev_obs, reward):
        
        # Note: make sure objectives, constraints and heuristics are consistent with the order of the names
        if self.new_reward:
            if self.include_weights:
                prev_obs_design = np.int32(prev_obs[:(len(prev_obs) - (len(self.objective_names)-1))])
                current_state = "".join([str(dec) for dec in prev_obs_design])
            else:
                current_state = "".join([str(dec) for dec in prev_obs])
        else:
            current_state = "".join([str(dec) for dec in prev_obs])
        
        # Save step results with step number as key (TODO: find and save NFE separately)
        step_info = {}
        step_info['Step Number'] = step_number
        step_info['Action'] = action
        step_info['Observation'] =  current_state # bitstring or array as string
        if self.new_reward:
            # for i in range(len(self.objective_names)-1):
            #     step_info['Objective Weight ' + str(i)] = prev_obs[-(len(self.objective_names)-i)]
            # step_info['Objective Weight ' + str(i+1)] = 1 - prev_obs[-(len(self.objective_names)-i)]
            
            # Hard coded for two objectives for now
            if self.include_weights:
                step_info['Objective Weight 0'] = prev_obs[-1]
                step_info['Objective Weight 1'] = 1 - prev_obs[-1]

        step_info['Reward'] = reward

        ## OLd formulation
        # Evaluate a new state or extract objectivesm constraints and heuristics for previously explored design
        if not current_state in list(self.explored_design_objectives.keys()):
            if self.new_reward:
                #prev_obs_design = np.int32(prev_obs[:(len(prev_obs) - len(self.objective_names))])
                true_objectives, objectives, constraints, heuristics = self.evaluate_design(artery_prob=artery_prob, design=prev_obs_design)
            else:
                true_objectives, objectives, constraints, heuristics = self.evaluate_design(artery_prob=artery_prob, design=prev_obs)
            self.explored_design_objectives[current_state] = true_objectives
            self.explored_design_constraints[current_state] = constraints
            self.explored_design_heuristics[current_state] = heuristics
        else:
            true_objectives = self.explored_design_objectives[current_state]
            constraints = self.explored_design_constraints[current_state]
            heuristics = self.explored_design_heuristics[current_state]

        for i in range(len(self.objective_names)):
            step_info[self.objective_names[i]] = true_objectives[i]

        for j in range(len(self.constraint_names)):
            step_info[self.constraint_names[j]] = constraints[j]

        for k in range(len(self.heuristic_names)):
            step_info[self.heuristic_names[k]] = heuristics[k]
        
        self.step_logger[step_number] = step_info

    def save_to_logger_pytorch(self, metamat_prob, artery_prob, step_number, action, prev_obs, reward, obj_weight=-1):

        # Note: make sure objectives, constraints and heuristics are consistent with the order of the names
        prev_design = prev_obs
        obj_weight0 = obj_weight
                
        prev_des_bitstr = "".join([str(dec) for dec in prev_design])

        if not prev_des_bitstr in list(self.explored_design_objectives.keys()):
            prev_true_objs, prev_objs, prev_constrs, prev_heurs  = self.evaluate_design(metamat_prob=metamat_prob, artery_prob=artery_prob, design=prev_design) # objs are normalized with no constraint penalties added
            self.explored_design_true_objectives[prev_des_bitstr] = prev_true_objs
            self.explored_design_objectives[prev_des_bitstr] = prev_objs
            self.explored_design_constraints[prev_des_bitstr] = prev_constrs
            self.explored_design_heuristics[prev_des_bitstr] = prev_heurs
        else:
            prev_objs = self.explored_design_objectives[prev_des_bitstr]
            prev_true_objs = self.explored_design_true_objectives[prev_des_bitstr]
            prev_constrs = self.explored_design_constraints[prev_des_bitstr]
            prev_heurs = self.explored_design_heuristics[prev_des_bitstr]

        # Save step results with step number as key (TODO: find and save NFE separately)
        step_info = {}
        step_info['Step Number'] = step_number
        step_info['Action'] = action
        step_info['Observation'] = prev_des_bitstr # bitstring or array as string
        step_info['Reward'] = reward
        step_info['NFE'] = len(self.explored_design_objectives)
        if self.new_reward:
            # for i in range(len(self.objective_names)-1):
            #     step_info['Objective Weight ' + str(i)] = prev_obs[-(len(self.objective_names)-i)]
            # step_info['Objective Weight ' + str(i+1)] = 1 - prev_obs[-(len(self.objective_names)-i)]
            
            # Hard coded for two objectives for now
            if self.include_weights:
                step_info['Objective Weight 0'] = obj_weight0
                step_info['Objective Weight 1'] = 1 - obj_weight0

        step_info['Reward'] = reward

        # Save step results with step number as key
        for i in range(len(self.objective_names)):
            step_info[self.objective_names[i]] = prev_true_objs[i]

        for j in range(len(self.constraint_names)):
            step_info[self.constraint_names[j]] = prev_constrs[j]

        for k in range(len(self.heuristic_names)):
            step_info[self.heuristic_names[k]] = prev_heurs[k]
        
        self.step_logger[step_number] = step_info

        return len(self.explored_design_objectives)
        

    def save_to_csv(self):

        # Initialize csv filewriter and write to file using dictwriter
        if self.new_reward:
            field_names = ['Step Number', 'NFE', 'Observation']
            if self.include_weights:
                for i in range(len(self.objective_names)):
                    field_names.append('Objective Weight ' + str(i))
            field_names.append('Action')
            field_names.append('Reward')
        else:
            field_names = ['Step Number', 'NFE', 'Observation', 'Action', 'Reward']
        field_names.extend(self.objective_names)
        field_names.extend(self.constraint_names)
        field_names.extend(self.heuristic_names)

        with open(self.save_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)

            writer.writeheader()
            writer.writerows(list(self.step_logger.values()))

    def evaluate_design(self, metamat_prob, artery_prob, design):

        # Set and evaluate current design in Java
        if 1 in design:
            if not self.use_python_model:
                self.operations_instance.resetDesignGoals()
                self.operations_instance.setCurrentDesign(design.tolist())
                self.operations_instance.evaluate()

                # Obtain objectives and constraints
                objs = list(self.operations_instance.getObjectives())
                if metamat_prob:
                    constrs = list(self.operations_instance.getConstraints())
                else:
                    constrs = []

                # Modify stiffness ratio constraint based on target delta
                if (metamat_prob) and (not artery_prob):
                    stiffrat_index = self.constraint_names.index('StiffnessRatioViolation')
                    if np.abs(constrs[stiffrat_index]) <= self.target_stiffrat_delta:
                        constrs[stiffrat_index] = 0

                heurs = list(self.operations_instance.getHeuristics())

                true_objs = list(self.operations_instance.getTrueObjectives())
            elif metamat_prob:
                objs, constrs, heurs, true_objs = self.compute_objectives_constraints_python(design=design)

        else:
            objs = np.zeros(len(self.objective_names))
            objs.fill(1)

            if metamat_prob:
                constrs = np.zeros(len(self.constraint_names))
                constrs.fill(10)
            else:
                constrs = []

            heurs = np.zeros(len(self.heuristic_names))
            heurs.fill(10)

            true_objs = [1, 1]

        return true_objs, objs, constrs, heurs
    
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

            constrs = np.zeros(2)
            constrs[0] = feas
            stiffrat = np.absolute((stiff_22/stiff_11) - self.target_stiffrat)
            if stiffrat < self.target_stiffrat_delta:
                stiffrat = 0
            constrs[1] = stiffrat

        return objs, constrs, heurs, true_objs