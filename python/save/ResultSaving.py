# -*- coding: utf-8 -*-
"""
Class to save results into csv files

@author: roshan94
"""
import csv
import numpy as np

class ResultSaver:
    def __init__(self, save_path, operations_instance, obj_names, constr_names, heur_names, new_reward, include_weights, c_target_delta):
        
        # Define class parameters
        self.save_path = save_path # must include {filename}.csv
        self.operations_instance = operations_instance # use operations instance from 
        self.objective_names = obj_names
        self.constraint_names = constr_names
        self.heuristic_names = heur_names
        self.new_reward = new_reward
        self.include_weights = include_weights
        self.target_stiffrat_delta = c_target_delta

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

    def save_to_logger(self, step_number, action, prev_obs, reward):
        
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
                true_objectives, objectives, constraints, heuristics = self.evaluate_design(prev_obs_design)
            else:
                true_objectives, objectives, constraints, heuristics = self.evaluate_design(prev_obs)
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

    def evaluate_design(self, design):

        # Set and evaluate current design in Java
        self.operations_instance.resetDesignGoals()
        self.operations_instance.setCurrentDesign(design.tolist())
        self.operations_instance.evaluate()

        # Obtain objectives and constraints
        objs = list(self.operations_instance.getObjectives())
        constrs = list(self.operations_instance.getConstraints())

        # Modify stiffness ratio constraint based on target delta
        stiffrat_index = self.constraint_names.index('StiffnessRatioViolation')
        if np.abs(constrs[stiffrat_index]) <= self.target_stiffrat_delta:
            constrs[stiffrat_index] = 0

        heurs = list(self.operations_instance.getHeuristics())

        true_objs = list(self.operations_instance.getTrueObjectives())

        return true_objs, objs, constrs, heurs

