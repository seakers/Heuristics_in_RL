# -*- coding: utf-8 -*-
"""
Support class for the metamaterial design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import numpy as np
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

class MetamaterialSupport:
    def __init__(self, sel, sidenum, rad, E, c_target, nuc_fac, n_vars, model_sel, artery_prob, save_path, obj_names, constr_names, heur_names, heurs_used):

        # Define class parameters
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E
        self.target_stiffrat = c_target
        self.heurs_used = heurs_used
        self.nuc_fac = nuc_fac

        self.save_path = save_path
        self.obj_names = obj_names
        self.constr_names = constr_names
        self.heur_names = heur_names

        self.current_PF_objs = {}
        self.current_PF_constrs = {}

        self.step_counter = 0

        # Access java gateway and pass parameters to operations class instance
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))

        # Get operations instance (the class instance will be different depending on problem, artery or equal stiffness)
        self.operations_instance = self.gateway.entry_point.getOperationsInstance()

        self.operations_instance.setSideElementLength(sel) 
        self.operations_instance.setSideNodeNumber(float(sidenum)) 
        self.operations_instance.setRadius(rad) 
        self.operations_instance.setYoungsModulus(E) 
        self.operations_instance.setTargetStiffnessRatio(c_target) 
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

    ## Internal method to check constrained domination
    def dominates(self, objectives, constraints, current_PF_objectives, current_PF_constraints):
        # Assuming both objectives must be minimized
        dominates = False

        domination_counter = 0
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

        if domination_counter == 0:
            dominates = True

        return dominates
    
    ## Method to obtain the nodal position array from sidenum and sel
    def get_nodal_position_array(self):
        nodal_position_array = np.zeros((self.side_node_number, self.side_node_number))

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
    def obtain_current_design_CA(self):
        return np.array(self.operations_instance.getFullConnectivityArray())
    
    ## Method to obtain connectivity array for the new design (note that the linked java class already has the current design as one of its parameters)
    def obtain_new_design_CA(self):
        return np.array(self.operations_instance.getNewDesignConnectivityArray())
    
    ## Method to obtain the member added or removed based on the action (call method only after modify_by_action)
    def obtain_action_members(self):
        current_CA = self.obtain_current_design_CA()
        new_CA = self.obtain_new_design_CA()

        action_members = []

        # Find action members by comparing the members the new CA with the old CA (unique members are action members)
        # More than one action member is possible if the added/removed member is an edge member
        no_member_change = True
        if new_CA.shape[0] > current_CA.shape[0]: # member addition
            larger_CA = new_CA
            smaller_CA = current_CA
            no_member_change = False
        elif new_CA.shape[0] < current_CA.shape[0]: # member removal
            larger_CA = current_CA
            smaller_CA = new_CA
            no_member_change = False

        if not no_member_change:
            for member in larger_CA:
                member_present = False
                for or_member in smaller_CA:
                    if np.array_equal(member, or_member):
                        member_present = True
                        break
                if not member_present:
                    action_members.append(member)

        return action_members

    ## Method to modify state based on action
    def modify_by_action(self, state, action):

        # Pass state and action to java operator class
        self.operations_instance.setCurrentDesign(state.tolist())
        self.operations_instance.setAction(action.tolist())

        # Take action and obtain new state
        self.operations_instance.operate()
        new_state = np.array(self.operations_instance.getNewDesign()) # possible ways to speed up: convert to byte[] in java, import and convert to python list

        return new_state
    
    ## Method to update step number
    def update_step_number(self):

        self.step_counter += 1

    ## Method to compute reward based on new state (assuming deterministic action outcome from previous state)
    def compute_reward(self, state):

        # Pass new state to evaluator and evaluate
        self.operations_instance.setCurrentDesign(state.tolist())
        self.operations_instance.evaluate()

        # Obtain objectives and constraints
        objs = list(self.operations_instance.getObjectives())
        constrs = list(self.operations_instance.getConstraints())
        heurs = list(self.operations_instance.getHeuristics())

        true_objs = list(self.operations_instance.getTrueObjectives())

        if len(self.current_PF_objs) == 0: # Add evaluated objectives and constraints if Pareto Front is empty
            self.current_PF_objs[self.step_counter] = objs
            self.current_PF_constrs[self.step_counter] = constrs

        objs_PF = list(self.current_PF_objs.values())
        constrs_PF = list(self.current_PF_constrs.values())

        is_PF = False
        if self.dominates(objs, constrs, objs_PF, constrs_PF): # Add evaluated objectives and constraints if current design dominates the Pareto Front designs
            self.current_PF_objs[self.step_counter] = objs
            self.current_PF_constrs[self.step_counter] = constrs
            is_PF = True

        # Compute reward
        r = 0
        if np.any(constrs): # one or more constraints not satisfied (not equal to zero)
            r = -100
        else:
            if is_PF:
                r = 10
            else:
                r = 1

        return r
