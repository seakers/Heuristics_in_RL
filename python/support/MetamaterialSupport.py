# -*- coding: utf-8 -*-
"""
Support class for the metamaterial design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import numpy as np
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

class MetamaterialSupport():
    def __init__(self, sel, sidenum, rad, E, c_target, obj_names, constr_names, heur_names, heurs_used):

        # Define class parameters
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E
        self.target_stiffrat = c_target
        self.heurs_used = heurs_used

        self.obj_names = obj_names
        self.constr_names = constr_names
        self.heur_names = heur_names

        # Access java gateway and pass parameters to operations class instance
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))

        # Get operations instance (the class instance will be different depending on problem, artery or equal stiffness)
        self.operations_instance = self.gateway.entry_point.getOperationsInstance()

        self.operations_instance.setSideElementLength(sel)
        self.operations_instance.setSideNodeNumber(sidenum)
        self.operations_instance.setRadius(rad)
        self.operations_instance.setYoungsModulus(E)
        self.operations_instance.setTargetStiffnessRatio(c_target)

        self.operations_instance.setObjectiveNames(obj_names)
        self.operations_instance.setConstraintNames(constr_names)
        self.operations_instance.setHeuristicNames(heur_names)
        self.operations_instance.setHeuristicsDeployed(heurs_used)

        # Initialize problem instance and heuristic operators (if any) in java
        self.operations_instance.setProblem()

    ## Internal method to check constrained domination
    def dominates(objectives, constraints, current_PF_objectives, current_PF_constraints):
        # Assuming both objectives must be minimized
        dominates = False

        domination_counter = 0
        obj_num = len(objectives[0])

        # Compute aggregate constraint violation
        aggr_constraint = np.mean(constraints)

        current_PF_aggr_constraints = np.zeros((len(current_PF_constraints)))
        for j in range(len(current_PF_constraints)):
            current_PF_aggr_constraints[j] = np.mean(current_PF_constraints[j])

        for i in range(len(objectives)):
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
        return self.operations_instance.getFullConnectivityArray()
    
    ## Method to obtain connectivity array for the new design (note that the linked java class already has the current design as one of its parameters)
    def obtain_new_design_CA(self):
        return self.operations_instance.getNewDesignConnectivityArray()
    
    ## Method to obtain the member added or removed based on the action (call method only after modify_by_action)
    def obtain_action_members(self):
        current_CA = self.obtain_current_design_CA()
        new_CA = self.obtain_new_design_CA()

        action_members = []

        # Find action members by comparing the members the new CA with the old CA (unique members are action members)
        # More than one action member is possible if the added/removed member is an edge member
        if len(new_CA) > len(current_CA): # member addition
            larger_CA = new_CA
            smaller_CA = current_CA
        else: # member removal
            larger_CA = current_CA
            smaller_CA = new_CA

        for member in larger_CA:
            member_present = False
            for or_member in smaller_CA:
                if np.array_equal(member, member_present):
                    member_present = True
                    break
            if not member_present:
                action_members.append(member)

        return action_members

    ## Method to modify state based on action
    def modify_by_action(self, state, action):

        # Pass state and action to java operator class
        self.operations_instance.setDesign(state)
        self.operations_instance.setAction(action)

        # Take action and obtain new state
        self.operations_instance.operate()
        new_state = self.operations_instance.getNewDesign()

        return new_state

    ## Method to compute reward based on new state (assuming deterministic action outcome from previous state)
    def compute_reward(self, state, current_PF_objs, current_PF_constrs):

        # Pass new state to evaluator and evaluate
        self.operations_instance.setDesign(state)
        self.operations_instance.evaluate()

        # Obtain objectives and constraints
        objs = self.operations_instance.getObjectives()
        constrs = self.operations_instance.getConstrs()

        # Compute reward
        r = 0
        if np.any(constrs): # one or more constraints not satisfied (not equal to zero)
            r = -100
        else:
            if self.dominates(objs, constrs, current_PF_objs, current_PF_constrs):
                r = 10
            else:
                r = 1

        return r
