# -*- coding: utf-8 -*-
"""
Support class for the metamaterial design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import numpy as np
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

class MetamaterialSupport():
    def __init__(self, sel, sidenum, rad, E, c_target, obj_names, constr_names, heur_names):

        # Define class parameters
        self.side_elem_length = sel
        self.side_node_number = sidenum
        self.radius = rad
        self.Youngs_modulus = E
        self.target_stiffrat = c_target

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
        self.operations_instance.srtHeuristicNames(heur_names)

    ## Internal method
    def dominates(objectives, constraints, current_PF_objectives, current_PF_constraints):
        dominates = True

        #### ADD #######

        return dominates

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
