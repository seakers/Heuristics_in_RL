# -*- coding: utf-8 -*-
"""
Support class for the EOSS design problems with support methods including evaluation and reward computation methods

@author: roshan94
"""
import numpy as np
from support.EOSSDesign import EOSSDesign
from collections import OrderedDict
import copy

class EOSSSupport:
    def __init__(self, operations_instance, instrs_names, orbits_names, n_vars, assign_prob, save_path, obj_names, heur_names, heurs_used, new_reward, obj_max, obs_space, include_weights):
        
        # Define class parameters
        self.orbits = orbits_names
        self.instrs = instrs_names

        self.n_orbits = len(orbits_names)
        self.n_instrs = len(instrs_names)

        self.obj_names = obj_names
        self.heur_names = heur_names

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
        self.operations_instance = operations_instance

        self.operations_instance.setAssignmentProblem(assign_prob)
        self.operations_instance.setSavePath(save_path) 
        self.operations_instance.setNumberOfVariables(n_vars)
        self.operations_instance.setObjectiveNames(obj_names)
        self.operations_instance.setHeuristicNames(heur_names)
        self.operations_instance.setHeuristicsDeployed(heurs_used)

        # Initialize problem instance and heuristic operators (if any) in java
        self.operations_instance.setProblem()




