# -*- coding: utf-8 -*-
"""
Plotting HV evolution of the training runs for the PPO Agent

@author: roshan94
"""
from Utils.dataHandler import DataHandler
from Utils.normalizationHandler import NormalizationHandler
from Utils.mOORunStatistics import MOORunStatistics
from Utils.mOOCaseStatistics import MOOCaseStatistics
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt

results_dir = 'C:\\SEAK Lab\\Coev results\\'
problem_dir = 'EqStiff\\' # EqStiff, Artery

constr_names = ''
obj_names = ['TrueObjective1','TrueObjective2']
heuristic_names = ['P','N','O','I']
#obj_names = ['Normalized Stiffness', 'Normalized Volume Fraction']

#constr_names = ['FeasibilityViolation','ConnectivityViolation','StiffnessRatioViolation']
constr_names = ['ConnectivityViolation','StiffnessRatioViolation']

# Set parameters for DataHandler.get_objectives() method
objs_norm_num = [0, 0] 
objs_norm_den = [1.8162e6, 1] # Youngs modulus used to normalize stiffness
objs_max = [False, False] 

# To be set to true if negative of any objective is to be used to compute HV, 
# first objective (stiffness) is to be maximized and second objective (volume fraction/deviation) is to be minimized, however -normalized stiffness is stored in csv so -1 multiplication is not required
true_obj_names = [r'$C_{22}$',r'$v_f$']
if problem_dir == 'Artery\\':
    #obj_names = ['Normalized Stiffness', 'Normalized Deviation']
    constr_names = ['FeasibilityViolation','ConnectivityViolation']
    # Set parameters for DataHandler.get_objectives() method
    objs_norm_num = [2e5, 0]
    objs_norm_den = [1e6, 1]
    true_obj_names = [r'$\frac{C_{11}}{v_f}$',r'deviation']
    