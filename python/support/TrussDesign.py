# -*- coding: utf-8 -*-
"""
Support class representing a metamaterial design of binary decisions

@author: roshan94
"""
import numpy as np

class TrussDesign:
    def __init__(self, design_array, weight):
        self.dec_array = design_array
        self.obj_weight = weight # assuming only two objectives (other objective weight = 1 - weight)
        self.nfe_val = 0
        self.obj_vals = []
        self.constr_vals = []
        self.heur_vals = []

    def set_nfe(self, nfe):
        self.nfe_val = nfe

    def get_weight(self):
        return self.obj_weight

    def get_design(self):
        return self.dec_array
    
    def get_nfe(self):
        return self.nfe_val
    
    def set_objs(self, objs):
        self.obj_vals = objs

    def set_constr_vals(self, constrs):
        self.constr_vals = constrs

    def set_heur_vals(self, heurs):
        self.heur_vals = heurs

    def get_heurs(self):
        return self.heur_vals
    
    def get_constrs(self):
        return self.constr_vals
    
    def get_objs(self):
        return self.obj_vals
