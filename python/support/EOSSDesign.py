# -*- coding: utf-8 -*-
"""
Support class representing an EOSS design of integer decisions (generalised for both Assigning and Partitioning problems)

@author: roshan94
"""
import numpy as np

class EOSSDesign:
    def __init__(self, design_array, weight):
        self.dec_array = design_array
        self.obj_weight = weight # assuming only two objectives (other objective weight = 1 - weight)
        self.instrs_des = {} # dictionary of instruments in an orbit
        self.nfe_val = 0
        self.obj_vals = []
        self.heur_vals = []

    def set_nfe(self, nfe):
        self.nfe_val = nfe

    def get_weight(self):
        return self.obj_weight

    def get_design(self):
        return self.dec_array
    
    def obtain_instruments_and_orbits(self, assigning_prob, all_instr_names, all_orbit_names): # NOTE: Modify for partitioning problem
        if assigning_prob:
            for i in range(len(all_orbit_names)):
                if np.any(np.equal(self.dec_array[i*len(all_instr_names):(i+1)*len(all_instr_names)],1)): # Check if there is any instrument assigned to the orbit
                    self.instrs_des[all_orbit_names[i]] = []
                    for j in range(len(all_instr_names)):
                        if self.dec_array[i*len(all_instr_names)+j] == 1:
                            self.instrs_des[all_orbit_names[i]] = all_instr_names[j]

    def get_instrs_and_orbits(self): # Run obtain_instruments_and_orbits() first
        return self.instrs_des
    
    def get_nfe(self):
        return self.nfe_val
    
    def set_objs(self, objs):
        self.obj_vals = objs

    def set_heur_vals(self, heurs):
        self.heur_vals = heurs

    def get_heurs(self):
        return self.heur_vals
    
    def get_objs(self):
        return self.obj_vals