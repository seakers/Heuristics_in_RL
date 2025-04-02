# -*- coding: utf-8 -*-
"""
Base class for metamaterial heuristic repair operators
Contains methods to obtain common truss design features

@author: roshan94
"""
from abc import abstractmethod
from models.truss.TrussFeatures import TrussFeatures
from models.truss.stiffness.generateNC import generateNC
import numpy as np
from itertools import compress

class BaseOperator:
    def __init__(self, sidenum, problem, sel):
        self.design = None
        self.sidenum = sidenum
        self.problem = problem

        self.design_truss_features = None
        self.nodal_positions = generateNC(sel=sel, sidenum=sidenum)
        self.repeatable_member_indices = None

    @abstractmethod
    def evolve(self):
        """
        Each child operator will implement its own version of evolve().
        This merely necessitates the definition of the evolve class for each child operator
        """
        pass

    def set_design(self, design_bits):
        self.design = design_bits
        self.design_truss_features = TrussFeatures(bit_list=self.design, sidenum=self.sidenum, problem=self.problem)
        self.repeatable_member_indices = self.get_repeated_member_indices()
      
    def get_design_connectivity_array(self):
        return self.design_truss_features.design_conn_array
    
    def get_full_connectivity_array(self):
        return self.design_truss_features.connectivity_array
    
    def get_repeatable_design_bits(self, design_conn_array):

        full_conn_array = self.get_full_connectivity_array()

        # Get full bitstring
        design_bitstring = []
        for member in full_conn_array:
            if member in design_conn_array:
                design_bitstring.append(1)
            else:
                design_bitstring.append(0)

        # Convert to repeatable bitstring
        design_rep_bitstring = [dec for idx, dec in enumerate(design_bitstring) if idx not in self.repeatable_member_indices]

        return design_rep_bitstring

    def get_repeated_member_indices(self):

        full_conn_array = self.get_full_connectivity_array()

        top_edge_nodes = self.design_truss_features.get_top_edge_nodes(sidenum=self.sidenum)

        rep_member_indices = []

        for i in range(len(full_conn_array)):
            member = full_conn_array[i]

            # Check if member is a top edge member
            if np.all([True if node in top_edge_nodes else False for node in member]):
                rep_member_indices.append(i)

            # Check if member is a right edge member
            if np.all([True if node >= (self.sidenum**2 - self.sidenum + 1) else False for node in member]):
                rep_member_indices.append(i)

        return rep_member_indices
    
    def get_absent_members(self):

        design_conn_array = self.get_design_connectivity_array()
        full_conn_array = self.get_full_connectivity_array()

        # Get all absent members in design
        abs_members_ind = [False if x in design_conn_array else True for x in full_conn_array]
        abs_members = list(compress(full_conn_array, abs_members_ind))

        return abs_members
        
    def add_member_to_design(self, member_to_add):

        design_conn_array = self.get_design_connectivity_array()

        add_loc = 0
        for member in design_conn_array:
            if member[0] >= member_to_add[0]:
                if member[1] > member_to_add[1]:
                    break
            if member[0] > member_to_add[0]:
                break
            add_loc += 1

        design_conn_array.insert(add_loc, member_to_add)
        
        return design_conn_array
    