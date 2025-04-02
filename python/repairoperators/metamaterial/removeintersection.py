# -*- coding: utf-8 -*-
"""
Repair operator class for the intersection heuristic for the metamaterial problems
Repurposed from RemoveIntersection2.java

Action: Remove a member (at random if there are more than one candidate member) with the maximum number of intersections

@author: roshan94
"""
from repairoperators.metamaterial.baseoperator import BaseOperator
import numpy as np
import copy

class RemoveIntersection(BaseOperator):

    def __init__(self, sidenum, problem, sel):
        super(RemoveIntersection, self).__init__(sidenum=sidenum, problem=problem, sel=sel)

    def evolve(self): # Make sure to call set_design() before this method
        super().evolve() # doesn't return anything

        design_conn_array = self.get_design_connectivity_array()

        # Find members with one or more intersections
        inters_mems = self.get_intersecting_members(design_CA=design_conn_array)

        # Create list of intersecting members and their number of intersections
        mem_n_inters = []
        n_inters = []
        for inters_mem_idx in list(inters_mems.keys()):
            mem_n_inters.append([inters_mem_idx, len(inters_mems[inters_mem_idx])])
            n_inters.append(len(inters_mems[inters_mem_idx]))
        
        # Find member(s) with the maximum number of intersections
        max_n_inters = np.max(n_inters)
        mems_idx_max_inters = [x[0] for x in mem_n_inters if x[1] == max_n_inters]

        # Choose member to remove
        member_to_be_removed = False
        if len(mems_idx_max_inters) == 1:
            member_rem = design_conn_array[mems_idx_max_inters[0]]
            member_to_be_removed = True
        elif len(mems_idx_max_inters) > 0:
            member_rem_idx = np.random.randint(len(mems_idx_max_inters))
            member_rem = design_conn_array[mems_idx_max_inters[member_rem_idx]]
            member_to_be_removed = True

        if member_to_be_removed:
            new_design_CA = copy.copy(design_conn_array)
            new_design_CA.remove(member_rem)

            # Get repeatable design bitstring
            new_design_bits = self.get_repeatable_design_bits(design_conn_array=new_design_CA)
        else:
            new_design_bits = copy.copy(self.design)

        return new_design_bits

    def get_intersecting_members(self, design_CA):

        mem_intersections = {}
        for member_idx_a in range(len(design_CA)):
            member_a = design_CA[member_idx_a]
            member_a_node1_pos, member_a_node2_pos = self.nodal_positions[member_a[0]-1], self.nodal_positions[member_a[1]-1]
            inters_mem_idxs = []
            for member_idx_b in range(member_idx_a+1, len(design_CA)):
                member_b = design_CA[member_idx_b]
                member_b_node1_pos, member_b_node2_pos = self.nodal_positions[member_b[0]-1], self.nodal_positions[member_b[1]-1]
                intersects = self.design_truss_features.intersect(A=member_a_node1_pos, B=member_a_node2_pos, C=member_b_node1_pos, D=member_b_node2_pos)
                if intersects:
                    inters_mem_idxs.append(member_idx_b)
            if len(inters_mem_idxs) > 0:
                mem_intersections[member_idx_a] = inters_mem_idxs

        return mem_intersections
