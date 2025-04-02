# -*- coding: utf-8 -*-
"""
Repair operator class for the orientation heuristic for the metamaterial problems
Repurposed from ImproveOrientation2.java

Action: Add member that reduces the deviation to target orientation the most

@author: roshan94
"""
from repairoperators.metamaterial.baseoperator import BaseOperator
import numpy as np
import math
import copy

class ImproveOrientation(BaseOperator):

    def __init__(self, sidenum, problem, sel, target_c_ratio):
        super(ImproveOrientation, self).__init__(sidenum=sidenum, problem=problem, sel=sel)
        self.target_orientation = math.degrees(np.arctan(target_c_ratio))

    def evolve(self): # Make sure to call set_design() before this method
        super().evolve() # doesn't return anything

        design_conn_array = self.get_design_connectivity_array()

        abs_members = self.get_absent_members()

        # Find difference between mean orientation of design and target orientation
        mean_or = self.find_mean_orientation(design_CA=design_conn_array)

        or_diff = np.abs(mean_or - self.target_orientation)

        member_to_be_added = False
        if not or_diff == 0:
            # Find members with minimum offset (when added to design) from target orientation
            candidate_members_and_ordiffs = {}
            for member_idx in range(len(abs_members)):
                member = abs_members[member_idx]
                abs_node1_pos, abs_node2_pos = self.nodal_positions[member[0]-1], self.nodal_positions[member[1]-1]
                abs_member_or = self.calculate_member_orientation(member_node1_pos=abs_node1_pos, member_node2_pos=abs_node2_pos)

                mean_or_mod = ((mean_or*len(design_conn_array)) + abs_member_or)/(len(design_conn_array) + 1)

                candidate_members_and_ordiffs[member_idx] = np.abs(mean_or_mod - self.target_orientation)

            ordiffs = list(candidate_members_and_ordiffs.values())

            min_ordiff = np.min(ordiffs)
            if min_ordiff < or_diff:
                min_ordiff_idxs = [i for i,x in enumerate(ordiffs) if x == np.min(ordiffs)]
            else:
                min_ordiff_idxs = []

            if len(min_ordiff_idxs) > 0:
                if len(min_ordiff_idxs) > 1:
                    member_idx_choice = min_ordiff_idxs[np.random.randint(len(min_ordiff_idxs))]
                    member_add_idx = list(candidate_members_and_ordiffs.keys())[member_idx_choice]
                    member_add = abs_members[member_add_idx]
                else:
                    member_add_idx = list(candidate_members_and_ordiffs.keys())[min_ordiff_idxs[0]]
                    member_add = abs_members[member_add_idx]

                member_to_be_added = True

        if member_to_be_added:
            new_design_CA = self.add_member_to_design(member_to_add=member_add)

            # Get repeatable design bitstring
            new_design_bits = self.get_repeatable_design_bits(design_conn_array=new_design_CA)
        else:
            new_design_bits = copy.copy(self.design)

        return new_design_bits

    def find_mean_orientation(self, design_CA):
        
        total_orientation = 0
        for member in design_CA:
            node1_pos, node2_pos = self.nodal_positions[member[0]-1], self.nodal_positions[member[1]-1]
            member_orientation = self.calculate_member_orientation(member_node1_pos=node1_pos, member_node2_pos=node2_pos)
            total_orientation += member_orientation
        
        return total_orientation/len(design_CA)


    def calculate_member_orientation(self, member_node1_pos, member_node2_pos):
        num = (member_node2_pos[1] - member_node1_pos[1]) # y2 - y1
        den = (member_node2_pos[0] - member_node1_pos[0]) # x2 - x1
        return np.abs(math.degrees(np.arctan(num/den)))
                