# -*- coding: utf-8 -*-
"""
Repair operator class for the partial collapsibility heuristic for the metamaterial problems
Repurposed from AddDiagonalMember.java

Action: Add random diagonal member to design

@author: roshan94
"""
from repairoperators.metamaterial.baseoperator import BaseOperator
from itertools import compress
import numpy as np
import copy

class AddDiagonalMember(BaseOperator):

    def __init__(self, sidenum, problem, sel):
        super(AddDiagonalMember, self).__init__(sidenum=sidenum, problem=problem, sel=sel)
        self.sidenum = sidenum

    def evolve(self): # Make sure to call set_design() before this method
        super().evolve() # doesn't return anything

        abs_diag_members = self.get_absent_diagonal_members()

        member_to_be_added = False
        if len(abs_diag_members) > 0:
            # Choose member at random to add to design
            diag_member_add = abs_diag_members[np.random.randint(0, len(abs_diag_members))]
            member_to_be_added = True

        if member_to_be_added:
            new_design_CA = self.add_member_to_design(member_to_add=diag_member_add)

            # Get repeatable design bitstring
            new_design_bits = self.get_repeatable_design_bits(design_conn_array=new_design_CA)
        else:
            new_design_bits = copy.deepcopy(self.design)

        return new_design_bits

    def get_absent_diagonal_members(self):

        # Get all absent members in design
        abs_members = self.get_absent_members()

        # Pare down vertical (same x-position) and horizontal members (same y-position)
        if len(abs_members) > 0:
            non_horizontal_members_ind = [False if (self.nodal_positions[x[0]-1,1] == self.nodal_positions[x[1]-1,1]) else True for x in abs_members]
            non_horizontal_members = list(compress(abs_members, non_horizontal_members_ind))

            non_vertical_members_ind = [False if (self.nodal_positions[x[0]-1,0] == self.nodal_positions[x[1]-1,0]) else True for x in non_horizontal_members]
            diag_members = list(compress(non_horizontal_members, non_vertical_members_ind))
        else:
            diag_members = []

        return diag_members
    