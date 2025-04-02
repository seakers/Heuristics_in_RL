# -*- coding: utf-8 -*-
"""
Repair operator class for the nodal properties heuristic for the metamaterial problems
Repurposed from AddMember.java

Action: Add 
a) random member to random node with less than 3 connections OR
b) member connecting two random nodes with less than 3 connections 

@author: roshan94
"""
from repairoperators.metamaterial.baseoperator import BaseOperator
from itertools import permutations
import copy
import numpy as np

class AddMember(BaseOperator):

    def __init__(self, sidenum, problem, sel):
        super(AddMember, self).__init__(sidenum=sidenum, problem=problem, sel=sel)
        self.sidenum = sidenum

    def evolve(self): # Make sure to call set_design() before this method
        super().evolve() # doesn't return anything

        abs_members = self.get_absent_members()

        # Choose member at random to add to design
        low_conection_nodes = self.get_nodes_with_low_connections()

        member_to_be_added = False
        if len(low_conection_nodes) > 0:
            if len(low_conection_nodes) == 1: # Add random member connected to node with low connections
                abs_node_members = [x for x in abs_members if low_conection_nodes[0][0] in x]
                member_add = abs_node_members[np.random.randint(len(abs_node_members))]
                member_to_be_added = True
            elif len(low_conection_nodes) > 0: 
                # Add random member connecting two random nodes with low connections
                rng = np.random.default_rng()
                low_conn_node_pos = [x[0] for x in low_conection_nodes]
                rng.shuffle(low_conn_node_pos)
                member_permutations = list(permutations(low_conn_node_pos, 2))
                for member in member_permutations:
                    if member in abs_members:
                        member_add = member
                        member_to_be_added = True
                        break
                
                # If no candidate member is found, add a random member connecting to a random low connection node
                node_choice = low_conn_node_pos[0]
                abs_node_choice_members = [x for x in abs_members if node_choice in x]
                rng.shuffle(abs_node_choice_members)
                member_add = abs_node_choice_members[0]
                member_to_be_added = True
        
        # member_add = abs_members[random.randint(0, len(abs_members))]

        if member_to_be_added:
            new_design_CA = self.add_member_to_design(member_to_add=member_add)

            # Get repeatable design bitstring
            new_design_bits = self.get_repeatable_design_bits(design_conn_array=new_design_CA)
        else:
            new_design_bits = copy.deepcopy(self.design)

        return new_design_bits
    
    def get_nodes_with_low_connections(self):
        # Output format [[n_i, c_i]...] where n_i is the low connectivity node and c_i is the number of node connections

        design_CA = self.get_design_connectivity_array()

        nodes_list = np.add(np.arange(self.sidenum**2), 1)

        node_connections = np.zeros(len(nodes_list))

        for i in range(len(nodes_list)):
            node = nodes_list[i]
            member_has_node_list = [1 if node in x else 0 for x in design_CA]
            node_connections[i] = np.sum(member_has_node_list)
    
        low_node_connections = [[n+1, int(val)] for n, val in enumerate(node_connections) if val < 3]

        return low_node_connections
    
    
