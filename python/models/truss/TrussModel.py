# -*- coding: utf-8 -*-
"""
Truss model class for design evaluation

@author: gabeapaza, modified by roshan94
"""
import time
from models.truss.vol.TrussVolumeFraction import TrussVolumeFraction
from models.truss.stiffness.TrussStiffness import TrussStiffness


class TrussModel:

    def __init__(self, sidenum):
        self.sidenum = sidenum

    def evaluate(self, design_array, y_modulus, member_radii, side_length):

        # 1. Calculate the volume fraction
        curr_time = time.time()
        vf_client = TrussVolumeFraction(self.sidenum, design_array, side_length=side_length)
        design_conn_array = vf_client.design_conn_array
        volume_fraction, feasibility_constraint, interaction_list = vf_client.evaluate2(member_radii, side_length)
        # volume_fraction, feasibility_constraint, interaction_list = 0, 0, 0

        # print("Time taken for volume fraction: ", time.time() - curr_time)

        # 2. Calculate the stiffness
        curr_time = time.time()
        stiffness_tensor = TrussStiffness.evaluate(design_conn_array, self.sidenum, side_length, member_radii, y_modulus)
        # print("Time taken for stiffness: ", time.time() - curr_time)

        # print("Volume fraction: ", volume_fraction)
        # print("Vertical stiffness: ", v_stiff)
        # print("Horizontal stiffness: ", h_stiff)
        # print("Stiffness ratio: ", stiff_ratio)
        # print("Feasibility constraint: ", feasibility_constraint)

        return stiffness_tensor, volume_fraction, feasibility_constraint

    def evaluate_decomp(self, design_array, y_modulus, member_radii, side_length):

        # 1. Calculate the volume fraction
        curr_time = time.time()
        vf_client = TrussVolumeFraction(self.sidenum, design_array, side_length=side_length)
        design_conn_array = vf_client.design_conn_array
        volume_fraction, feasibility_constraint, interaction_list = vf_client.evaluate2(member_radii, side_length)
        new_nodes, new_design_conn_array = vf_client.get_intersections()
        # volume_fraction, feasibility_constraint, interaction_list = 0, 0, 0

        # print("Time taken for volume fraction: ", time.time() - curr_time)

        # 2. Calculate the stiffness
        curr_time = time.time()
        stiffness_tensor = TrussStiffness.evaluate_decomp(new_design_conn_array, self.sidenum, side_length, member_radii, y_modulus, new_nodes)
        # print("Time taken for stiffness: ", time.time() - curr_time)

        # print("Volume fraction: ", volume_fraction)
        # print("Vertical stiffness: ", v_stiff)
        # print("Horizontal stiffness: ", h_stiff)
        # print("Stiffness ratio: ", stiff_ratio)
        # print("Feasibility constraint: ", feasibility_constraint)

        return stiffness_tensor, volume_fraction, feasibility_constraint

# if __name__ == '__main__':

#     sidenum = config.sidenum
#     # design_array = [1 for x in range(config.num_vars)]
#     design_str = '111010000010100010100000101010'
#     design_array = [int(bit) for bit in design_str]


#     y_modulus = 18162.0
#     member_radii = 0.1
#     side_length = 3

#     truss_model = TrussModel(sidenum)
#     curr_time = time.time()
#     truss_model.evaluate(design_array, y_modulus, member_radii, side_length)
#     print('\n----------')
#     truss_model.evaluate_decomp(design_array, y_modulus, member_radii, side_length)
#     print("Time taken: ", time.time() - curr_time)
