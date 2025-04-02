# -*- coding: utf-8 -*-
"""
Testing heuristic repair operators implemented in Python

@author: roshan94
"""
import sys
import os
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

from repairoperators.metamaterial.adddiagonalmember import AddDiagonalMember
from repairoperators.metamaterial.addmember import AddMember
from repairoperators.metamaterial.improveorientation import ImproveOrientation
from repairoperators.metamaterial.removeintersection import RemoveIntersection

problem = "Artery"
sidenum = 3
sel = 10e-3
target_c_ratio = 0.421

test_design = [1,0,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0]
# Connectivity array -> [[1, 2], [1, 5], [1, 6], [1, 7], [1, 9], [2, 5], [2, 6], [2, 7], [3, 4], [3, 5], [3, 7], [3, 9], [4, 5], [4, 6], [4, 9], [5, 8], [5, 9], [6, 7], [7, 8]]
# For AddMember -> node 8 only has 2 connections
# For RemoveIntersection -> member [1,9] has the most number of intersections 

## Choose operator
op_choice = "Partial Collapsibility"
# choices -> "Partial Collapsibility", "Nodal Properties", "Orientation", "Intersection"

match op_choice:
    case "Partial Collapsibility":
        operator = AddDiagonalMember(sidenum=sidenum, problem=problem, sel=sel)

    case "Nodal Properties":
        operator = AddMember(sidenum=sidenum, problem=problem, sel=sel)

    case "Orientation":
        operator = ImproveOrientation(sidenum=sidenum, problem=problem, sel=sel, target_c_ratio=target_c_ratio)

    case "Intersection":
        operator = RemoveIntersection(sidenum=sidenum, problem=problem, sel=sel)

    case _:
        print("Invalid operator selection")

## Operate on design
operator.set_design(design_bits=test_design)
new_design = operator.evolve()

print("Test design: " + str(test_design))
print("New design: " + str(new_design))
