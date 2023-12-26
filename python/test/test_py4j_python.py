# -*- coding: utf-8 -*-
"""
Test Py4J python script (Both Integer and Boolean Design Decisions)

@author: roshan94
"""

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

# Access java gateway (run after server has been started in Java)
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))

# Access current test design instance from java gateway
current_test_design = gateway.entry_point.getCurrentTestProblemDesign()

# create a design here and pass it to the current test design instance
#new_design = [1,2,3,4,5]
new_design = [True, True, False, False, True]
current_test_design.setCurrentDesign(new_design)

# compute decision total
current_test_design.computeTotal()
dec_total = current_test_design.getDecisionTotal()
print(f"Decision Total = {dec_total}")