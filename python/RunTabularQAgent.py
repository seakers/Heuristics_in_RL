# -*- coding: utf-8 -*-
"""
Implementing learned Tabular Q-Agent on the metamaterial and satellite problems
Q-learning agent taken from tutorial example in https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/

@author: roshan94
"""
import math
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
import numpy as np
import pickle
from tqdm import tqdm

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"
file_name = "RL_execution_results_tabQ"

if artery_prob:
    file_name += "_artery"
else:
    file_name += "_eqstiff"
file_name += ".csv"

# model_sel = 0 --> Fibre Stiffness Model
#           = 1 --> Truss Stiffness Model
#           = 2 --> Beam Model
model_sel = 1

rad = 250e-6 # in m
sel = 10e-3 # in m
E_mod = 1.8162e6 # in Pa
sidenum = 5
nucFac = 3

obj_names = ['TrueObjective1','TrueObjective2']
heur_names = ['PartialCollapsibilityViolation','NodalPropertiesViolation','OrientationViolation','IntersectionViolation'] # make sure this is consistent with the order of the heuristic operators in the Java code
heurs_used = [False, False, False, False] # for [partial collapsibility, nodal properties, orientation, intersection]
n_heurs_used = heurs_used.count(True)
constr_names = ['FeasibilityViolation','ConnectivityViolation']
if not artery_prob:
    constr_names.append('StiffnessRatioViolation')

c_target = 1
if artery_prob:
    c_target = 0.421

render_steps = False

# find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = (2*n_states) + n_heurs_used # number of actions = (2*number of design variables) (first n_states actions are adding the corresponding member, 
                       # last n_states actions are removing the corresponding member)
max_steps = 50

if artery_prob:
    env = ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
else:
    env = EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)

## Load agent
pickle_file_name = "Q-table" 
if artery_prob:
    pickle_file_name += "_artery"
else:
    pickle_file_name += "_eqstiff"
pickle_file_name += ".pkl"

q_table_file = open(save_path + pickle_file_name,'rb')
learned_q_table = pickle.load(q_table_file)
obs, info = env.reset()

for step_current in tqdm(range(max_steps)):
    # Obtain action (from Q-table if key exists or else choose random action)
    try:
        action = np.argmax(learned_q_table[tuple(obs)])
    except:
        action = np.random.randint(n_actions)

    # Take one step
    next_obs, reward, done, info = env.step(action)

    # Check if done or else continue from new observation
    if not done:
        obs = next_obs

# Save results to the file
env.get_support_result_logger().save_to_csv()