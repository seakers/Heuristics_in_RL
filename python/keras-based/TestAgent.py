# -*- coding: utf-8 -*-
"""
Testing a trained DQN agent using keras functions instead of tf-agents

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"
import sys

import json

import keras 
from keras import layers

import numpy as np

import csv

import tensorflow as tf

from tqdm import tqdm

from pathlib import Path

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[0]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

from save.ResultSaving import ResultSaver

from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv

import matplotlib.pyplot as plt

import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

run_num = 0

dqn = False # if True -> test DQN network, False -> test PPO actor network
random_states_mode = False

max_steps = 100

## Get savepath from config file
if dqn:
    f = open('keras-based\\dqn\\dqn-config.json')
    alg_dir = "dqn\\"
else:
    f = open('keras-based\\ppo\\ppo-config.json')
    alg_dir = "PPO-H\\"
data = json.load(f)

save_path = data["Savepath"]

new_reward = data["Use new problem formulation"]
include_weights = data["Include weights in state"]

## Load problem parameters from config file
f_prob = open('keras-based\\problem-config.json')
data_prob = json.load(f_prob)

artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

if artery_prob:
    prob_dir = "Artery\\"
else:
    prob_dir = "EqStiff\\"

current_save_path = save_path + prob_dir + alg_dir + "run " + str(run_num) + "\\"

#print(device_lib.list_local_devices())

# model_sel = 0 --> Fibre Stiffness Model
#           = 1 --> Truss Stiffness Model
#           = 2 --> Beam Model        
model_sel = data_prob["Model selection"]

rad = data_prob["Member radius in m"] # in m
sel = data_prob["Lattice side element length in m"] # in m
E_mod = data_prob["Young's Modulus in Pa"] # in Pa
sidenum = data_prob["Lattice number of side nodes"]
nucFac = data_prob["NucFac"]

obj_names = data_prob["Objective names"]
heur_names = data_prob["Heuristic names"] # make sure this is consistent with the order of the heuristic operators in the Java code
heur_abbr = data_prob["Heuristic abbreviations"]
heurs_used = data_prob["Heuristics used"] # for [partial collapsibility, nodal properties, orientation, intersection]
n_heurs_used = heurs_used.count(True)
constr_names = data_prob["Constraint names"]
# if not artery_prob:
#     constr_names = ['FeasibilityViolation','ConnectivityViolation','StiffnessRatioViolation']
# else:
#     constr_names = ['FeasibilityViolation','ConnectivityViolation']
objs_max = data_prob["Objective maximized"]

c_target = data_prob["Target stiffness ratio"]
# c_target = 1
# if artery_prob:
#     c_target = 0.421

feas_c_target_delta = data_prob["Feasible stiffness delta"] # delta about target stiffness ratio defining satisfying designs

render_steps = data_prob["Render steps"]

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
n_actions = 1

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

if dqn:
    file_name = "RL_execution_results_dqnQ_" 
else:
    file_name = "RL_execution_results_ppo_"

if random_states_mode:
    file_name += "rand" + str(run_num)
else:
    file_name += str(run_num)

if artery_prob:
    file_name += "_artery"
else:
    file_name += "_eqstiff"

file_name += ".csv"

# Initialize result saver
result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)

if artery_prob:
    env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
else:
    env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)

# Load Keras Network
nfe_agent_ext = "final" # or "ep" + str(nfe)
if dqn:
    model_filename = "learned_Q_Network"
else:
    model_filename = "learned_actor_network_" + nfe_agent_ext
    seed = 25

if artery_prob:
    model_filename += "_artery"
else:
    model_filename += "_eqstiff"
model_filename += ".h5"

learned_keras_agent = tf.keras.models.load_model(os.path.join(current_save_path, model_filename))

current_state = env.reset()
explored_states = []

step_counter = 0
current_nfe_val = 0

csv_rows = []
fig = plt.figure()

with tqdm(total=max_steps) as pbar:
    while step_counter < max_steps:
        row = {}
        row['Step Number'] = step_counter
        row['State'] = ''.join(map(str,current_state))

        print("step number = " + str(step_counter))

        state_tensor = tf.convert_to_tensor(current_state)
        state_tensor = tf.expand_dims(state_tensor, axis=0)

        if dqn:
            outputs = learned_keras_agent(state_tensor, training=False)
            action = tf.argmax(outputs[0]).numpy()
        else:
            logits = learned_keras_agent(state_tensor, training=False)
            action = tf.squeeze(tf.random.categorical(logits, 1, seed=seed), axis=1).numpy()[0]

        row['Action'] = action    

        traj_start = False
        if step_counter == 0:
            traj_start = True
        
        new_state, reward, done, kw_args = env.step(action=action, nfe_val=current_nfe_val, traj_start=traj_start)

        if new_reward:
            current_nfe_val = kw_args['Current NFE']
            current_truss_des = kw_args['Current truss design']
            true_objs = current_truss_des.get_objs()
            constrs = current_truss_des.get_constrs()
            heurs = current_truss_des.get_heurs()

            new_truss_des = kw_args['New truss design']
            new_true_objs = new_truss_des.get_objs()
            new_constrs = new_truss_des.get_constrs()
            new_heurs = new_truss_des.get_heurs()
        else:
            current_state_str = "".join([str(dec) for dec in current_state])
            if not current_state_str in explored_states:
                current_nfe_val += 1
                explored_states.append(current_state_str)
            # Evaluate design and save objectives, constraints and heuristics
            objs, constrs, heurs, true_objs = env.pyenv.envs[0].get_metamaterial_support().evaluate_design(current_state)
            new_objs, new_constrs, new_heurs, new_true_objs = env.pyenv.envs[0].get_metamaterial_support().evaluate_design(new_state)

        ## Plotting current agent step
        if np.all(constrs == 0):
            prev_color = 'green'
        else:
            prev_color = 'red'

        if np.all(new_constrs == 0):
            new_color = 'green'
        else:
            new_color = 'red'

        # Plot initial state objectives
        if traj_start:
            plt.scatter(true_objs[0], true_objs[1], color=prev_color)
            plt.text(true_objs[0]+0.05*true_objs[0], true_objs[1]+0.05*true_objs[1], str(step_counter))

        # Plot final state objectives
        plt.scatter(new_true_objs[0], new_true_objs[1], color=new_color)
        plt.text(new_true_objs[0]+0.05*new_true_objs[0], new_true_objs[1]+0.05*new_true_objs[1], str(step_counter+1))

        # Plot arrow from initial to final state
        dx = new_true_objs[0] - true_objs[0]
        dy = new_true_objs[1] - true_objs[1]
        plt.arrow(true_objs[0], true_objs[1], dx, dy)

        row['Reward'] = reward

        for i in range(len(obj_names)):
            row[obj_names[i]] = true_objs[i]

        for j in range(len(constr_names)):
            row[constr_names[j]] = constrs[j]

        for k in range(len(heur_names)):
            row[heur_names[k]] = heurs[k]

        if random_states_mode:
            current_state = env.reset()
        else:
            current_state = new_state
        
        csv_rows.append(row)

        step_counter += 1
        pbar.update(1)

field_names = ['Step Number', 'State', 'Action', 'Reward']
field_names.extend(obj_names)
field_names.extend(constr_names)
field_names.extend(heur_names)

plt.xlabel(obj_names[0], fontsize=12)
plt.ylabel(obj_names[1], fontsize=12)
plt.savefig(os.path.join(current_save_path, 'agent_test_steps_' + str(nfe_agent_ext) + '.png', dpi=600))

# write to csv
with open(save_path + file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, lineterminator = '\n')

    writer.writeheader()

    writer.writerows(csv_rows)
