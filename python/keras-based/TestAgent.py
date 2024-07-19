# -*- coding: utf-8 -*-
"""
Testing a trained DQN agent using keras functions instead of tf-agents

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"

import json

import keras 
from keras import layers

import csv

import tensorflow as tf

from tqdm import tqdm

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
    f = open('.\\keras-based\\dqn\\dqn-config.json')
else:
    f = open('.\\keras-based\\ppo\\ppo-config.json')
data = json.load(f)

save_path = data["Savepath"]

## Load problem parameters from config file
f_prob = open('.\\keras-based\\problem-config.json')
data_prob = json.load(f_prob)

artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

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

c_target = data_prob["Target stiffness ratio"]
# c_target = 1
# if artery_prob:
#     c_target = 0.421

render_steps = data_prob["Render steps"]

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)

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
result_logger = ResultSaver(save_path=os.path.join(save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)

if artery_prob:
    env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
else:
    env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)

# Load Keras Network
if dqn:
    model_filename = "learned_Q_Network"
else:
    model_filename = "learned_actor_Network"
    seed = 25

if artery_prob:
    model_filename += "_artery"
else:
    model_filename += "_eqstiff"
model_filename += ".h5"

learned_keras_agent = tf.keras.models.load_model(os.path.join(save_path, ('run '+str(run_num)), model_filename))

time_step = env.reset()

step_counter = 0

if random_states_mode:

    csv_rows = []

    with tqdm(total=max_steps) as pbar:
        while step_counter < max_steps:
            row = {}
            row['Step Number'] = step_counter
            row['State'] = ''.join(map(str,time_step.observation.__array__()[0]))

            print("step number = " + str(step_counter))

            # Evaluate design and save objectives, constraints and heuristics
            objs, constrs, heurs, true_objs = env.pyenv.envs[0].get_metamaterial_support().evaluate_design(time_step.observation.__array__()[0])

            state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
            state_tensor = tf.expand_dims(state_tensor, axis=0)

            if dqn:
                outputs = learned_keras_agent(state_tensor, training=False)
                action = tf.argmax(outputs[0]).numpy()
            else:
                logits = learned_keras_agent(state_tensor, training=False)
                action = tf.squeeze(tf.random.categorical(logits, 1, seed=seed), axis=1).numpy()[0]

            row['Action'] = action    
            
            next_time_step = env.step(action)
            row['Reward'] = next_time_step.reward.__float__()

            for i in range(len(obj_names)):
                row[obj_names[i]] = true_objs[i]

            for j in range(len(constr_names)):
                row[constr_names[j]] = constrs[j]

            for k in range(len(heur_names)):
                row[heur_names[k]] = heurs[k]

            time_step = env.reset()

            csv_rows.append(row)

            step_counter += 1
            pbar.update(1)

    field_names = ['Step Number', 'State', 'Action', 'Reward']
    field_names.extend(obj_names)
    field_names.extend(constr_names)
    field_names.extend(heur_names)

    # write to csv
    with open(save_path + file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, lineterminator = '\n')

        writer.writeheader()

        writer.writerows(csv_rows)

else:

    with tqdm(total=max_steps) as pbar:
        while not time_step.is_last():
            # Obtain action (from Q-table if key exists or else choose random action)
            #try:
                #action = np.argmax(learned_deepQ_agent.predict(obs))
            #except:
                #action = np.random.randint(n_actions)

            #print("step number = " + str(step_counter))
            #print('Internal step number = ' + str(env.pyenv.envs[0].gym.get_step_counter()))
            
            current_observation = time_step.observation._numpy()[0]

            state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
            state_tensor = tf.expand_dims(state_tensor, axis=0)

            if dqn:
                outputs = learned_keras_agent(state_tensor, training=False)
                action = tf.argmax(outputs[0]).numpy()
            else:
                logits = learned_keras_agent(state_tensor, training=False)
                action = tf.squeeze(tf.random.categorical(logits, 1, seed=seed), axis=1).numpy()[0]

            current_action = action

            # Take one step
            next_time_step = env.step(action)
            reward = next_time_step.reward._numpy()[0]

            #print('Environment is done = ' + str(env.pyenv.envs[0].gym.get_isdone()))

            result_logger.save_to_logger(step_number=step_counter, action=current_action, prev_obs=current_observation, reward=reward)
            
            #time_step = env.reset()
            step_counter += 1
            print("step number = " + str(step_counter))

            time_step = next_time_step
            pbar.update(1)
            
            # Check if done or else continue from new observation
            #if not done:
                #obs = next_obs

    # Save results to the file
    result_logger.save_to_csv()

