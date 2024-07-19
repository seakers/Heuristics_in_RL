# -*- coding: utf-8 -*-
"""
Implementing policy learned from Deep Q or PPO Agent on the metamaterial and satellite problems
Adapted from https://www.tensorflow.org/agents/tutorials/9_c51_tutorial and https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial

@author: roshan94
"""
import os
import math
import csv
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments import tf_py_environment
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv

from save.ResultSaving import ResultSaver

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import tensorflow as tf

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

run_num = 0

dqn = False # if True -> test DQN network, False -> test PPO actor network
use_keras_net = False # if True -> use keras Q/Actor network, False -> use saved TF policy
random_states_mode = False

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"

# Load Keras Network
if use_keras_net:
    if dqn:
        model_filename = "learned_Q_Network"
    else:
        model_filename = "learned_actor_Network"

    if artery_prob:
        model_filename += "_artery"
    else:
        model_filename += "_eqstiff"
    model_filename += ".keras"

    if dqn:
        learned_keras_deepQ_agent = tf.keras.models.load_model(os.path.join(save_path, 'DQN', ('run '+str(run_num)), model_filename))
    else:
        learned_keras_deepQ_agent = tf.keras.models.load_model(os.path.join(save_path, 'PPO', ('run '+str(run_num)), model_filename)) 

else:
    # Load TF policy
    if dqn:
        policy_dir = os.path.join(save_path, 'DQN', ('run '+str(run_num)), ('policy'+str(run_num)))
        saved_policy = tf.saved_model.load(policy_dir)
    else:
        policy_dir = os.path.join(save_path, 'PPO', ('run '+str(run_num)), 'savedPolicy', 'policy')
        saved_policy = tf.saved_model.load(policy_dir)

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

if use_keras_net:
    file_name += "_keras"
else:
    file_name += "_tf"
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

gamma = 0.99

render_steps = True

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

# find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = (2*n_states) + n_heurs_used # number of actions = (2*number of design variables) (first n_states actions are adding the corresponding member, 
                       # last n_states actions are removing the corresponding member)
max_steps = 100

# Initialize result saver
result_logger = ResultSaver(save_path=save_path + file_name, operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)

if artery_prob:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
else:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

time_step = env.reset()

step_counter = 0

if random_states_mode:

    csv_rows = []

    while step_counter < max_steps:
        row = {}
        row['Step Number'] = step_counter
        row['State'] = ''.join(map(str,time_step.observation.__array__()[0]))

        print("step number = " + str(step_counter))

        # Evaluate design and save objectives, constraints and heuristics
        objs, constrs, heurs, true_objs = env.pyenv.envs[0].get_metamaterial_support().evaluate_design(time_step.observation.__array__()[0])
        
        if use_keras_net:
            state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            outputs = learned_keras_deepQ_agent(state_tensor, training=False)

            action = tf.argmax(outputs[0]).numpy()
            row['Action'] = action
        else:
            action = saved_policy.action(time_step)
            row['Action'] = action.action.__int__()

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

    while not time_step.is_last():
        # Obtain action (from Q-table if key exists or else choose random action)
        #try:
            #action = np.argmax(learned_deepQ_agent.predict(obs))
        #except:
            #action = np.random.randint(n_actions)

        #print("step number = " + str(step_counter))
        #print('Internal step number = ' + str(env.pyenv.envs[0].gym.get_step_counter()))
        
        current_observation = time_step.observation._numpy()[0]

        if use_keras_net:
            state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            outputs = learned_keras_deepQ_agent(state_tensor, training=False)

            action = tf.argmax(outputs[0]).numpy()
            current_action = action
        else:
            action = saved_policy.action(time_step)
            current_action = action.action.numpy()[0]

        # Take one step
        next_time_step = env.step(action)
        reward = next_time_step.reward._numpy()[0]

        #print('Environment is done = ' + str(env.pyenv.envs[0].gym.get_isdone()))

        result_logger.save_to_logger(step_number=step_counter, action=current_action, prev_obs=current_observation, reward=reward)
        
        #time_step = env.reset()
        step_counter += 1
        print("step number = " + str(step_counter))

        time_step = next_time_step
        
        #next_obs, reward, done, info = env.step(action)

        # Check if done or else continue from new observation
        #if not done:
            #obs = next_obs

    # Save results to the file
    result_logger.save_to_csv()