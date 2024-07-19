# -*- coding: utf-8 -*-
"""
Comparing the recommended actions from the TF policy learned from Deep Q Learning Agent on the metamaterial problems
adapted from https://www.tensorflow.org/agents/tutorials/9_c51_tutorial and https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial
and the saved Keras sequential model with the same weights as the trained TF Q-Network

@author: roshan94
"""
import os
import math
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments import tf_py_environment
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import tensorflow as tf

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

run_num = 0

dqn = False # if True -> test DQN networks, False -> test PPO actor networks

random_states_mode = True

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"

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

render_steps = False

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

# find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = (2*n_states) + n_heurs_used # number of actions = (2*number of design variables) (first n_states actions are adding the corresponding member, 
                       # last n_states actions are removing the corresponding member)
max_steps = 100

if artery_prob:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
else:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

## Load Keras Network
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

## Load TF Policy
if dqn:
    policy_dir = os.path.join(save_path, 'DQN', ('run '+str(run_num)), ('policy'+str(run_num)))
else:
    policy_dir = os.path.join(save_path, 'PPO', ('run '+str(run_num)), 'savedPolicy', 'policy')
saved_policy = tf.saved_model.load(policy_dir)

time_step = env.reset()

step_counter = 0

if random_states_mode:

    while step_counter < max_steps:
        print("step number = " + str(step_counter))

        # Evaluate design and save objectives, constraints and heuristics
        objs, constrs, heurs, true_objs = env.pyenv.envs[0].get_metamaterial_support().evaluate_design(time_step.observation.__array__()[0])
        
        tf_action = saved_policy.action(time_step)

        state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        outputs = learned_keras_deepQ_agent(state_tensor, training=False)

        keras_action = tf.argmax(outputs[0]).numpy()
        print("TF action = " + str(tf_action.action.__int__()))
        print("Keras action = " + str(keras_action))
        print("\n")

        next_time_step = env.step(tf_action)

        time_step = env.reset()

        step_counter += 1

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
        action = saved_policy.action(time_step)

        # Take one step
        next_time_step = env.step(action)
        reward = next_time_step.reward._numpy()[0]
        tf_action = action.action.numpy()[0]

        state_tensor = tf.convert_to_tensor(time_step.observation.__array__()[0])
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        outputs = learned_keras_deepQ_agent(state_tensor, training=False)

        #print('Environment is done = ' + str(env.pyenv.envs[0].gym.get_isdone()))
        
        #time_step = env.reset()
        step_counter += 1
        print("step number = " + str(step_counter))

        keras_action = tf.argmax(outputs[0]).numpy()
        print("TF action = " + str(tf_action))
        print("Keras action = " + str(keras_action))
        print("\n")

        time_step = next_time_step




