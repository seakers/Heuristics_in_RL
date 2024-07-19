# -*- coding: utf-8 -*-
"""
Test to create dataset generation function to use for agent training instead of a replay buffer sample

@author: roshan94
"""
import math
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv

import tensorflow as tf

from tf_agents.environments.gym_wrapper import GymWrapper
#from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import tf_py_environment

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

from tf_agents.trajectories import trajectory

from tf_agents.utils import common

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"

#print(device_lib.list_local_devices())

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
heur_abbr = ['p','n','o','i']
heurs_used = [False, False, False, False] # for [partial collapsibility, nodal properties, orientation, intersection]
n_heurs_used = heurs_used.count(True)
constr_names = ['FeasibilityViolation','ConnectivityViolation']
if not artery_prob:
    constr_names.append('StiffnessRatioViolation')

c_target = 1
if artery_prob:
    c_target = 0.421

gamma = 0.99
num_iterations = 100 
max_steps = num_iterations

render_steps = False

file_name = "RL_training_designs_dqn"
if artery_prob:
    file_name += "_artery_"
else:
    file_name += "_eqstiff_"

if n_heurs_used > 0:
    for i in range(len(heur_abbr)):
        if heurs_used[i]:
            file_name += heur_abbr[i]
    
file_name += "0.csv"

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
#max_steps = 1000

## DQN Hyperparameters
num_iterations = 100 
max_steps = num_iterations

initial_collect_steps = 42 # number of steps 
collect_steps_per_iteration = 1  
replay_buffer_capacity = 10000 

fc_layer_params = (50, 10)
dropout_layer_params = (0.10, 0.02)

batch_size = 32
learning_rate = 1e-3  
gamma = 0.99
log_interval = 50  
prob_eps_greedy = 0.05

num_eval_episodes = 10  
eval_interval = 10 

n_step_update = 10

if artery_prob:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
    #env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
else:
    env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
    #env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

## Define Q-network
q_net = q_network.QNetwork(
    input_tensor_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    fc_layer_params=fc_layer_params,
    q_layer_activation_fn=tf.keras.activations.softmax,
    dropout_layer_params=dropout_layer_params)

global_step = tf.compat.v1.train.get_or_create_global_step()

## Define DQN Agent
## Define exponential decay of epsilon
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=initial_collect_steps, decay_rate=0.96)

## Define optimizer
#optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.96, momentum=0.001)

agent = dqn_agent.DqnAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    n_step_update=n_step_update,
    gamma=gamma,
    td_errors_loss_fn=common.element_wise_squared_loss,
    #train_step_counter=train_step_counter,
    train_step_counter=global_step,
    epsilon_greedy=prob_eps_greedy)

agent.initialize()

def generate_trajectory(env, num_steps, agent):
    observations = []
    rewards = []
    actions = []
    next_observations = []
    discounts = []
    step_types = []
    next_step_types = []

    step = 0
    time_step = env.reset()
    while(step < num_steps):
        observations.append(time_step.observation._numpy()[0])
        action_step = agent.collect_policy.action(time_step)
        actions.append(action_step.action._numpy()[0])
        next_time_step = env.step(action_step.action)
        rewards.append(next_time_step.reward._numpy()[0])
        next_observations.append(next_time_step.observation._numpy()[0])
        discounts.append(next_time_step.discount._numpy()[0])
        step_types.append(time_step.step_type._numpy()[0])
        next_step_types.append(next_time_step.step_type._numpy()[0])

        step += 1
    
    return trajectory.Trajectory(observation=tf.expand_dims(tf.convert_to_tensor(observations),0),
                                action=tf.expand_dims(tf.convert_to_tensor(actions)),
                                reward=tf.expand_dims(tf.convert_to_tensor(rewards)),
                                policy_info=action_step.info,
                                discount=tf.expand_dims(tf.convert_to_tensor(discounts)),
                                step_type=tf.expand_dims(tf.convert_to_tensor(step_types)),
                                next_step_type=tf.expand_dims(tf.convert_to_tensor(next_step_types)))

traj = generate_trajectory(env, 10, agent)
