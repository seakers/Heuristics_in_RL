# -*- coding: utf-8 -*-
"""
Training and saving a PPO agent using keras functions instead of tf-agents
Reference: https://keras.io/examples/rl/ppo_cartpole/

Changes to make:
1. Instead of sampling from buffer for each training iteration, either 
    a. generate a new trajectory and sample minibatch for training in each iteration (- Done) or 
    b. empty buffer at the end of each iteration and repopulate with new trajectories after each iteration
2. Update the weights of the actor and critic after each iteration if not already done (check if KL-divergence(pi_old, pi_new) = 0 at the beginning of training in each iteration) (- Done)
3. Instead of sampling minibatch trajectories from each generated trajectory, sample transitions from the combined set of trajectories 
(although difference in performance is worth investigating) (- Done)

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"

# Add path for parent directory with environment classes
import sys
import json
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)
from itertools import chain

import tensorflow as tf
import keras 
from keras import layers

from tqdm import tqdm

from ppo_buffer import Buffer
from envs.metamaterial.ArteryProblemEnv import ArteryProblemEnv
from envs.metamaterial.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from save.ResultSaving import ResultSaver

import matplotlib.pyplot as plt

import numpy as np
import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import time

## Setup and train parameters from config file
start_time = time.time()
f_ppo = open('.\\keras-based\\ppo\\ppo-config.json')
data_ppo = json.load(f_ppo)

n_runs = data_ppo["Number of runs"]

gamma = data_ppo["Value discount (gamma)"] # discount factor
original_max_train_episodes = data_ppo["Number of training episodes"] # number of training episodes

max_steps = data_ppo["Maximum steps in training episode (for train environment termination)"] # no termination in training environment
max_eval_steps = data_ppo["Maximum steps in evaluation episode (for evaluation environment termination)"] # termination in evaluation environment
max_eval_episodes = data_ppo["Number of evaluation episodes"] # number of episodes per evaluation of the actor

max_unique_nfe_run = data_ppo["Maximum unique NFE"]

use_buffer = data_ppo["Buffer used"]

eval_interval = data_ppo["Episode interval for evaluation"] # After how many episodes is the actor being evaluated
new_reward = data_ppo["Use new problem formulation"]
include_weights = data_ppo["Include weights in state"]

sample_minibatch = data_ppo["Sample minibatch"] # Whether to sample minibatch or use the entire set of generated trajectories

initial_collect_trajs = data_ppo["Initial number of stored trajectories"] # number of trajectories in the driver to populate replay buffer before beginning training (only used if replay buffer is used)
trajectory_collect_steps = data_ppo["Number of steps in a collected trajectory"] # number of steps in each trajectory
episode_training_trajs = data_ppo["Number of trajectories used for training per episode"] # number of trajectories sampled in each iteration to train the actor and critic
minibatch_steps = data_ppo["Number of steps in a minibatch"]
replay_buffer_capacity = data_ppo["Replay buffer capacity"] # maximum number of trajectories that can be stored in the buffer

if minibatch_steps > trajectory_collect_steps:
    print("Number of steps in a minibatch is greater than the number of collected steps, reduce the minibatch steps")
    sys.exit(0)

if not sample_minibatch:
    minibatch_steps = trajectory_collect_steps

## NOTE: Total number of designs used for training in each run = episode_training_trajs*minibatch_steps*max_train_episodes

advantage_norm = data_ppo["Normalize advantages"] # whether to normalize advantages for training
discrete_actions = data_ppo["Discrete actions"]
lam = data_ppo["Advantage discount (lambda)"] # advantage discount factor

actor_fc_layer_params = data_ppo["Actor network layer units"]
actor_dropout_layer_params = data_ppo["Actor network dropout probabilities"]

critic_fc_layer_params = data_ppo["Critic network layer units"]
critic_dropout_layer_params = data_ppo["Critic network dropout probabilities"]

## NOTE: At least clipping or adaptive KL penalty must be used
use_clipping = data_ppo["Use clipping loss"] # Use PPO clipping term in actor loss
clip_ratio = data_ppo["Clipping ratio threshold"]

use_adaptive_kl_penalty = data_ppo["Use Adaptive KL penalty loss"] # Use adaptive KL-divergence based penalty term in actor loss
kl_targ = data_ppo["KL target"]
beta = data_ppo["Adaptive KL coefficient (beta)"]

use_entropy_bonus = data_ppo["Use entropy loss bonus"] # Use additional entropy of actor distribution term in actor loss
ent_coeff = data_ppo["Entropy coefficient"] # coefficient for the entropy bonus term in actor loss
use_early_stopping = data_ppo["Use early stopping for actor training"] # Stop training epoch early if current KL-divergence crosses 1.5 * kl_targ

train_policy_iterations = data_ppo["Number of actor training iterations"]
train_value_iterations = data_ppo["Number of critic training iterations"]

initial_actor_learning_rate = data_ppo["Initial actor training learning rate"]
initial_critic_learning_rate = data_ppo["Initial critic training learning_rate"]  
decay_rate = data_ppo["Learning rate decay rate"]
decay_steps_actor = data_ppo["Learning rate decay steps (actor)"]
decay_steps_critic = data_ppo["Learning rate decay steps (critic)"]
#decay_steps_actor = max_train_episodes*train_policy_iterations 
#decay_steps_critic = max_train_episodes*train_value_iterations 
rho = data_ppo["RMSprop optimizer rho"]
momentum = data_ppo["RMSprop optimizer momentum"]

compute_periodic_returns = data_ppo["Compute periodic returns"]
use_continuous_minibatch = data_ppo["Continuous minibatch"] # use continuous trajectory slices to generate minibatch for training or random transitions from all trajectories
network_save_intervals = data_ppo["Episode interval to save actor and critic networks"]

save_path = data_ppo["Savepath"]

if network_save_intervals > original_max_train_episodes:
    print("Episode interval to save networks is greater than number of training episodes")
    sys.exit(0)

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

n_episodes_per_fig = 4 # used for plotting returns and losses 
linestyles = ['solid','dotted','dashed','dashdot']

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

## Define method to compute average return for DQN training
def compute_avg_return(environment, actor, num_steps=100, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        start_obs = environment.reset()
        if new_reward:
            if include_weights:
                #start_obs_list = sum(start_obs.values(), [])
                start_obs_list = list(chain(*start_obs.values()))
                state = np.array(start_obs_list)
            else:
                state = start_obs
        else:
            state = start_obs

        episode_return = 0.0

        eval_step = 0

        while eval_step < num_steps:
            # Predict action from current Q-network
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            action_logits = actor(state_tensor, training=False)

            # Take best action
            action_probs = tf.nn.softmax(action_logits)
            action = tf.argmax(tf.squeeze(action_probs)).numpy()

            traj_start = False
            if eval_step == 0:
                traj_start = True
            next_obs, reward, done, kw_args = environment.step(action=action, nfe_val=eval_step, include_prev_des=traj_start)
            episode_return += (gamma**eval_step)*reward

            if done:
                break
            else:
                if new_reward:
                    if include_weights:
                        #next_obs_list = sum(next_obs.values(), [])
                        next_obs_list = list(chain(*next_obs.values()))
                        state = np.array(next_obs_list)
                    else:
                        state = next_obs
                else:
                    state = next_obs
                eval_step += 1

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return

## Method to generate actor network with given number of hidden layer neurons and dropout probabilities
def create_actor(num_states, hidden_layer_params, hidden_layer_dropout_params, num_action_vals, num_actions):
    actor = keras.Sequential()
    actor.add(layers.Dense(num_states, activation='relu'))
    for n_units, dropout_prob in zip(hidden_layer_params, hidden_layer_dropout_params):
        actor.add(layers.Dense(n_units, activation='relu'))
        actor.add(layers.Dropout(dropout_prob))
    if discrete_actions:
        actor.add(layers.Dense(num_actions*num_action_vals, activation=None)) 
    else:
        actor.add(layers.Dense(2 * num_actions, activation=None)) 
    return actor

## Method to generate critic network with given number of hidden layer neurons and dropout probabilities
def create_critic(num_states, hidden_layer_params, hidden_layer_dropout_params):
    critic = keras.Sequential()
    critic.add(layers.Dense(num_states, activation='relu'))
    for n_units, dropout_prob in zip(hidden_layer_params, hidden_layer_dropout_params):
        critic.add(layers.Dense(n_units, activation='relu'))
        critic.add(layers.Dropout(dropout_prob))
    critic.add(layers.Dense(1, activation=None)) 
    return critic

## Method to compute the log-probabilities of the actor network logit for the selected action of the batch
def log_probabilities(logits, a, num_actions):
    log_probs_all = tf.nn.log_softmax(logits)
    log_prob_action = tf.reduce_sum(tf.math.multiply(tf.one_hot(a, num_actions), log_probs_all), axis=1)
    return log_prob_action

seed = 10

## Generate trajectories for training
def generate_trajectories(num_states, num_actions, num_action_vals, n_trajs, collect_steps, actor, current_nfe):
    
    traj_truss_des = []
    if new_reward:
        traj_states = np.zeros((n_trajs, collect_steps, num_states), dtype=np.float32) # The explicit datatype is for the additional objective weights states, design is later converted to int32 for py4j compatibility
    else:
        traj_states = np.zeros((n_trajs, collect_steps, num_states), dtype=np.int32) # The explicit datatype is for py4j compatibility with the Java methods
    traj_actions = np.zeros((n_trajs, collect_steps, num_actions))
    traj_rewards = np.zeros((n_trajs, collect_steps))
    traj_dones = np.zeros((n_trajs, collect_steps))

    if discrete_actions:
        traj_policy_logits = np.zeros((n_trajs, collect_steps, num_action_vals)) # for discrete actions, the logits are directly converted to probabilities
    else:
        traj_policy_logits = np.zeros((n_trajs, collect_steps, 2 * num_action_vals)) # for continuous actions, the probability distribution for an action 
        # is taken as gaussian with two logits representing the mean and variance respectively

    # Genarating and storing trjectories (buffer used to compute advantages and values)
    for traj_count in range(n_trajs):
        print('Generating trajectory')

        observation = train_env.reset()
        if new_reward:
            if include_weights:
                #observation = sum(observation.values(), [])
                observation = list(chain(*observation.values()))
        state = np.array(observation)    
        
        current_traj_des = []

        for step in range(collect_steps):

            traj_states[traj_count, step, :] = state

            # Sample action from the current actor
            logits, action = sample_action(actor, state)
            traj_policy_logits[traj_count, step, :] = logits
            traj_actions[traj_count, step, :] = action

            # Apply action and save to the replay buffer
            traj_start = False
            if step == 0:
                traj_start = True

            next_obs, reward, done, add_arg = train_env.step(action=action, nfe_val=current_nfe, include_prev_des=traj_start)
            if new_reward:
                if include_weights:
                    #next_obs = sum(next_obs.values(), [])
                    next_obs = list(chain(*next_obs.values()))

                current_nfe = add_arg['Current NFE']
                current_truss_des = add_arg['Current truss design']
                new_truss_des = add_arg['New truss design']
            next_state = np.array(next_obs)
            traj_rewards[traj_count, step] = reward
            traj_dones[traj_count, step] = done

            if new_reward:
                if not current_truss_des == None: # To match truss designs with the saved states, initial current_truss_des is saved and final new_truss_des is skipped
                    if new_truss_des.get_objs() == [] or current_truss_des.get_objs == []:
                        print('New design not evaluated')
                    current_traj_des.append(current_truss_des)
                if step < (collect_steps-1):
                    current_traj_des.append(new_truss_des)
            
            #buffer.store_to_trajectory(state, action, reward, logits)

            if done:
                traj_dones[traj_count, step+1:] = np.ones(len(collect_steps)-step)
                break

            state = next_state

        if new_reward:
            traj_truss_des.append(current_traj_des)

    return traj_states, traj_actions, traj_rewards, traj_dones, traj_policy_logits, traj_truss_des, current_nfe

## Method to store trajectories to buffer
def store_into_buffer(buffer, n_trajs, collect_steps, trajectory_states, trajectory_truss_designs, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits):
    for traj_count in range(n_trajs):
        print('Storing trajectory ' + str(traj_count) + ' into buffer')
        for step in range(collect_steps):
            buffer.store_to_trajectory(trajectory_states[traj_count, step, :], trajectory_truss_designs[traj_count][step], trajectory_actions[traj_count, step, :], trajectory_rewards[traj_count, step], \
                    trajectory_dones[traj_count, step], trajectory_policy_logits[traj_count, step, :])
        buffer.store_to_buffer()

## Method to sample action from actor network for the current observation
#@tf.function
def sample_action(actor, observation):
    # if new_reward:
    #     observation = sum(observation.values(), [])
    observation_tensor = tf.convert_to_tensor(observation)
    observation_tensor = tf.expand_dims(observation_tensor, axis=0)
    logits = actor(observation_tensor, training=False)
    action = tf.squeeze(tf.random.categorical(tf.nn.log_softmax(logits), 1, seed=seed), axis=1).numpy()[0]
    if new_reward and include_weights:
        if action >= (len(observation) - (len(obj_names)-1)): # Assuming the weights of all but last objective are in the state
            print('Invalid action')
    else:
        if action >= len(observation): # Assuming the weights of all but last objective are in the state
            print('Invalid action')
    logits_array = logits.numpy()[0]
    return logits_array, action

## Define method to compute cumulative discounted rewards and advantages
def discounted_cumulative_sums(rewards, compute_advantage=True):
    # compute_advantage = True -> discount value is lam*gamma^t
    # compute_advantage = False -> discount value is gamma^t (i.e. value is being computed)

    discount_array = np.ones(len(rewards))
    for i in range(len(rewards)):
        if compute_advantage:
            discount_array[i] = (lam*gamma)**i
        else:
            discount_array[i] = gamma**i
    
    sum = 0

    # Ensure that first value in discount array is 1
    for (current_reward, current_discount) in zip(rewards, discount_array):
        sum += current_discount*current_reward

    return sum

## Method to compute advantages and returns (true value from generated trajectory) for the chosen trajectory
def compute_advantages_and_returns(indices, full_rewards, full_values, ad_norm):
    # gamma and lam are defined globally

    full_deltas = np.zeros((len(full_rewards)-1)) # size of actual trajectory
    for i in range(len(full_rewards)-1):
        full_deltas[i] = full_rewards[i] + gamma*full_values[i+1] - full_values[i]

    full_advantages = np.zeros(len(full_rewards)-1)
    full_returns = np.zeros(len(full_rewards)-1)
    for ind in range(len(full_rewards)-1):
        full_advantages[ind] = discounted_cumulative_sums(full_deltas[ind:], compute_advantage=True)
        full_returns[ind] = discounted_cumulative_sums(full_rewards[ind:], compute_advantage=False)

    if ad_norm:
        full_advantage_mean = np.mean(full_advantages)
        full_advantage_std = np.std(full_advantages)
        full_advantages = np.divide(np.subtract(full_advantages, full_advantage_mean), full_advantage_std)

        #full_advantages_num = np.subtract(full_advantages, np.min(full_advantages))
        #full_advantages_den = np.max(full_advantages) - np.min(full_advantages)
        #full_advantages = np.divide(full_advantages_num, full_advantages_den)

    #self.trajectory_start_index = self.current_end_position

    return full_returns[indices], full_advantages[indices]

## Train the policy network (actor) by maximizing PPO objective (possibly including clip and/or adaptive KL penalty coefficients and entropy regularization)
#@tf.function
def train_actor(observation_samples, action_samples, logits_samples, advantage_samples, target_kl, entropy_coeff):
    action_samples_array = tf.squeeze(action_samples).numpy()

    with tf.GradientTape() as tape:
        actor_predictions = actor_net(observation_samples, training=False)
        probabilities_old = tf.nn.softmax(logits_samples)
        probabilities_new = tf.nn.softmax(actor_predictions)
        
        log_prob_old = log_probabilities(logits_samples, action_samples_array, n_action_vals)
        log_prob_new = log_probabilities(actor_predictions, action_samples_array, n_action_vals)
        
        ratio = tf.exp(log_prob_new - log_prob_old)
        policy_loss_clipping = 0
        adaptive_kl_penalty = 0
        entropy = 0
        if use_clipping:
            clipped_ratio = tf.where(ratio > (1 + clip_ratio), (1 + clip_ratio), ratio) # clipping the coefficient of the advantage in the loss function if ratio > (1 + clip_ratio)
            clipped_ratio = tf.where(clipped_ratio < (1 - clip_ratio), (1 - clip_ratio), clipped_ratio) # clipping the coefficient of the advantage in the loss function if ratio < (1 - clip_ratio)
            policy_loss_clipping = tf.reduce_mean(tf.minimum(tf.math.multiply(ratio, advantage_samples), tf.math.multiply(clipped_ratio, advantage_samples)))
        
        kl_divergence = tf.reduce_sum(tf.math.multiply(probabilities_old, tf.math.log(tf.math.divide(probabilities_old, probabilities_new))), axis=1)
        kl_mean = tf.reduce_mean(kl_divergence)
        if use_adaptive_kl_penalty:    
            adaptive_kl_penalty = beta * kl_mean

            if kl_mean < (target_kl/1.5):
                beta = beta/2
            elif kl_mean > (target_kl * 1.5):
                beta = beta * 2

        # entropy bonus is added to discourage premature convergence of the policy and allow for more exploration (Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, 
        # Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." In International conference on machine learning, 
        # pp. 1928-1937. PMLR, 2016.)
        if use_entropy_bonus: 
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(probabilities_new, tf.math.log(probabilities_new)), axis=1)) 

        policy_loss = -policy_loss_clipping + adaptive_kl_penalty - entropy_coeff*entropy
        print('Clipping loss: ' + str(policy_loss_clipping))
        print('Entropy bonus: ' + str(entropy))
    
    policy_grads = tape.gradient(policy_loss, actor_net.trainable_variables)
    # Check if there are any NaNs in the gradients tensor
    nan_grads = False
    for layer in range(len(policy_grads)):
        if np.any(tf.math.is_nan(policy_grads[layer]).numpy()):
            nan_grads = True
            break

    if nan_grads:
        print('NaN actor gradients')
        
    print('Policy optimizer learning rate: ' + str(policy_optimizer.learning_rate.numpy()))
    policy_optimizer.apply_gradients(zip(policy_grads, actor_net.trainable_variables))

    return policy_loss, kl_mean

## Train the value network (critic) using MSE
#@tf.function
def train_critic(observation_samples, return_samples):
    with tf.GradientTape() as tape:
        critic_predictions = critic_net(observation_samples, training=False)
        value_loss = tf.reduce_mean(tf.math.pow(tf.math.subtract(return_samples, critic_predictions), 2))
    value_grads = tape.gradient(value_loss, critic_net.trainable_variables)
    # Check if there are any NaNs in the gradients tensor
    nan_grads = False
    for layer in range(len(value_grads)):
        if np.any(tf.math.is_nan(value_grads[layer]).numpy()):
            nan_grads = True
            break

    if nan_grads:
        print('NaN critic gradients')

    print('Value optimizer learning rate: ' + str(value_optimizer.learning_rate.numpy()))
    value_optimizer.apply_gradients(zip(value_grads, critic_net.trainable_variables))

    return value_loss

########################################## Operation ##########################################

if compute_periodic_returns:
    returns_runs = {}
    returns_steps_runs = {}

critic_losses_runs = {}
actor_losses_runs = {}
actor_end_steps_runs = {}

critic_loss_steps_runs = {}
actor_loss_steps_runs = {}

n_train_episodes_runs = np.zeros(n_runs)
n_train_episodes_runs.fill(original_max_train_episodes)

for run_num in range(n_runs):

    print('Run ' + str(run_num))

    global_nfe = 0
    max_train_episodes = original_max_train_episodes

    current_save_path = save_path + "run " + str(run_num) 

    if not os.path.exists(current_save_path):
        os.mkdir(current_save_path)

    file_name = "RL_training_designs_ppo"
    if artery_prob:
        file_name += "_artery_"
    else:
        file_name += "_eqstiff_"

    if n_heurs_used > 0:
        for i in range(len(heur_abbr)):
            if heurs_used[i]:
                file_name += heur_abbr[i]
        
    file_name += str(run_num) + ".csv"

    ## Initialize actor and critic
    if new_reward and include_weights:
        actor_net = create_actor(num_states=n_states+len(obj_names)-1, hidden_layer_params=actor_fc_layer_params, hidden_layer_dropout_params=actor_dropout_layer_params, num_action_vals=n_action_vals, num_actions=n_actions) # Assuming the weights of all but last objective are in the state
        critic_net = create_critic(num_states=n_states+len(obj_names)-1, hidden_layer_params=critic_fc_layer_params, hidden_layer_dropout_params=critic_dropout_layer_params) # Assuming the weights of all but last objective are in the state
    else:
        actor_net = create_actor(num_states=n_states, hidden_layer_params=actor_fc_layer_params, hidden_layer_dropout_params=actor_dropout_layer_params, num_action_vals=n_action_vals, num_actions=n_actions)
        critic_net = create_critic(num_states=n_states, hidden_layer_params=critic_fc_layer_params, hidden_layer_dropout_params=critic_dropout_layer_params)

    ## Initialize the optimizers
    policy_lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_actor_learning_rate, decay_steps=decay_steps_actor, decay_rate=decay_rate)
    value_lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_critic_learning_rate, decay_steps=decay_steps_critic, decay_rate=decay_rate)
    policy_optimizer = keras.optimizers.RMSprop(learning_rate=policy_lr_schedule, rho=rho, momentum=momentum)
    value_optimizer = keras.optimizers.RMSprop(learning_rate=value_lr_schedule, rho=rho, momentum=momentum)

    # Initialize result saver
    result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)

    if artery_prob:
        train_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
        eval_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
    else:
        train_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)
        eval_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)

    observation = train_env.reset()
    episode_return, episode_length = 0,0

    # Initialize the buffer
    if use_buffer:
        print('Storing initial trajectories into buffer')
        buffer = Buffer(discrete_actions=discrete_actions, max_buffer_size=replay_buffer_capacity, max_traj_steps=trajectory_collect_steps+1, num_actions=n_actions, observation_dimensions=n_states, action_dimensions=n_action_vals)

        # Generate the initial set of trajectories to add to buffer
        if new_reward:
            if include_weights:
                # Assuming the weights of all but last objective are in the state
                num_states = n_states+len(obj_names)-1
            else:
                num_states = n_states
            init_trajectory_states, init_trajectory_actions, init_trajectory_rewards, init_trajectory_dones, init_trajectory_policy_logits, init_trajectory_truss_des, modified_nfe = generate_trajectories(num_states=num_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=initial_collect_trajs, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
            global_nfe = modified_nfe
        else:
            init_trajectory_states, init_trajectory_actions, init_trajectory_rewards, init_trajectory_dones, init_trajectory_policy_logits, init_trajectory_truss_des, modified_nfe = generate_trajectories(num_states=n_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=initial_collect_trajs, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
            global_nfe = modified_nfe
        store_into_buffer(buffer=buffer, n_trajs=initial_collect_trajs, collect_steps=trajectory_collect_steps, trajectory_states=init_trajectory_states, trajectory_truss_designs=init_trajectory_truss_des, trajectory_actions=init_trajectory_actions, trajectory_rewards=init_trajectory_rewards, trajectory_dones=init_trajectory_dones, trajectory_policy_logits=init_trajectory_policy_logits)

    ## Start Training
    print('Starting training')
    if compute_periodic_returns:
        returns_episodes = []
        returns_episodes_steps = []

    critic_losses_episodes = []
    actor_losses_episodes = []
    if use_early_stopping:
        actor_end_steps_episodes = []

    critic_losses_episodes_steps = []
    actor_losses_episodes_steps = []

    #created_designs = set()

    #returns_step_count = 0
    actor_train_count = 0
    critic_train_count = 0
    overall_step_counter = 0 # used for result logger

    try:
        while True:
            with tqdm(total=max_train_episodes, desc='Episode') as pbar:

                for episode in range(max_train_episodes):

                    #stop = False
                    critic_losses_iterations = []
                    actor_losses_iterations = []

                    critic_loss_steps = []
                    actor_loss_steps = []

                    # Evaluate actor at chosen intervals
                    if compute_periodic_returns:
                        if episode % eval_interval == 0:
                            avg_return = compute_avg_return(eval_env, actor_net, max_eval_steps, max_eval_episodes)
                            #print('step = {0}: Average Return = {1:.2f}'.format(returns_step_count, avg_return))
                            print('episode = {0}: Average Return = {1:.2f}'.format(episode, avg_return))
                            returns_episodes.append(avg_return)
                            returns_episodes_steps.append(episode)
                            #returns_step_count += 1

                    # Predefine arrays for states, actions, rewards, values, advantages, returns and logits for the training minibatch
                    if new_reward:
                        if include_weights:
                            # Assuming the weights of all but last objective are in the state
                            train_minibatch_states = np.zeros((episode_training_trajs*minibatch_steps, n_states+len(obj_names)-1), dtype=np.float32)
                        else:
                            train_minibatch_states = np.zeros((episode_training_trajs*minibatch_steps, n_states), dtype=np.int32)
                    else:
                        train_minibatch_states = np.zeros((episode_training_trajs*minibatch_steps, n_states), dtype=np.int32)
                    train_minibatch_actions = np.zeros((episode_training_trajs*minibatch_steps, n_actions))
                    train_minibatch_rewards = np.zeros(episode_training_trajs*minibatch_steps)
                    train_minibatch_values = np.zeros(episode_training_trajs*minibatch_steps)
                    train_minibatch_advantages = np.zeros(episode_training_trajs*minibatch_steps)
                    train_minibatch_returns = np.zeros(episode_training_trajs*minibatch_steps)
                    if discrete_actions:
                        train_minibatch_logits = np.zeros((episode_training_trajs*minibatch_steps, n_actions*n_action_vals), dtype=np.float32) 
                    else:
                        train_minibatch_logits = np.zeros((episode_training_trajs*minibatch_steps, 2 * n_actions), dtype=np.float32)

                    # Generate a trajectory to add to the buffer
                    if use_buffer:
                        print('Add a trajectory to the buffer')
                        if new_reward:
                            if include_weights:
                                # Assuming the weights of all but last objective are in the state
                                num_states = n_states+len(obj_names)-1
                            else:
                                num_states = n_states
                            trajectory_states, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits, trajectory_truss_des, modified_nfe = generate_trajectories(num_states=num_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=1, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
                            global_nfe = modified_nfe 
                        else:
                            trajectory_states, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits, trajectory_truss_des, modified_nfe = generate_trajectories(num_states=n_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=1, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
                            global_nfe = modified_nfe 
                        store_into_buffer(buffer=buffer, n_trajs=1, collect_steps=trajectory_collect_steps, trajectory_states=trajectory_states, trajectory_truss_designs=trajectory_truss_des, trajectory_actions=trajectory_actions, trajectory_rewards=trajectory_rewards, trajectory_dones=trajectory_dones, trajectory_policy_logits=trajectory_policy_logits)

                    # Generate/sample trajectories to populate training minibatch
                    print('Generate/sample trajectories for training')
                    for traj_step in range(episode_training_trajs):

                        if use_buffer:
                            # Sample trajectories from the buffer
                            n_stored_trajs = buffer.get_num_trajectories()

                            # Retrieve a random trajectory slice from the buffer to construct training minibatch 
                            traj_idx = np.random.randint(low=0, high=n_stored_trajs)
                            traj_full_obs, traj_full_truss_des, traj_full_acts, traj_full_rs, traj_full_dones, traj_full_logits = buffer.get_trajectory(trajectory_index=traj_idx)
                            if sample_minibatch:
                                if use_continuous_minibatch:
                                    start_idx = np.random.randint(low=0, high=(trajectory_collect_steps-minibatch_steps-1)) # -1 to account for additional observation sampled for last time step delta computation
                                    end_idx = start_idx + minibatch_steps
                                    indices_array = np.arange(start_idx, end_idx)
                                else:
                                    indices_array = np.random.randint(0, trajectory_collect_steps-1, size=minibatch_steps)
                            else:
                                indices_array = np.arange(0, trajectory_collect_steps)
                            traj_obs = traj_full_obs[indices_array, :]
                            traj_truss_des = traj_full_truss_des[indices_array]
                            traj_acts = traj_full_acts[indices_array, :]
                            traj_rs = traj_full_rs[indices_array]
                            traj_dones = traj_full_dones[indices_array]
                            traj_logits = traj_full_logits[indices_array, :]

                            # Log trajectory into the result logger (last observation not stored)
                            for step in range(minibatch_steps):
                                #chosen_act = np.argmax(traj_acts[step, :])
                                #result_logger.save_to_logger(step_number=overall_step_counter, action=traj_acts[step, :], prev_obs=traj_obs[step, :], reward=traj_rs[step])
                                #result_logger.save_to_logger(step_number=overall_step_counter, action=traj_acts[step, 0], prev_obs=traj_obs[step, :], reward=traj_rs[step])
                                result_logger.save_to_logger2(step_number=overall_step_counter, action=traj_acts[step, 0], truss_design=traj_truss_des[step], reward=traj_rs[step])
                                overall_step_counter += 1

                            # Obtain value estimation from critic for the trajectory observations
                            #traj_obs_exp = tf.expand_dims(traj_obs, axis=0)
                            traj_full_vals = critic_net(traj_full_obs, training=False)
                            traj_full_vals_array = tf.squeeze(traj_full_vals).numpy()
                            traj_vals_array = traj_full_vals_array[indices_array]

                            # Obtain estimated value for additional observation separately
                            #traj_objs_end_expanded = tf.expand_dims(traj_obs[-1,:], axis=0)
                            #traj_val_end = tf.squeeze(critic_net(traj_objs_end_expanded, training=False)).numpy()

                            # Compute advantages and returns using the rewards and value estimates
                            traj_returns, traj_advantages = compute_advantages_and_returns(indices_array, traj_full_rs, traj_full_vals_array, advantage_norm)

                            # Add to minibatch
                            train_minibatch_states[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_obs
                            train_minibatch_actions[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_acts
                            train_minibatch_rewards[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_rs
                            train_minibatch_values[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_vals_array
                            train_minibatch_returns[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_returns
                            train_minibatch_advantages[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_advantages
                            train_minibatch_logits[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_logits    
                        
                        else:
                            # Generate trajectory
                            if new_reward:
                                if include_weights:
                                    # Assuming the weights of all but last objective are in the state
                                    num_states = n_states+len(obj_names)-1
                                else:
                                    num_states = n_states
                                trajectory_states, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits, trajectory_truss_des, modified_nfe = generate_trajectories(num_states=num_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=1, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
                                global_nfe = modified_nfe
                            else:
                                trajectory_states, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits, trajectory_truss_des, modified_nfe = generate_trajectories(num_states=n_states, num_actions=n_actions, num_action_vals=n_action_vals, n_trajs=1, collect_steps=trajectory_collect_steps+1, actor=actor_net, current_nfe=global_nfe)
                                global_nfe = modified_nfe

                            # Extract slice from the trajectory 
                            if sample_minibatch:
                                if use_continuous_minibatch: #continuous trajectory slice is considered instead of random steps in the trajectory
                                    start_idx = np.random.randint(low=0, high=(trajectory_collect_steps-minibatch_steps-1)) # -1 to account for additional observation sampled for last time step delta computation
                                    end_idx = start_idx + minibatch_steps
                                    indices_array = np.arange(start_idx, end_idx)    
                                else:
                                    indices_array = np.random.randint(0, trajectory_collect_steps-1, size=minibatch_steps)
                            else:
                                indices_array = np.arange(0, trajectory_collect_steps)
                            
                            traj_slice_states = trajectory_states[0, indices_array, :]
                            traj_slice_truss_des = np.array(trajectory_truss_des)[0, indices_array] 
                            traj_slice_actions = trajectory_actions[0, indices_array, :]
                            traj_slice_rewards = trajectory_rewards[0, indices_array]
                            traj_slice_policy_logits = trajectory_policy_logits[0, indices_array, :]

                            # Log trajectory into the result logger
                            for step in range(minibatch_steps):
                                #chosen_act = np.argmax(traj_slice_actions[step, :])
                                #result_logger.save_to_logger(step_number=overall_step_counter, action=traj_slice_actions[step, :], prev_obs=traj_slice_states[step, :], reward=traj_slice_rewards[step])
                                #result_logger.save_to_logger(step_number=overall_step_counter, action=traj_slice_actions[step, 0], prev_obs=traj_slice_states[step, :], reward=traj_slice_rewards[step])
                                result_logger.save_to_logger2(step_number=overall_step_counter, action=traj_slice_actions[step, 0], truss_design=traj_slice_truss_des[step], reward=traj_slice_rewards[step])
                                overall_step_counter += 1

                            # Obtain value estimation from critic for the trajectory observations
                            #trajectory_states_exp = tf.expand_dims(trajectory_states, axis=0)
                            trajectory_vals = critic_net(trajectory_states, training=False)
                            trajectory_vals_array = tf.squeeze(trajectory_vals).numpy()
                            traj_slice_vals = trajectory_vals_array[indices_array]

                            # Obtain estimated value for additional observation separately
                            #traj_slice_end_state_expanded = tf.expand_dims(traj_slice_states[-1,:], axis=0)
                            #traj_slice_val_end = tf.squeeze(critic_net(traj_slice_end_state_expanded, training=False)).numpy()

                            # Compute advantages and returns using the rewards and value estimates
                            traj_slice_returns, traj_slice_advantages = compute_advantages_and_returns(indices_array, trajectory_rewards[0,:], trajectory_vals_array, advantage_norm)

                            # Add to minibatch
                            train_minibatch_states[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_slice_states
                            train_minibatch_actions[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_slice_actions
                            train_minibatch_rewards[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_slice_rewards
                            train_minibatch_values[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_slice_vals
                            train_minibatch_returns[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_slice_returns
                            train_minibatch_advantages[minibatch_steps*traj_step:minibatch_steps*(traj_step+1)] = traj_slice_advantages
                            train_minibatch_logits[minibatch_steps*traj_step:minibatch_steps*(traj_step+1), :] = traj_slice_policy_logits

                    # Use the minibatch to train the actor and critic
                    print('Actor Training')
                    with tqdm(total=train_policy_iterations, desc='Actor training') as pbar_act:
                        if use_early_stopping:
                            training_end_step = 0
                        for _ in range(train_policy_iterations):
                            actor_train_loss, mean_kl = train_actor(observation_samples=train_minibatch_states, action_samples=train_minibatch_actions, logits_samples=train_minibatch_logits, advantage_samples=train_minibatch_advantages, target_kl=kl_targ, entropy_coeff=ent_coeff)
                            actor_losses_iterations.append(actor_train_loss.numpy())
                            print('Step: ' + str(actor_train_count) + ', actor loss: ' + str(actor_train_loss.numpy()) + ', mean KL-divergence: ' + str(mean_kl))
                            actor_train_count += 1
                            pbar_act.update(1)
                            if use_early_stopping:
                                if mean_kl > 1.5 * kl_targ:
                                    # Early Stopping
                                    break
                                else:
                                    training_end_step += 1
                        if use_early_stopping:
                            actor_end_steps_episodes.append(training_end_step)

                    print('Critic Training \n')
                    with tqdm(total=train_value_iterations, desc='Critic training') as pbar_val:
                        for _ in range(train_value_iterations):
                            critic_train_loss = train_critic(observation_samples=train_minibatch_states, return_samples=train_minibatch_returns)
                            critic_losses_iterations.append(critic_train_loss.numpy())
                            print('Step: ' + str(critic_train_count) + ', critic loss: ' + str(critic_train_loss.numpy()))
                            critic_train_count += 1
                            pbar_val.update(1)

                    critic_loss_steps.append([i for i in range(critic_train_count)])
                    actor_loss_steps.append([i for i in range(actor_train_count)])
                    actor_train_count = 0
                    critic_train_count = 0

                    critic_losses_episodes.append(critic_losses_iterations)
                    actor_losses_episodes.append(actor_losses_iterations)
                    critic_losses_episodes_steps.append(critic_loss_steps)
                    actor_losses_episodes_steps.append(actor_loss_steps)
                    
                    pbar.update(1)

                    # Save current trained actor and critic networks at regular intervals
                    if episode % network_save_intervals == 0:
                        actor_model_filename = "learned_actor_network_ep" + str(episode)
                        if artery_prob:
                            actor_model_filename += "_artery"
                        else:
                            actor_model_filename += "_eqstiff"
                        actor_model_filename += ".h5"

                        actor_net.save(os.path.join(current_save_path, actor_model_filename), save_format='h5')

                        critic_model_filename = "learned_critic_network_ep" + str(episode)
                        if artery_prob:
                            critic_model_filename += "_artery"
                        else:
                            critic_model_filename += "_eqstiff"
                        critic_model_filename += ".h5"

                        critic_net.save(os.path.join(current_save_path, critic_model_filename), save_format='h5')

                    if global_nfe >= max_unique_nfe_run:
                        #stop = True
                        n_train_episodes_runs[run_num] = episode+1 # in case max_unique_nfe_run is reached and the run ends prematurely, otherwise max_train_episodes stays the same
                        break
                    else:
                        print('Global NFE: ' + str(global_nfe))

            break

    except KeyboardInterrupt: # Stop early using keyboard interruption (Ctrl + z) to avoid continuing training which does not improve actor and/or critic learning
        n_train_episodes_runs[run_num] = episode+1
        break

    # Save results to the file
    result_logger.save_to_csv()

    # Save trained actor and critic networks
    actor_model_filename = "learned_actor_network_final"
    if artery_prob:
        actor_model_filename += "_artery"
    else:
        actor_model_filename += "_eqstiff"
    actor_model_filename += ".h5"

    actor_net.save(os.path.join(current_save_path, actor_model_filename), save_format='h5')

    critic_model_filename = "learned_critic_network_final"
    if artery_prob:
        critic_model_filename += "_artery"
    else:
        critic_model_filename += "_eqstiff"
    critic_model_filename += ".h5"

    critic_net.save(os.path.join(current_save_path, critic_model_filename), save_format='h5')

    # Save computed returns
    if compute_periodic_returns:
        returns_runs['run' + str(run_num)] = returns_episodes
        #returns_steps_runs['run' + str(run_num)] = [i for i in range(returns_step_count)]
        returns_steps_runs['run' + str(run_num)] = returns_episodes_steps

    # Save episode losses and steps to run dictionaries
    critic_losses_runs['run' + str(run_num)] = critic_losses_episodes
    critic_loss_steps_runs['run' + str(run_num)] = critic_losses_episodes_steps
    actor_losses_runs['run' + str(run_num)] = actor_losses_episodes
    actor_loss_steps_runs['run' + str(run_num)] = actor_losses_episodes_steps
    if use_early_stopping:
        actor_end_steps_runs['run' + str(run_num)] = actor_end_steps_episodes

## Visualize Training
for i in range(n_runs):

    # Plot individual episodic critic training
    critic_losses_current_run = critic_losses_runs['run' + str(i)]
    train_steps_current_run = critic_loss_steps_runs['run' + str(i)]
    episode_counter = 0
    figure_counter = 0
    while episode_counter < n_train_episodes_runs[i]: # NOTE: Edit to use different_max_train_episodes for each run
        plt.figure()
        for k in range(n_episodes_per_fig):
            critic_losses_episode = critic_losses_current_run[episode_counter]
            steps = train_steps_current_run[episode_counter][0]
            plt.plot(steps, critic_losses_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
            episode_counter += 1
            if episode_counter == n_train_episodes_runs[i]:
                break
        plt.ylabel('Critic loss')
        plt.xlabel('Evaluation Step #')
        plt.grid()
        plt.title('Agent Evaluation: Run ' + str(i))
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(save_path + "run " + str(i) + "\\" + "ppo_fig" + str(figure_counter) + "critic_loss.png", dpi=600)    
        figure_counter += 1

for i in range(n_runs):

    # Plot individual episodic actor training
    actor_losses_current_run = actor_losses_runs['run' + str(i)]
    train_steps_current_run = actor_loss_steps_runs['run' + str(i)]
    episode_counter = 0
    figure_counter = 0
    while episode_counter < n_train_episodes_runs[i]:
        plt.figure()
        for k in range(n_episodes_per_fig):
            losses_episode = actor_losses_current_run[episode_counter]
            loss_steps = train_steps_current_run[episode_counter][0]
            plt.plot(loss_steps, losses_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
            episode_counter += 1
            if episode_counter == n_train_episodes_runs[i]:
                break
        plt.ylabel('Loss')
        plt.xlabel('Iteration #')
        plt.grid()
        plt.title('Agent Training Loss: Run ' + str(i))
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(save_path + "run " + str(i) + "\\" + "ppo_fig" + str(figure_counter) + "actor_loss.png", dpi=600)
        figure_counter += 1

if use_early_stopping:
    for i in range(n_runs):
        plt.figure()
        actor_end_steps_current_run = actor_end_steps_runs['run' + str(i)]
        actor_episode_step_numbers = np.arange(n_train_episodes_runs[i])
        plt.plot(actor_episode_step_numbers, actor_end_steps_current_run)
        plt.ylabel('Actor training end step')
        plt.xlabel('Episode #')
        plt.grid()
        plt.title('Agent Training End Steps: Run ' + str(i))
        #plt.show()
        plt.savefig(save_path + "run " + str(i) + "\\" + "ppo_fig" + str(figure_counter) + "actor_train_end_step.png", dpi=600)
            

if compute_periodic_returns:
    run_counter = 0
    while run_counter < n_runs:
        plt.figure()
        returns_current_run = returns_runs['run' + str(run_counter)]
        returns_steps_current_run = returns_steps_runs['run' + str(run_counter)]
        plt.plot(returns_steps_current_run, returns_current_run)
        plt.ylabel('Return')
        plt.xlabel('Episode #')
        plt.grid()
        plt.title('Agent Return ' + str(run_counter))
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(save_path + "run " + str(run_counter) + "\\" + "ppo_fig_agent_return.png", dpi=600)
        run_counter += 1

# Plot combined actor and critic training (TODO: Use the maximum number of training episodes across all runs, pad the losses for the smaller runs accordingly)
actor_loss_combined_runs = {}
critic_loss_combined_runs = {}

actor_combined_train_steps = {}
critic_combined_train_steps = {}

max_train_episode_allruns = np.max(n_train_episodes_runs)

actor_loss_mean_runs = {}
critic_loss_mean_runs = {}

for i in range(n_runs):
    critic_losses_current_run = critic_losses_runs['run' + str(i)]
    critic_train_steps_current_run = critic_loss_steps_runs['run' + str(i)]

    actor_losses_current_run = actor_losses_runs['run' + str(i)]
    actor_train_steps_current_run = actor_loss_steps_runs['run' + str(i)]

    # Approach 1: Combine losses for each episode into a single list while filling gaps for early stopped iterations (conducive for statistics)
    critic_loss_combined_current_run = []
    actor_loss_combined_current_run = []

    critic_loss_mean_current_run = []
    actor_loss_mean_current_run = []

    for j in range(int(n_train_episodes_runs[i])):
        critic_loss_combined_current_run.extend(critic_losses_current_run[j]) # critic training is never early stopped
        critic_loss_mean_current_run.append(np.mean(critic_losses_current_run[j]))
        
        actor_loss_current_episode = actor_losses_current_run[j]
        actor_loss_combined_current_run.extend(actor_loss_current_episode)
        actor_loss_mean_current_run.append(np.mean(actor_loss_current_episode))

        if len(actor_loss_current_episode) < train_policy_iterations:
            for k in range(train_policy_iterations - len(actor_loss_current_episode)):
                actor_loss_combined_current_run.append(actor_loss_current_episode[-1]) # extend list to match the number of policy training iterations by copying the last policy loss

    if n_train_episodes_runs[i] < max_train_episode_allruns: # extend losses array to match the maximum number of training episodes across all runs
        for j2 in range(int(max_train_episode_allruns - n_train_episodes_runs[i])):
            current_episode_actor_losses = list(np.multiply(actor_loss_combined_current_run[-1], np.ones(train_policy_iterations)))
            actor_loss_combined_current_run.extend(current_episode_actor_losses)

            current_episode_critic_losses = list(np.multiply(critic_loss_combined_current_run[-1], np.ones(train_value_iterations)))
            critic_loss_combined_current_run.extend(current_episode_critic_losses)


    #actor_train_steps_combined_current_run = np.arange(n_train_episodes_runs[i]*train_policy_iterations)
    #critic_train_steps_combined_current_run = np.arange(n_train_episodes_runs[i]*train_value_iterations)

    actor_train_steps_combined_current_run = np.arange(max_train_episode_allruns*train_policy_iterations)
    critic_train_steps_combined_current_run = np.arange(max_train_episode_allruns*train_value_iterations)

    # # Approach 2: Combine losses for each episode into a single list without filling gaps 
    # critic_loss_combined_current_run = list(chain(*critic_losses_current_run.values()))
    # actor_loss_combined_current_run = list(chain(*actor_losses_current_run.values()))

    # # Combine train steps for actor and critic into a single list 
    # current_episode_count = 0
    # critic_train_steps_combined_current_run = []
    # while current_episode_count < max_train_episodes_runs[i]:
    #     critic_train_steps_combined_current_run.extend(np.add(critic_train_steps_current_run[current_episode_count], train_value_iterations*current_episode_count))
    #     current_episode_count += 1

    # actor_train_steps_combined_current_run = []
    # current_end_step = 0
    # for j in range(max_train_episodes_runs[i]):
    #     actor_train_steps_episode = actor_train_steps_current_run[j]
    #     for j2 in range(len(actor_train_steps_episode)):
    #         actor_train_steps_combined_current_run.append(actor_train_steps_episode[j2] + (current_end_step + 1))
    #     current_end_step = actor_train_steps_combined_current_run[-1]

    actor_loss_combined_runs['run ' + str(i)] = actor_loss_combined_current_run
    critic_loss_combined_runs['run ' + str(i)] = critic_loss_combined_current_run

    actor_loss_mean_runs['run ' + str(i)] = actor_loss_mean_current_run
    critic_loss_mean_runs['run ' + str(i)] = critic_loss_mean_current_run

    actor_combined_train_steps['run ' + str(i)] = actor_train_steps_combined_current_run
    critic_combined_train_steps['run ' + str(i)] = critic_train_steps_combined_current_run

# Compute statistics (TODO: Use the maximum number of training epoisodes across all runs)
n_stats_interval = 5

actor_stats_steps = np.arange(max_train_episode_allruns*train_policy_iterations, step=n_stats_interval)
critic_stats_steps = np.arange(max_train_episode_allruns*train_value_iterations, step=n_stats_interval)

actor_mean_stats_steps = np.arange(max_train_episodes)
critic_mean_stats_steps = np.arange(max_train_episodes)

# Computing stats for combined actor loss 
actor_loss_med = []
actor_loss_1q = []
actor_loss_3q = []
for step_val in actor_stats_steps:
    actor_loss_runs = np.zeros(n_runs)

    for i in range(n_runs):
        actor_step_idx = list(actor_combined_train_steps['run ' + str(i)]).index(step_val)
        actor_loss_runs[i] = actor_loss_combined_runs['run ' + str(i)][actor_step_idx]

    actor_loss_med.append(np.median(actor_loss_runs))
    actor_loss_1q.append(np.percentile(actor_loss_runs, 25))
    actor_loss_3q.append(np.percentile(actor_loss_runs, 75))

# Computing stats for mean actor loss 
actor_mean_loss_med = []
actor_mean_loss_1q = []
actor_mean_loss_3q = []
for step_val in actor_mean_stats_steps:
    actor_mean_loss_runs = np.zeros(n_runs)

    for i in range(n_runs):
        actor_mean_loss_runs[i] = actor_loss_mean_runs['run ' + str(i)][step_val]

    actor_mean_loss_med.append(np.median(actor_mean_loss_runs))
    actor_mean_loss_1q.append(np.percentile(actor_mean_loss_runs, 25))
    actor_mean_loss_3q.append(np.percentile(actor_mean_loss_runs, 75))

# Computing stats for combined critic loss 
critic_loss_med = []
critic_loss_1q = []
critic_loss_3q = []
for step_val in critic_stats_steps:
    critic_loss_runs = np.zeros(n_runs)

    for i in range(n_runs):
        critic_step_idx = list(critic_combined_train_steps['run ' + str(i)]).index(step_val)
        critic_loss_runs[i] = critic_loss_combined_runs['run ' + str(i)][critic_step_idx]

    critic_loss_med.append(np.median(critic_loss_runs))
    critic_loss_1q.append(np.percentile(critic_loss_runs, 25))
    critic_loss_3q.append(np.percentile(critic_loss_runs, 75))

# Computing stats for mean critic loss 
critic_mean_loss_med = []
critic_mean_loss_1q = []
critic_mean_loss_3q = []
for step_val in critic_mean_stats_steps:
    critic_mean_loss_runs = np.zeros(n_runs)

    for i in range(n_runs):
        critic_mean_loss_runs[i] = critic_loss_mean_runs['run ' + str(i)][step_val]

    critic_mean_loss_med.append(np.median(critic_mean_loss_runs))
    critic_mean_loss_1q.append(np.percentile(critic_mean_loss_runs, 25))
    critic_mean_loss_3q.append(np.percentile(critic_mean_loss_runs, 75))

# Plotting actor combined training
plt.figure()
plt.plot(actor_stats_steps, actor_loss_med)
plt.fill_between(actor_stats_steps, actor_loss_1q, actor_loss_3q, alpha=0.6)
plt.xlabel('Training step')
plt.ylabel('Actor loss')
plt.grid()
plt.title('Actor training: episodes combined and averaged over runs')
#plt.show()
plt.savefig(save_path + "ppo_actor_combined_train.png", dpi=600)

# Plotting actor mean training
plt.figure()
plt.plot(actor_mean_stats_steps, actor_mean_loss_med)
plt.fill_between(actor_mean_stats_steps, actor_mean_loss_1q, actor_mean_loss_3q, alpha=0.6)
plt.xlabel('Training step')
plt.ylabel('Mean actor loss')
plt.grid()
plt.title('Actor training: averaged within episodes, combined and averaged over runs')
#plt.show()
plt.savefig(save_path + "ppo_actor_mean_train.png", dpi=600)

# Plotting critic combined training
plt.figure()
plt.plot(critic_stats_steps, critic_loss_med)
plt.fill_between(critic_stats_steps, critic_loss_1q, critic_loss_3q, alpha=0.6)
plt.xlabel('Training step')
plt.ylabel('Critic loss')
plt.grid()
plt.title('Critic training: episodes combined and averaged over runs')
#plt.show()
plt.savefig(save_path + "ppo_critic_combined_train.png", dpi=600)

# Plotting critic combined training
plt.figure()
plt.plot(critic_mean_stats_steps, critic_mean_loss_med)
plt.fill_between(critic_mean_stats_steps, critic_mean_loss_1q, critic_mean_loss_3q, alpha=0.6)
plt.xlabel('Training step')
plt.ylabel('Mean critic loss')
plt.grid()
plt.title('Critic training: averaged within episodes, combined and averaged over runs')
#plt.show()
plt.savefig(save_path + "ppo_critic_mean_train.png", dpi=600)

end_time = time.time()
print("Total time: " + str(end_time-start_time))