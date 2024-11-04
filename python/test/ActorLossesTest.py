# -*- coding: utf-8 -*-
"""
Testing of actor losses based on engineered minibatches

@author: roshan94
"""
import sys
import os
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[0]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)
import tensorflow as tf
import keras 
from keras import layers
import numpy as np
import json
import math
from itertools import chain
from tqdm import tqdm

from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from save.ResultSaving import ResultSaver

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

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

seed = 32
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

## Method to compute advantage (true value from generated trajectory) for the state 
def compute_advantage(actor, critic, next_state, reward, value, n_traj_des):
    # gamma and lam are defined globally

    # Generate trajectory from current state
    rewards = np.zeros(n_traj_des)
    values = np.zeros(n_traj_des+1) # Last element is the estimated value of state after the end of the trajectory (for delta calculation)

    rewards[0] = reward
    values[0] = value
    for n in range(n_traj_des-1):
        logits, action = sample_action(actor=actor, observation=next_state)

        next_state_tensor = tf.convert_to_tensor(next_state)
        next_state_tensor = tf.expand_dims(next_state_tensor, axis=0)
        next_state_critic_val = critic(next_state_tensor, training=False)

        values[n+1] = tf.squeeze(next_state_critic_val).numpy()

        traj_start = False 
        if n == 0:
            traj_start = True

        next_obs, current_reward, done, add_arg = env.step(action=action, nfe_val=n+1, include_prev_des=traj_start)
        rewards[n+1] = current_reward
        
        next_state = next_obs

    # Compute critic estimated value for last state in trajectory for delta calculation
    next_state_tensor = tf.convert_to_tensor(next_state)
    next_state_tensor = tf.expand_dims(next_state_tensor, axis=0)
    next_state_critic_val = critic(next_state_tensor, training=False)
    values[-1] = tf.squeeze(next_state_critic_val).numpy()

    deltas = np.zeros(n_traj_des)
    for i in range(n_traj_des):
        deltas[i] = rewards[i] + gamma*values[i+1] - values[i]

    state_advantage = discounted_cumulative_sums(deltas, compute_advantage=True)

    return state_advantage

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

################# OPERATION #################
## Setup and train parameters from config file
f_ppo = open('.\\keras-based\\ppo\\ppo-config.json')
data_ppo = json.load(f_ppo)

discrete_actions = data_ppo["Discrete actions"]
use_clipping = data_ppo["Use clipping loss"] # Use PPO clipping term in actor loss
clip_ratio = data_ppo["Clipping ratio threshold"]
use_adaptive_kl_penalty = data_ppo["Use Adaptive KL penalty loss"] # Use adaptive KL-divergence based penalty term in actor loss
use_entropy_bonus = data_ppo["Use entropy loss bonus"] # Use additional entropy of actor distribution term in actor loss
initial_actor_learning_rate = data_ppo["Initial actor training learning rate"]
decay_steps_actor = data_ppo["Learning rate decay steps (actor)"]
decay_rate = data_ppo["Learning rate decay rate"]
rho = data_ppo["RMSprop optimizer rho"]
momentum = data_ppo["RMSprop optimizer momentum"]
actor_fc_layer_params = data_ppo["Actor network layer units"]
actor_dropout_layer_params = data_ppo["Actor network dropout probabilities"]
critic_fc_layer_params = data_ppo["Critic network layer units"]
critic_dropout_layer_params = data_ppo["Critic network dropout probabilities"]
new_reward = data_ppo["Use new problem formulation"]
include_weights = data_ppo["Include weights in state"]
gamma = data_ppo["Value discount (gamma)"] # discount factor
lam = data_ppo["Advantage discount (lambda)"] # advantage discount factor
max_steps = data_ppo["Maximum steps in training episode (for train environment termination)"] # no termination in training environment
current_save_path = data_ppo["Savepath"]
advantage_norm = data_ppo["Normalize advantages"] # whether to normalize advantages for training
kl_targ = data_ppo["KL target"]
ent_coeff = data_ppo["Entropy coefficient"] # coefficient for the entropy bonus term in actor loss
train_policy_iterations = data_ppo["Number of actor training iterations"]
use_early_stopping = data_ppo["Use early stopping for actor training"] # Stop training epoch early if current KL-divergence crosses 1.5 * kl_targ

## Load problem parameters from config file
f_prob = open('.\\keras-based\\problem-config.json')
data_prob = json.load(f_prob)

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

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

artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
n_actions = 1

if artery_prob:
    env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
else:
    env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)

heurs_used = data_prob["Heuristics used"] # for [partial collapsibility, nodal properties, orientation, intersection]
n_heurs_used = heurs_used.count(True)
sidenum = data_prob["Lattice number of side nodes"]
obj_names = data_prob["Objective names"]
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
n_actions = 1

policy_lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_actor_learning_rate, decay_steps=decay_steps_actor, decay_rate=decay_rate)
policy_optimizer = keras.optimizers.RMSprop(learning_rate=policy_lr_schedule, rho=rho, momentum=momentum)

## Initialize actor and critic
if new_reward and include_weights:
    actor_net = create_actor(num_states=n_states+len(obj_names)-1, hidden_layer_params=actor_fc_layer_params, hidden_layer_dropout_params=actor_dropout_layer_params, num_action_vals=n_action_vals, num_actions=n_actions) # Assuming the weights of all but last objective are in the state
    critic_net = create_critic(num_states=n_states+len(obj_names)-1, hidden_layer_params=critic_fc_layer_params, hidden_layer_dropout_params=critic_dropout_layer_params) # Assuming the weights of all but last objective are in the state
else:
    actor_net = create_actor(num_states=n_states, hidden_layer_params=actor_fc_layer_params, hidden_layer_dropout_params=actor_dropout_layer_params, num_action_vals=n_action_vals, num_actions=n_actions)
    critic_net = create_critic(num_states=n_states, hidden_layer_params=critic_fc_layer_params, hidden_layer_dropout_params=critic_dropout_layer_params)

## Initialize result saver
all_improved = True # generate a minibatch of designs that give positive rewards for the selected actions for True (or nergative rewards for False)

file_name = "RL_test_minibatch_ppo"
if all_improved:
    file_name += "_positive_"
else:
    file_name += "_negative_"

if artery_prob:
    file_name += "_artery_"
else:
    file_name += "_eqstiff_"

if n_heurs_used > 0:
    for i in range(len(heur_abbr)):
        if heurs_used[i]:
            file_name += heur_abbr[i]
    
file_name += ".csv"

result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)

## Additional parameters
n_des_minibatch = 100 # number of designs in the minibatch
n_traj_des = 50 # number of designs in a trajectory used to compute the advatange and return for a state

## Generate minibatch
des_count = 0

# Predefine arrays for states, actions, rewards, values, advantages, returns and logits for the training minibatch
if new_reward:
    if include_weights:
        # Assuming the weights of all but last objective are in the state
        train_minibatch_states = np.zeros((n_des_minibatch, n_states+len(obj_names)-1), dtype=np.float32)
    else:
        train_minibatch_states = np.zeros((n_des_minibatch, n_states), dtype=np.int32)
else:
    train_minibatch_states = np.zeros((n_des_minibatch, n_states), dtype=np.int32)
train_minibatch_actions = np.zeros((n_des_minibatch, n_actions))
train_minibatch_rewards = np.zeros(n_des_minibatch)
train_minibatch_values = np.zeros(n_des_minibatch)
train_minibatch_advantages = np.zeros(n_des_minibatch)
if discrete_actions:
    train_minibatch_logits = np.zeros((n_des_minibatch, n_actions*n_action_vals), dtype=np.float32) 
else:
    train_minibatch_logits = np.zeros((n_des_minibatch, 2 * n_actions), dtype=np.float32)

## Add states to minibatch that have positive or negative rewards based on all_improved
with tqdm(total=n_des_minibatch, desc='Minibatch generation') as pbar:
    while des_count < n_des_minibatch:
        state = env.reset()

        logits, action = sample_action(actor=actor_net, observation=state)

        next_obs, reward, done, add_arg = env.step(action=action, nfe_val=des_count, include_prev_des=True)
        if new_reward:
            if include_weights:
                #next_obs = sum(next_obs.values(), [])
                next_obs = list(chain(*next_obs.values()))

            current_nfe = add_arg['Current NFE']
            current_truss_des = add_arg['Current truss design']
            new_truss_des = add_arg['New truss design']

        if (all_improved and (reward > 0)) or ((not all_improved) and (reward <= 0)):        
            # Save the states, actions, rewards and actor logits
            train_minibatch_states[des_count, :] = state
            train_minibatch_logits[des_count, :] = logits
            train_minibatch_actions[des_count, :] = action
            train_minibatch_rewards[des_count]  = reward

            # Compute advantage and critic estimated value
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            state_critic_val = critic_net(state_tensor, training=False)
            train_minibatch_values[des_count] = tf.squeeze(state_critic_val).numpy()

            # Compute advantage using the rewards and value estimates
            state_advantage = compute_advantage(actor=actor_net, critic=critic_net, next_state=next_obs, reward=reward, value=state_critic_val, n_traj_des=n_traj_des)

            train_minibatch_advantages[des_count] = state_advantage

            des_count += 1
            pbar.update(1)

# Use the minibatch to train the actor and critic
print('Actor Training')
actor_train_count = 0
with tqdm(total=train_policy_iterations, desc='Actor training') as pbar_act:
    for _ in range(train_policy_iterations):
        actor_train_loss, mean_kl = train_actor(observation_samples=train_minibatch_states, action_samples=train_minibatch_actions, logits_samples=train_minibatch_logits, advantage_samples=train_minibatch_advantages, target_kl=kl_targ, entropy_coeff=ent_coeff)
        print('Step: ' + str(actor_train_count) + ', actor loss: ' + str(actor_train_loss.numpy()) + ', mean KL-divergence: ' + str(mean_kl))
        actor_train_count += 1
        pbar_act.update(1)
        if use_early_stopping:
            if mean_kl > 1.5 * kl_targ:
                # Early Stopping
                break

# Test with training states to check updated action distributions
n_correct_prob_update = 0 # number of states for which probability of selected action was updated correctly (based on whether all_improve is True or not)
for i in range(train_minibatch_states.shape[0]):

    state = train_minibatch_states[i,:]
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, axis=0)
    updated_action_logits = tf.squeeze(actor_net(state_tensor, training=False))

    prev_action_logits = train_minibatch_logits[i,:]
    improving_action = int(train_minibatch_actions[i,0])

    old_probs = tf.nn.softmax(prev_action_logits)
    new_probs = tf.nn.softmax(updated_action_logits)

    print('Action probabilities before training: ' + str(old_probs))
    print('Action probabilities after training: ' + str(new_probs))
    print('Old probability of selected action: ' + str(old_probs[improving_action]))
    print('Updated probability of selected action: ' + str(new_probs[improving_action]))
    print('Selected action: ' + str(improving_action))

    if ((new_probs[improving_action] > old_probs[improving_action]) and (all_improved)) or ((new_probs[improving_action] < old_probs[improving_action]) and (not all_improved)):
        n_correct_prob_update += 1

    print('\n')

print('Fraction of states with correctly updated action probabilities: ' + str(n_correct_prob_update/n_des_minibatch))
