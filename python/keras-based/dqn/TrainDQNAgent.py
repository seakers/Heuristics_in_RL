# -*- coding: utf-8 -*-
"""
Training and saving a DQN agent using keras functions instead of tf-agents
Reference: https://keras.io/examples/rl/deep_q_network_breakout/

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"

import json

# Add path for parent directory with environment classes
import sys
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

import tensorflow as tf
import keras 
from keras import layers

from tqdm import tqdm

from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv

import matplotlib.pyplot as plt

import numpy as np
import math 

from save.ResultSaving import ResultSaver

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

## Setup train parameters from config files
f_dqn = open('.\\keras-based\\dqn\\dqn-config.json')
data_dqn = json.load(f_dqn)

gamma = data_dqn["Value discount (gamma)"] # discount factor
epsilon = data_dqn["Epsilon greedy probability theshold (epsilon)"] # epsilon-greedy parameter
epsilon_min = data_dqn["Minimum epsilon"]
epsilon_max = data_dqn["Maximum epsilon"]
epsilon_interval = (epsilon_max - epsilon_min) # used to anneal epsilon as training continues
batch_size = data_dqn["Training batch_size"] # training batch size sampled from replay buffer
num_unique_designs_per_episode = data_dqn["Number of unique designs per episode"] # minimum number of unique designs encountered by the agent, signalling the end of the episode (similar to max_steps_per_episode)
max_train_episodes = data_dqn["Number of training episodes"]

n_runs = data_dqn["Number of runs"]

compute_periodic_returns = data_dqn["Compute periodic returns"]

max_steps = data_dqn["Maximum steps in training episode"] # no termination in training environment
max_eval_steps = data_dqn["Maximum steps in evaluation episode"] # termination in evaluation environment
max_eval_episodes = data_dqn["Maximum number of evaluation of episodes"]

eval_interval = data_dqn["Episode interval for evaluation"]

initial_collect_trajs = data_dqn["Initial number of trajectories to populate buffer"] # number of trajectories in the driver to populate replay buffer before beginning training
initial_collect_steps = data_dqn["Number of steps in a trajectory"] # number of steps in each trajectory
episode_collect_steps = data_dqn["Number of steps in a training episode"] # number of steps in the driver to populate replay buffer during training
replay_buffer_capacity = data_dqn["Replay buffer capacity (transitions)"]

initial_learning_rate = data_dqn["Optimizer initial learning rate"]
decay_rate = data_dqn["Learning rate decay rate"]
decay_steps = max_train_episodes*((num_unique_designs_per_episode/batch_size) + 10) # chosen heuristically assuming "(num_unique_designs_per_episode/batch_size) + 10" steps per episode
rho = data_dqn["RMSprop optimizer rho"]
momentum = data_dqn["RMSprop optimizer momentum"]

fc_layer_params = data_dqn["DQN layer units"]
dropout_layer_params = data_dqn["DQN layer dropout probabilities"]

use_target_q_net = data_dqn["Target Q-network used"]
target_update_steps = data_dqn["Number of steps to update target Q-network"] # number of steps to update target Q-network (steps here is steps as computed to train agent)

save_path = data_dqn["Savepath"]

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

n_episodes_per_fig = 4 # used for plotting returns and losses 
linestyles = ['solid','dotted','dashed','dashdot']

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

## Define method to compute average return for DQN training
def compute_avg_return(environment, q_net, num_steps=100, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        start_obs = environment.reset()
        state = np.array(start_obs)

        episode_return = 0.0

        eval_step = 0

        while eval_step < num_steps:
            # Predict action from current Q-network
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, axis=0)
            action_vals = q_net(state_tensor, training=False)

            # Take best action
            action = tf.argmax(action_vals[0]).numpy()

            next_obs, reward, done, _ = environment.step(action)
            episode_return += reward

            if done:
                break
            else:
                state = np.array(next_obs)
        
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

## Method to generate Q-network with given number of hidden layer neurons and dropout probabilities
def create_Q_net(hidden_layer_params, hidden_layer_dropout_params):
    model = keras.Sequential()
    model.add(layers.Dense(n_states, activation='relu'))
    for n_units, dropout_prob in zip(hidden_layer_params, hidden_layer_dropout_params):
        model.add(layers.Dense(n_units, activation='relu'))
        model.add(layers.Dropout(dropout_prob))
    model.add(layers.Dense(n_actions, activation='linear')) 
    return model

## Method to take action based on epsilon-greedy exploration policy
def get_explore_action(q_network, num_actions, current_state, current_epsilon):

    if np.random.rand(1)[0] < current_epsilon:
        # Take random action
        action = np.random.choice(num_actions)
    else:
        # Predict action from current Q-network
        state_tensor = tf.convert_to_tensor(current_state)
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        action_probs = q_network(state_tensor)

        # Take best action
        action = tf.argmax(action_probs[0]).numpy()

    return action

q_net = create_Q_net(fc_layer_params, dropout_layer_params)
if use_target_q_net:
    target_q_net = create_Q_net(fc_layer_params, dropout_layer_params)

run_train_steps = {}
losses_run = {}
#avg_losses_run = []

if compute_periodic_returns:
    run_eval_steps = {}
    returns_run = {}
    #avg_returns_run = []

# Optimizer
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_rate=decay_rate, decay_steps=decay_steps)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=rho, momentum=momentum)

loss_function = keras.losses.MeanSquaredError()

for run_num in range(n_runs):

    print('Run ' + str(run_num))

    current_save_path = save_path + "run " + str(run_num) 

    if not os.path.exists(current_save_path):
        os.mkdir(current_save_path)

    file_name = "RL_training_designs_dqn"
    if artery_prob:
        file_name += "_artery_"
    else:
        file_name += "_eqstiff_"

    if n_heurs_used > 0:
        for i in range(len(heur_abbr)):
            if heurs_used[i]:
                file_name += heur_abbr[i]
        
    file_name += str(run_num) + ".csv"

    # Initialize result saver
    result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)

    if artery_prob:
        train_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
        eval_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
    else:
        train_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
        eval_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)

    # Replay buffer initialization
    action_history = []
    state_history = []
    next_state_history = []
    reward_history = []
    episode_reward_history = []

    running_reward = 0
    episode_count = 0
    step_count = 0

    # Initial addition of states to the replay buffer
    print('Adding initial trajectories to replay buffer')
    for traj_count in range(initial_collect_trajs):
        print('Adding trajectory ' + str(traj_count) + ' to replay buffer')

        observation = train_env.reset()
        state = np.array(observation)    

        for _ in range(initial_collect_steps):

            # Use epsilon-greedy for exploration
            action = get_explore_action(q_network=q_net, num_actions=n_actions, current_state=state, current_epsilon=epsilon)

            # Decay epsilon parameter  
            epsilon -= epsilon_interval/initial_collect_steps
            epsilon = np.max([epsilon, epsilon_min])

            # Apply action and save to the replay buffer
            next_obs, reward, done, _ = train_env.step(action)
            next_state = np.array(next_obs)

            state_history.append(state)
            action_history.append(action)
            next_state_history.append(next_state)
            reward_history.append(reward)

            state = next_state

    ## Start Training
    episode_count = 0
    returns_episodes = []
    losses_episodes = []

    episode_step_counts = []
    episode_eval_step_counts = []
    created_designs = set()

    while episode_count < max_train_episodes: # Episode loop
        print('Episode ' + str(episode_count))

        observation = train_env.reset()
        state = np.array(observation)

        episode_reward = 0

        unique_design_count = 0

        returns_steps = []
        losses_steps = []

        return_step_counts = []
        loss_step_counts = []

        with tqdm(total=num_unique_designs_per_episode) as pbar:

            while unique_design_count < num_unique_designs_per_episode: # Iterations loop

                episode_collect_step_count = 0
                
                # Add trajectory to the replay buffer
                while episode_collect_step_count < episode_collect_steps:

                    # Use epsilon-greedy for exploration
                    action = get_explore_action(q_network=q_net, num_actions=n_actions, current_state=state, current_epsilon=epsilon)

                    # Take action and add to the replay buffer
                    #print('Environment step')
                    next_obs, reward, done, _ = train_env.step(action)

                    next_state = np.array(next_obs)
                    state_history.append(state)
                    action_history.append(action)
                    reward_history.append(reward)
                    next_state_history.append(next_state)

                    state = next_state

                    episode_collect_step_count += 1

                # Decay epsilon parameter  
                epsilon -= epsilon_interval/initial_collect_steps
                epsilon = np.max([epsilon, epsilon_min])

                ## Train the agent using a minibatch sampled from the replay buffer (assuming replay buffer size > batch size)
                # Get indices of training samples from replay buffer (NOTE: can be modified to extract batches of continuous steps instead of random steps)
                indices = np.random.choice(len(action_history), size=batch_size)

                # Extract samples from replay buffer
                states_batch = np.array([state_history[x] for x in indices])
                actions_batch = np.array([action_history[x] for x in indices])
                rewards_batch = np.array([reward_history[x] for x in indices])
                next_states_batch = np.array([next_state_history[x] for x in indices])

                # Get estimated future cumulative reward from current target Q-network (for stability)
                future_rewards = target_q_net(next_states_batch)

                # Compute Q-value as current reward + gamma * max future cumulative reward
                future_rewards_array = future_rewards.numpy()
                updated_q_vals_batch = rewards_batch + gamma*np.amax(future_rewards_array, axis=1)

                # Create a mask to calculate loss only on the updated Q-values
                mask = tf.one_hot(actions_batch, n_actions)

                # Instantiate gradient tape to train model on states and updated Q-values 
                with tf.GradientTape() as tape:
                    # Current estimated Q-values 
                    q_values = q_net(states_batch)

                    # Apply mask
                    q_mask = tf.math.reduce_sum(tf.math.multiply(q_values, mask), axis=1)

                    # Compute loss function between current and estimated Q-values
                    loss = loss_function(updated_q_vals_batch, q_mask)

                # Backpropagation
                grads = tape.gradient(loss, q_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

                # Update target-Q network every "target_update_steps" steps
                if step_count % target_update_steps == 0:
                    target_q_net.set_weights(q_net.get_weights())

                # Remove excess transitions from the beginning of the buffer if capacity is reached
                if len(action_history) > replay_buffer_capacity:
                    excess = len(action_history) - replay_buffer_capacity
                    del action_history[:excess]
                    del state_history[:excess]
                    del reward_history[:excess]
                    del next_state_history[:excess]

                # Save results to logger
                #print('Environment is done = ' + str(train_env.pyenv.envs[0].gym.get_isdone()))
                #print('Internal step number = ' + str(train_env.pyenv.envs[0].gym.get_step_counter()))
                for i in range(batch_size):
                    current_state = states_batch[i]
                    current_state_str = "".join([str(dec) for dec in current_state])
                    # Do not count repeated designs towards NFEs (but these designs are used to train the agent)
                    if not current_state_str in created_designs:
                        result_logger.save_to_logger(step_number=step_count, action=actions_batch[i], prev_obs=current_state, reward=rewards_batch[i])
                        created_designs.add(current_state_str)
                        unique_design_count += 1
                        step_count += 1   
                        pbar.update(1)

                step_count += 1

                print('step = {0}: loss = {1}'.format(step_count, loss))

                if compute_periodic_returns:
                    if (step_count-1) % eval_interval == 0:
                        avg_return = compute_avg_return(eval_env, q_net, max_eval_steps, max_eval_episodes)
                        print('step = {0}: Average Return = {1:.2f}'.format(step_count, avg_return))
                        returns_steps.append(avg_return)
                        return_step_counts.append(step_count)
                        #eval_step_count += 1

        if compute_periodic_returns:
            episode_eval_step_counts.append(return_step_counts)
            returns_episodes.append(returns_steps)        

        episode_step_counts.append(loss_step_counts)
        losses_episodes.append(losses_steps)

    # Save results to the file
    result_logger.save_to_csv()

    # Save trained Q-network
    model_filename = "learned_Q_Network"
    if artery_prob:
        model_filename += "_artery"
    else:
        model_filename += "_eqstiff"
    model_filename += ".h5"

    q_net.save(os.path.join(current_save_path,model_filename), save_format='h5')

    run_train_steps['run' + str(run_num)] = episode_step_counts
    if compute_periodic_returns:
        run_eval_steps['run' + str(run_num)] = episode_eval_step_counts

    losses_run['run' + str(run_num)] = losses_episodes
    if compute_periodic_returns:
        returns_run['run' + str(run_num)] = returns_episodes
        
## Visualize Training
if compute_periodic_returns:
    for i in range(n_runs):
        returns_current_run = returns_run['run' + str(i)]
        eval_steps_current_run = run_eval_steps['run' + str(i)]
        episode_counter = 0
        while episode_counter < max_train_episodes:
            plt.figure()
            for k in range(n_episodes_per_fig):
                returns_episode = returns_current_run[episode_counter]
                steps = eval_steps_current_run[episode_counter]
                plt.plot(steps, returns_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
                episode_counter += 1
                if episode_counter == max_train_episodes:
                    break
            plt.ylabel('Average Return')
            plt.xlabel('Evaluation Step #')
            plt.grid()
            plt.title('Agent Evaluation: Run ' + str(i))
            plt.legend(loc='best')
            #plt.show()
            plt.savefig(save_path + "dqn_run" + str(i) + "_fig" + str(k) + "avg_returns.png", dpi=600)

for i in range(n_runs):
    losses_current_run = losses_run['run' + str(i)]
    train_steps_current_run = run_train_steps['run' + str(i)]
    episode_counter = 0
    while episode_counter < max_train_episodes:
        plt.figure()
        for k in range(n_episodes_per_fig):
            losses_episode = losses_current_run[episode_counter]
            loss_steps = train_steps_current_run[episode_counter] 
            plt.plot(loss_steps, losses_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
            episode_counter += 1
            if episode_counter == max_train_episodes:
                break
        plt.ylabel('Loss')
        plt.xlabel('Iteration #')
        plt.grid()
        plt.title('Agent Training Loss: Run ' + str(i))
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(save_path + "dqn_run" + str(i) + "_fig" + str(k) + "avg_losses.png", dpi=600)








            

            


                
                









