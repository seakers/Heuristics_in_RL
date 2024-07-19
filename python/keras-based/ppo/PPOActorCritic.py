# -*- coding: utf-8 -*-
"""
General keras-only PPO class: used for creating individual PPO instances for hyperparameter tuning
Reference: https://keras.io/examples/rl/ppo_cartpole/

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"

import gymnasium

from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete

import tensorflow as tf
import keras 
from keras import layers

from tqdm import tqdm

from ppo_buffer import Buffer

from save.ResultSaving import ResultSaver

import matplotlib.pyplot as plt

import numpy as np
import math 

class PPOActorCritic():

    def __init__(self, train_env, eval_env, max_buffer_size, max_traj_steps):
        super(PPOActorCritic, self).__init__()

        # Define class parameters
        self.training_environment = train_env
        self.evaluation_environment = eval_env

        self.states_space = train_env.observation_space
        self.action_space = train_env.action_space

        self.discrete_actions = False
        if isinstance(self.action_space, (Discrete, MultiDiscrete)):
            self.discrete_actions = True
            # this is the number of neurons in the output layer of the actor
            self.n_actions_total = np.sum(np.ndarray.flatten(self.action_space.nvec)) # summing over the total number of options for each discrete state
        else:
            self.n_actions_total = 2*np.sum(self.action_space.shape) # each continuous state distribution is represented by a gaussian with mean and variance from the actor output

        # Initialize actor and critic networks (Networks are defined in a method called before training)
        self.actor_network = None
        self.critic_network = None

        ### To modify and improve
        #self.buffer = Buffer(discrete_actions=self.discrete_actions, max_buffer_size=max_buffer_size, max_traj_steps=max_traj_steps, num_actions=, observation_dimensions=, action_dimensions=)

    ## Define method to compute average return for agent evaluation
    def compute_avg_return(self, num_steps=100, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            start_obs = self.evaluation_environment.reset()
            state = np.array(start_obs)

            episode_return = 0.0

            eval_step = 0

            while eval_step < num_steps:
                # Predict action from current Q-network
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, axis=0)
                action_logits = self.actor_network(state_tensor, training=False)

                # Take best action
                action_probs = tf.nn.softmax(action_logits)
                action = tf.argmax(tf.squeeze(action_probs)).numpy()

                next_obs, reward, done, _ = self.evaluation_environment.step(action)
                episode_return += reward

                if done:
                    break
                else:
                    state = np.array(next_obs)

            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return

    ## Method to generate actor network with given number of hidden layer neurons and dropout probabilities (input layer is generated automatically from observation space)
    def create_actor(self, layers, hidden_layer_params, hidden_layer_dropout_params, activations, initial_activation):
        actor = keras.Sequential()
        if len(self.states_space.shape) == 1: # Single dimension observations
            actor.add(layers.Dense(self.states_space[0], activation=initial_activation))
        elif len(self.states_space.shape) == 2: # Grayscale or binary images or similar
            actor.add(layers.Conv2D(self.states_space, activation=initial_activation))
        elif len(self.states_space.shape) == 3: # RGB images or similar
            actor.add(layers.Conv3D(self.states_space, activation=initial_activation))

        for layer, num_units, dropout_prob, activ in zip(layers, hidden_layer_params, hidden_layer_dropout_params, activations):
            eval("actor.add(layers." + layer + "(" + str(num_units) + ",activation=" + activ + ")")
            eval("actor.add(layers.Dropout(" + str(dropout_prob) + ")")

        actor.add(layers.Dense(self.n_actions_total, activation=None)) # output layer
        return actor

    ## Method to generate critic network with given number of hidden layer neurons and dropout probabilities
    def create_critic(self, hidden_layer_params, hidden_layer_dropout_params, activations, initial_activation):
        critic = keras.Sequential()
        if len(self.states_space.shape) == 1: # Single dimension observations
            critic.add(layers.Dense(self.states_space[0], activation=initial_activation))
        elif len(self.states_space.shape) == 2: # Grayscale or binary images or similar
            critic.add(layers.Conv2D(self.states_space, activation=initial_activation))
        elif len(self.states_space.shape) == 3: # RGB images or similar
            critic.add(layers.Conv3D(self.states_space, activation=initial_activation))
        
        for layer, num_units, dropout_prob, activ in zip(layers, hidden_layer_params, hidden_layer_dropout_params, activations):
            eval("critic.add(layers." + layer + "(" + str(num_units) + ", activation=" + activ + ")")
            eval("critic.add(layers.Dropout(" + str(dropout_prob) + ")")
        
        critic.add(layers.Dense(1, activation=None)) # output layer
        return critic

    ## Method to compute the log-probabilities of the actor network logit for the selected action of the batch
    def log_probabilities(self, logits, a, num_actions):
        log_probs_all = tf.nn.log_softmax(logits)
        log_prob_action = tf.reduce_sum(tf.math.multiply(tf.one_hot(a, num_actions), log_probs_all), axis=1)
        return log_prob_action

    seed = 10

    ## Generate trajectories for training
    def generate_trajectories(self, n_trajs, collect_steps, actor):
        traj_shape_states = [n_trajs, collect_steps]
        traj_shape_actions = [n_trajs, collect_steps]
        for dim in self.states_space.shape:
            traj_shape_states.append(dim)
        
        for dim in self.action_space.shape:
            traj_shape_actions.append(dim)

        traj_states = np.zeros(tuple(traj_shape_states))
        traj_actions = np.zeros(tuple(traj_shape_actions))
        traj_rewards = np.zeros((n_trajs, collect_steps))
        traj_dones = np.zeros((n_trajs, collect_steps))

        if self.discrete_actions:
            traj_policy_logits = np.zeros((n_trajs, collect_steps, num_action_vals)) # for discrete actions, the logits are directly converted to probabilities
        else:
            traj_policy_logits = np.zeros((n_trajs, collect_steps, 2 * num_action_vals)) # for continuous actions, the probability distribution for an action 
            # is taken as gaussian with two logits representing the mean and variance respectively

        # Genarating and storing trjectories (buffer used to compute advantages and values)
        for traj_count in range(n_trajs):
            print('Generating trajectory ' + str(traj_count))

            observation = train_env.reset()
            state = np.array(observation)    
            
            for step in range(collect_steps):

                traj_states[traj_count, step, :] = state

                # Sample action from the current actor
                logits, action = sample_action(actor, state)
                traj_policy_logits[traj_count, step, :] = logits
                traj_actions[traj_count, step, :] = action

                # Apply action and save to the replay buffer
                next_obs, reward, done, _ = train_env.step(action)
                next_state = np.array(next_obs)
                traj_rewards[traj_count, step] = reward
                traj_dones[traj_count, step] = done

                #buffer.store_to_trajectory(state, action, reward, logits)

                if done:
                    traj_dones[traj_count, step+1:] = np.ones(len(collect_steps)-step)
                    break

                state = next_state

            #buffer.store_to_buffer()

        return traj_states, traj_actions, traj_rewards, traj_dones, traj_policy_logits

    ## Method to store trajectories to buffer
    def store_into_buffer(self, buffer, n_trajs, collect_steps, trajectory_states, trajectory_actions, trajectory_rewards, trajectory_dones, trajectory_policy_logits):
        for traj_count in range(n_trajs):
            print('Storing trajectory ' + str(traj_count) + ' into buffer')
            for step in range(collect_steps):
                buffer.store_to_trajectory(trajectory_states[traj_count, step, :], trajectory_actions[traj_count, step, :], trajectory_rewards[traj_count, step], \
                        trajectory_dones[traj_count, step], trajectory_policy_logits[traj_count, step, :])
            buffer.store_to_buffer()

    ## Method to sample action from actor network for the current observation
    #@tf.function
    def sample_action(self, actor, observation):
        observation_tensor = tf.convert_to_tensor(observation)
        observation_tensor = tf.expand_dims(observation_tensor, axis=0)
        logits = actor(observation_tensor, training=False)
        action = tf.squeeze(tf.random.categorical(logits, 1, seed=seed), axis=1).numpy()[0]
        if action == 280:
            print('Invalid action')
        logits_array = logits.numpy()[0]
        return logits_array, action

    ## Define method to compute cumulative discounted rewards and advantages
    def discounted_cumulative_sums(self, rewards, discount):
        sum = 0

        # Ensure that first value in discount array is 1
        for (current_reward, current_discount) in zip(rewards, discount):
            sum += current_discount*current_reward

        return sum

    ## Method to compute advantages and returns for the chosen trajectory
    def compute_advantages_and_returns(self, rewards, values, value_end, ad_norm):
        # gamma and lam are defined globally

        deltas = rewards[:-1] + gamma*values[1:] - values[:-1] # deltas used to compute advantages 
        deltas_array = np.append(deltas, rewards[-1] + gamma*value_end - values[-1]) # append the final delta computed using value_end

        discount_array = np.ones(len(rewards))
        advantage_discount_array = np.ones(len(rewards))
        for i in range(len(rewards)):
            discount_array[i] = gamma**i
            advantage_discount_array[i] = (lam*gamma)**i
        
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        for ind in range(len(rewards)):
            advantages[ind] = discounted_cumulative_sums(deltas_array[:(ind+1)], advantage_discount_array[:(ind+1)])
            returns[ind] = discounted_cumulative_sums(rewards[:(ind+1)], discount_array[:(ind+1)])

        if ad_norm:
            advantage_mean = np.mean(advantages)
            advantage_std = np.std(advantages)

            advantages = np.divide(np.subtract(advantages, advantage_mean), advantage_std)

        #self.trajectory_start_index = self.current_end_position

        return returns, advantages

    ## Train the policy network (actor) by maximizing PPO objective (possibly including clip and/or adaptive KL penalty coefficients and entropy regularization)
    #@tf.function
    def train_actor(self, actor, policy_optimizer, observation_samples, action_samples, logits_samples, advantage_samples, beta_val, target_kl, entropy_coeff):
        action_samples_array = tf.squeeze(action_samples).numpy()

        with tf.GradientTape() as tape:

            actor_predictions = actor(observation_samples, training=False)
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
                adaptive_kl_penalty = beta_val * kl_mean

                if kl_mean < (target_kl/1.5):
                    beta_val = beta_val/2
                elif kl_mean > (target_kl * 1.5):
                    beta_val = beta_val * 2

            # entropy bonus is added to discourage premature convergence of the policy and allow for more exploration (Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, 
            # Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." In International conference on machine learning, 
            # pp. 1928-1937. PMLR, 2016.)
            if use_entropy_bonus: 
                entropy = tf.reduce_mean(tf.reduce_sum(tf.multiply(probabilities_new, tf.math.log(probabilities_new)), axis=1)) 

            policy_loss = policy_loss_clipping - adaptive_kl_penalty - entropy_coeff*entropy
        
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        if np.any(tf.math.is_nan(policy_grads).numpy()):
            print('NaN actor gradients')
            
        policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        return policy_loss, kl_mean, beta_val

    ## Train the value network (critic) using MSE
    #@tf.function
    def train_critic(self, critic, value_optimizer, observation_samples, return_samples):
        with tf.GradientTape() as tape:
            critic_predictions = critic(observation_samples, training=False)
            value_loss = tf.reduce_mean(tf.math.pow(tf.math.subtract(return_samples, critic_predictions), 2))
        value_grads = tape.gradient(value_loss, critic.trainable_variables)
        if np.any(tf.math.is_nan(value_grads).numpy()):
            print('NaN critic gradients')
            
        value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

        return value_loss