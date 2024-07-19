# -*- coding: utf-8 -*-
"""
Class for the PPO Buffer to store trajectories
Reference: https://keras.io/examples/rl/ppo_cartpole/

@author: roshan94
"""
import numpy as np

class Buffer:

    # Class instantiation method
    def __init__(self, discrete_actions, max_buffer_size, max_traj_steps, num_actions, observation_dimensions, action_dimensions):

        self.discrete_actions = discrete_actions
        self.max_buffer_size = max_buffer_size
        
        self.max_trajectory_steps = max_traj_steps
        self.obs_dimensions = observation_dimensions
        self.act_dimensions = action_dimensions
        self.n_actions = num_actions
        
        self.observation_buffer = np.zeros((max_buffer_size, max_traj_steps, observation_dimensions), dtype=np.int32)
        self.action_buffer = np.zeros((max_buffer_size, max_traj_steps, num_actions))
        self.reward_buffer = np.zeros((max_buffer_size, max_traj_steps))
        self.done_buffer = np.zeros((max_buffer_size, max_traj_steps), dtype=np.int32)

        self.current_trajectory_observations = np.zeros((max_traj_steps, observation_dimensions), dtype=np.int32)
        self.current_trajectory_actions = np.zeros((max_traj_steps, num_actions))
        self.current_trajectory_rewards = np.zeros(max_traj_steps)
        self.current_trajectory_dones = np.zeros(max_traj_steps, dtype=np.int32)
        
        self.current_traj_end_position = 0
        self.last_trajectory_index = 0
        self.num_trajectories = 0

        if self.discrete_actions:
            self.policy_logits_buffer = np.zeros((max_buffer_size, max_traj_steps, num_actions*action_dimensions), dtype=np.float32) # for discrete actions, the logits are directly converted to probabilities
            self.current_trajectory_policy_logits = np.zeros((max_traj_steps, num_actions*action_dimensions), dtype=np.float32)
        else:
            self.policy_logits_buffer = np.zeros((max_buffer_size, max_traj_steps, 2 * num_actions), dtype=np.float32) # for continuous actions, the probability distribution for an action 
            # is taken as gaussian with two logits representing the mean and variance respectively
            self.current_trajectory_policy_logits = np.zeros((max_traj_steps, 2 * num_actions), dtype=np.float32)

    # Method to store one step in the trajectory 
    def store_to_trajectory(self, observation, action, reward, done, logits):
        self.current_trajectory_observations[self.current_traj_end_position, :] = observation
        self.current_trajectory_actions[self.current_traj_end_position, :] = action
        self.current_trajectory_rewards[self.current_traj_end_position] = reward
        self.current_trajectory_policy_logits[self.current_traj_end_position, :] = logits
        self.current_trajectory_dones[self.current_traj_end_position] = done
        self.current_traj_end_position += 1

    # Method to end current trajectory and store it to the buffer
    def store_to_buffer(self):
        self.observation_buffer[self.last_trajectory_index,:,:] = self.current_trajectory_observations
        self.action_buffer[self.last_trajectory_index,:,:] = self.current_trajectory_actions
        self.reward_buffer[self.last_trajectory_index,:] = self.current_trajectory_rewards
        self.done_buffer[self.last_trajectory_index,:] = self.current_trajectory_dones

        self.policy_logits_buffer[self.last_trajectory_index,:,:] = self.current_trajectory_policy_logits

        self.last_trajectory_index += 1
        self.num_trajectories += 1
        
        self.reset_current_trajectory()

        # Remove earliest trajectory if buffer is full
        if self.num_trajectories == self.max_buffer_size:
            self.observation_buffer[0, :, :] = np.zeros((self.max_trajectory_steps, self.obs_dimensions))
            self.action_buffer[0, :, :] = np.zeros((self.max_trajectory_steps, self.n_actions))
            self.reward_buffer[0, :] = np.zeros(self.max_trajectory_steps)
            self.done_buffer[0, :] = np.zeros(self.max_trajectory_steps)
            if self.discrete_actions:
                self.policy_logits_buffer[0, :, :] = np.zeros((self.max_trajectory_steps, self.act_dimensions))
            else:
                self.policy_logits_buffer[0, :, :] = np.zeros((self.max_trajectory_steps, 2 * self.act_dimensions))
            
            self.last_trajectory_index = 0

    def reset_current_trajectory(self):

        # Reset current trajectory observations, actions and rewards
        self.current_trajectory_observations = np.zeros((self.max_trajectory_steps, self.obs_dimensions))
        self.current_trajectory_actions = np.zeros((self.max_trajectory_steps, self.n_actions))
        self.current_trajectory_rewards = np.zeros(self.max_trajectory_steps)
        self.current_trajectory_dones = np.zeros(self.max_trajectory_steps)

        if self.discrete_actions:
            self.current_trajectory_policy_logits = np.zeros((self.max_trajectory_steps, self.act_dimensions))
        else:
            self.current_trajectory_policy_logits = np.zeros((self.max_trajectory_steps, 2 * self.act_dimensions))

        self.current_traj_end_position = 0

    def get_num_trajectories(self):
        return self.num_trajectories

    # Method to get stats on the chosen trajectory (include last observation to compute delta for the last time step)
    def get_trajectory_slice(self, trajectory_index, start_index=0, end_index=1):
        return self.observation_buffer[trajectory_index, start_index:(end_index+1), :], \
            self.action_buffer[trajectory_index, start_index:end_index, :], \
            self.reward_buffer[trajectory_index, start_index:end_index], \
            self.done_buffer[trajectory_index, start_index:end_index], \
            self.policy_logits_buffer[trajectory_index, start_index:end_index, :]
    