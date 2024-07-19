# -*- coding: utf-8 -*-
"""
Tabular Q-learning Agent class for both the metamaterial and satelite problems
Q-learning agent taken from tutorial example in https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/

@author: roshan94
"""
from collections import defaultdict
import numpy as np

class TabularQLearningAgent:
    def __init__(self, env, learning_rate:float, initial_epsilon:float, epsilon_decay:float, final_epsilon:float, discount_factor:float=0.95,):
        #Initialize a Reinforcement Learning agent with an empty dictionary of state-action values (q_values), a learning rate and an epsilon.

        #Args:
            #env: Either ArteryProblemEnv or EqualStiffnesssProblemEnv environment instance
            #learning_rate: The learning rate
            #initial_epsilon: The initial epsilon value
            #epsilon_decay: The decay for epsilon
            #final_epsilon: The final epsilon value
            #discount_factor: The discount factor for computing the Q-value
        
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs) -> int:
        # Returns the best action with probability (1 - epsilon) otherwise a random action with probability epsilon to ensure exploration.
        
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return np.argmax(self.q_values[tuple(obs)])
        
    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        #Updates the Q-value of an action
        future_q_value = (not terminated) * np.max(self.q_values[tuple(next_obs)])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[tuple(obs)][action]
        )

        self.q_values[tuple(obs)][action] = (
            self.q_values[tuple(obs)][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_q_table_dict(self):
        return dict(self.q_values)