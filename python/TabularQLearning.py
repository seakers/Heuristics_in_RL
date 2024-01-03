# -*- coding: utf-8 -*-
"""
Tabular Q-learning main script for both the metamaterial and satelite problems

@author: roshan94
"""
import gymnasium as gym
import math
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from agents.TabularQAgent import TabularQLearningAgent
from tqdm import tqdm

artery_prob = True # If true -> artery problem, false -> equal stiffness problem

rad = 250e-6 # in m
sel = 10e-3 # in m
E = 1.8162e6 # in Pa
sidenum = 5
nucFac = 3

obj_names = ['TrueObjective1','TrueObjective2']
heur_names = ['PartialCollapsibilityViolation','NodalPropertiesViolation','OrientationViolation','IntersectionViolation'] # make sure this is consistent with the order of the heuristic operators in the Java code
heurs_used = [False, False, False, False] # for [partial collapsibility, nodal properties, orientation, intersection]
constr_names = ['FeasibilityViolation','ConnectivityViolation']
if not artery_prob:
    constr_names.append('StiffnessRatioViolation')

c_target = 1
if artery_prob:
    c_target = 0.421

# find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = (2*n_states)+1 # number of actions = (2*number of design variables) + 1 (first n_states actions are adding the corresponding member, 
                       # next action is no change, next n_states actions are removing the corresponding member)

if artery_prob:
    env = ArteryProblemEnv(n_actions=n_actions, n_states=n_states, sel=sel, sidenum=sidenum, rad=rad, E=E, c_target=c_target,obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)
else:
    env = EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, sel=sel, sidenum=sidenum, rad=rad, E=E, c_target=c_target,obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)

# hyperparameters for the agent
learning_rate = 0.01
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon/(n_episodes/2)  # reduce the exploration over time
final_epsilon = 0.1

agent = TabularQLearningAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

# Training
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
