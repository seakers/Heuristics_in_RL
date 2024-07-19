# -*- coding: utf-8 -*-
"""
Tabular Q-learning main script for both the metamaterial and satelite problems
Q-learning agent taken from tutorial example in https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/

@author: roshan94
"""
#import gymnasium as gym
import gym
import math
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from agents.TabularQAgent import TabularQLearningAgent
from tqdm import tqdm
import pickle

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"
file_name = "RL_training_designs_tabQ"

if artery_prob:
    file_name += "_artery"
else:
    file_name += "_eqstiff"
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

render_steps = False

# find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = (2*n_states) + n_heurs_used # number of actions = (2*number of design variables) (first n_states actions are adding the corresponding member, 
                       # last n_states actions are removing the corresponding member)
max_steps = 420

if artery_prob:
    env = ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
else:
    env = EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)

# hyperparameters for the agent
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1
epsilon_decay = start_epsilon/(n_episodes/2)  # reduce the exploration over time
final_epsilon = 0.01

agent = TabularQLearningAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

# Training
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    #print("Episode " + str(episode))
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        #print("Step number " + str(env.get_step_counter()))

        # update the agent
        agent.update(obs, action, reward, done, next_obs)

        # update if the environment is done and the current obs
        #done = terminated or truncated # deprecated version of gym directly outputs ``done''
        obs = next_obs

    agent.decay_epsilon()

# Save results to the file
env.get_support_result_logger().save_to_csv()

# Save learned Q-table
pickle_file_name = "Q-table" 
if artery_prob:
    pickle_file_name += "_artery"
else:
    pickle_file_name += "_eqstiff"
pickle_file_name += ".pkl"

q_file = open(save_path + pickle_file_name, 'ab')
pickle.dump(agent.get_q_table_dict(), q_file)
q_file.close()