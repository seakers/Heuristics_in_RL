# -*- coding: utf-8 -*-
"""
Plotting and creating a video of the saved execution steps by the trained agent

@author: roshan94
"""
import os
os.environ["KERAS BACKEND"] = "tensorflow"
import sys

import json

import numpy as np

import csv

from tqdm import tqdm

from time import sleep

from pathlib import Path

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[0]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

import math 

run_num = 4

dqn = False # if True -> test DQN network, False -> test PPO actor network
random_states_mode = False

max_steps = 100

## Get savepath from config file
if dqn:
    f = open('keras-based\\dqn\\dqn-config.json')
    alg_dir = "dqn\\"
else:
    f = open('keras-based\\ppo\\ppo-config.json')
    alg_dir = "PPO-H\\"
data = json.load(f)

save_path = data["Savepath"]

new_reward = data["Use new problem formulation"]
include_weights = data["Include weights in state"]

## Load problem parameters from config file
f_prob = open('keras-based\\problem-config.json')
data_prob = json.load(f_prob)

artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

if artery_prob:
    prob_dir = "Artery\\"
else:
    prob_dir = "EqStiff\\"

current_save_path = save_path + prob_dir + alg_dir + "run " + str(run_num) + "\\"

#print(device_lib.list_local_devices())

sidenum = data_prob["Lattice number of side nodes"]

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

plot_obj_names = ["$-C_{22}/E$","$v_f$"]
if artery_prob:
    plot_obj_names = ["$-C_{11}/v_f$","deviation"]

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
n_actions = 1

# Load Keras Network
nfe_agent_ext = "final" # final or "ep" + str(nfe)

if dqn:
    file_name = "RL_execution_results_dqnQ_" 
else:
    file_name = "RL_execution_results_ppo_"

if random_states_mode:
    file_name += "rand" + str(run_num)
else:
    file_name += str(run_num)

if artery_prob:
    file_name += "_artery"
else:
    file_name += "_eqstiff"

file_name += "_" + nfe_agent_ext

img_name = "agent_testing_"
file_name += ".csv"

artists = []
fig, ax = plt.subplots()

with tqdm(total=max_steps) as pbar:
    with open(current_save_path + file_name, mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:

            step = int(lines['Step Number'])
            #current_state = lines['State']

            reward = float(lines['Reward'])

            if step == 0:
                objs = np.zeros(len(obj_names))
                prev_constrs = np.zeros(len(constr_names))

                for i in range(len(obj_names)):
                    objs[i] = float(lines[obj_names[i]])

                for j in range(len(constr_names)):
                    prev_constrs[j] = float(lines[constr_names[j]])

                # Plot initial state objectives
                obj1_prev = objs[0]
                obj2_prev = objs[1]

                # Deal with nan first objective (in metamaterial design problems)
                # Since the first objective (stiffness related) must be maximized, a non-nan design will have negative first objective values
                if math.isnan(obj1_prev):
                    obj1_prev = 0.0

                ## Plotting current agent step
                if np.all(prev_constrs == 0):
                    prev_color = 'green'
                else:
                    prev_color = 'red'

                reward_prev = reward

                ax.scatter(obj1_prev, obj2_prev, color=prev_color)    
                ax.text(obj1_prev+1e-4*obj1_prev, obj2_prev+1e-4*obj2_prev, str(step))
                ax.set_xlabel(plot_obj_names[0], fontsize=12)
                ax.set_ylabel(plot_obj_names[1], fontsize=12)

                current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step) + '.jpeg')
                fig.savefig(current_img_name)
                plt.close(fig)

                im = plt.imshow(plt.imread(current_img_name), animated=True)

                artists.append([im])
            else: # All other steps

                objs = np.zeros(len(obj_names))
                new_constrs = np.zeros(len(constr_names))

                for i in range(len(obj_names)):
                    objs[i] = float(lines[obj_names[i]])

                for j in range(len(constr_names)):
                    new_constrs[j] = float(lines[constr_names[j]])

                # Plot initial state objectives
                obj1_next = objs[0]
                obj2_next = objs[1]

                if math.isnan(obj1_next):
                    obj1_next = 0.0

                if np.all(new_constrs == 0):
                    new_color = 'green'
                else:
                    new_color = 'red'

                # Plot final state objectives
                ax.scatter(obj1_next, obj2_next, color=new_color)
                ax.text(obj1_next+1e-4*obj1_next, obj2_next+1e-4*obj2_next, str(step))
                
                # Plot arrow from initial to final state
                dx = obj1_next - obj1_prev
                dy = obj2_next - obj2_prev
                ax.arrow(obj1_prev, obj2_prev, dx, dy, width=1e-5)

                # Add reward text at the center of the line
                x_r = (obj1_next + obj1_prev)/2
                y_r = (obj2_next + obj2_prev)/2
                text_obj = ax.text(x_r, y_r, 'r='+str(reward_prev)) 
                ax.set_xlabel(plot_obj_names[0], fontsize=12)
                ax.set_ylabel(plot_obj_names[1], fontsize=12)

                current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step) + '.jpeg')
                fig.savefig(current_img_name)
                plt.close(fig)

                text_obj.set_visible(False)

                im = plt.imshow(plt.imread(current_img_name), animated=True)

                artists.append([im])

                obj1_prev = obj1_next
                obj2_prev = obj2_next

                reward_prev = reward

            pbar.update(1)

# Save animation
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
#sleep(0.05)
#plt.show()
#writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
#ani.save(os.path.join(current_save_path, 'trained-agent-testing.mp4'), writer=writer)
ani.save(os.path.join(current_save_path, 'trained-agent-testing.mp4'), writer='ffmpeg')
