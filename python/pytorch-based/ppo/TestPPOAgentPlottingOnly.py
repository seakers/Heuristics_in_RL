# -*- coding: utf-8 -*-
"""
Plotting and creating a video of the saved execution steps by the trained agent

@author: roshan94
"""
import os
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

run_num = 0

dqn = False # if True -> test DQN network, False -> test PPO actor network
random_states_mode = False

max_steps = 100

## Get savepath from config file
f = open('pytorch-based\\ppo\\ppo-config.json')
alg_dir = "PPO-H\\"
data = json.load(f)

save_path = data["Savepath"]

new_reward = data["Use new problem formulation"]
include_weights = data["Include weights in state"]

## Load problem parameters from config file
problem_choice = 2 # 1 - Metamaterial problem, 2 - EOSS problem

match problem_choice:
    case 1:
        metamat_prob = True

        f_prob = open('.\\envs\\metamaterial\\problem-config.json')
        data_prob = json.load(f_prob)

        one_dec = data_prob["Use One Decision Environment"] # If true -> {problem}OneDecisionEnvironment.py, false -> {problem}ProblemEnvironment.py

        artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

        if artery_prob:
            print("Metamaterial - Artery Problem")
        else:
            print("Metamaterial - Equal Stiffness Problem")

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

        # c_target = data_prob["Target stiffness ratio"]
        c_target = 1
        if artery_prob:
            c_target = 0.421

        feas_c_target_delta = data_prob["Feasible stiffness delta"] # delta about target stiffness ratio defining satisfying designs

        render_steps = data_prob["Render steps"]

        ## find number of states and actions based on sidenum
        n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
        n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
        n_actions = 1
        
        plot_obj_names = ["$-C_{22}/E$","$v_f$"]
        if artery_prob:
            plot_obj_names = ["$-C_{11}/v_f$","deviation"]

        ## find number of states based on sidenum
        n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 

    case 2:
        f_prob = open('.\\envs\\eoss\\problem-config.json')
        data_prob = json.load(f_prob)

        assign_prob = data_prob["Solve assigning problem"] # If true -> assigning problem, false -> partitioning problem
        one_dec = data_prob["Use One Decision Environment"] # If true -> {problem}OneDecisionEnvironment.py, false -> {problem}ProblemEnvironment.py
        consider_feas = data_prob["Consider feasibility for architecture evaluator"] # Whether to consider design feasibility for evaluation (used for the Partitioning problem and always set to true)
        resources_path = data_prob["Resources Path"]

        obj_names = data_prob["Objective names"]
        heur_names = data_prob["Heuristic names"] # make sure this is consistent with the order of the heuristic operators in the Java code
        # [duty cycle violation, instrument orbit relations violation, instrument interference violation, packing efficiency violation, spacecraft mass violation, instrument synergy violation, instrument count violation(only for Assigning problem)]
        heur_abbr = data_prob["Heuristic abbreviations"]
        heurs_used = data_prob["Heuristics used"] 
        # in the order of heur_names
        n_heurs_used = heurs_used.count(True)

        objs_max = data_prob["Objective maximized"]

        dc_thresh = data_prob["Duty cycle threshold"]
        mass_thresh = data_prob["Spacecraft wet mass threshold (in kg)"]
        pe_thresh = data_prob["Packing efficiency threshold"]
        ic_thresh = data_prob["Instrument count threshold"] # Only for assignment problem

        render_steps = data_prob["Render steps"]

        if assign_prob:
            print("EOSS - Assigning Problem")
        else:
            print("EOSS - Partitioning Problem")
            print("TBD")

        metamat_prob = False
        artery_prob = False

        plot_obj_names = ["Science","Lifecycle Cost"]

    case _:
        print("Invalid problem choice")

current_save_path = os.path.join(save_path, "run " + str(run_num))

#print(device_lib.list_local_devices())

# Load Keras Network
nfe_agent_ext = "final" # final or "ep" + str(nfe)

file_name = "RL_execution_results_ppo_"
file_name += str(run_num)

if problem_choice == 1:
    if artery_prob:
        file_name += "_artery"
    else:
        file_name += "_eqstiff"
else:
    if assign_prob:
        file_name += "_assign"
    else:
        file_name += "_partition"

file_name += "_" + nfe_agent_ext

img_name = "agent_testing_"
file_name += ".csv"

artists = []
fig, ax = plt.subplots()

with tqdm(total=max_steps) as pbar:
    with open(os.path.join(current_save_path, file_name), mode='r') as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:

            if not one_dec:
                step = int(lines['Step Number'])
            else:
                step = int(lines['Trajectory'])
            #current_state = lines['State']

            reward = np.round(float(lines['Reward']), decimals=2)

            if step == 0:
                objs = np.zeros(len(obj_names))
                if metamat_prob:
                    prev_constrs = np.zeros(len(constr_names))
                else:
                    prev_constrs = []

                for i in range(len(obj_names)):
                    objs[i] = float(lines[obj_names[i]])

                if metamat_prob:
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
                if np.all(prev_constrs == 0) or (len(prev_constrs) == 0):
                    prev_color = 'green'
                else:
                    prev_color = 'red'

                reward_prev = reward

                ax.scatter(obj1_prev, obj2_prev, color=prev_color)   
                if not one_dec: 
                    ax.text(obj1_prev+1e-4*obj1_prev, obj2_prev+1e-4*obj2_prev, str(step))
                else:
                    obj_weight0 = np.round(float(lines['Objective Weight0']), decimals=2)
                    obj_weight1 = np.round(1.0 - obj_weight0, decimals=2)
                    ax.text(obj1_prev+1e-4*obj1_prev, obj2_prev+1e-4*obj2_prev, "Wts.: [" + str(obj_weight0) + ", " + str(obj_weight1) + "], r: " + str(reward), wrap=True)
                ax.set_xlabel(plot_obj_names[0], fontsize=12)
                ax.set_ylabel(plot_obj_names[1], fontsize=12)

                if not one_dec:
                    current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step) + '.jpeg')
                else:
                    current_img_name = os.path.join(current_save_path, img_name + 'traj' + str(step) + '.jpeg')

                fig.savefig(current_img_name)
                plt.close(fig)

                im = plt.imshow(plt.imread(current_img_name), animated=True)

                artists.append([im])
            else: # All other steps

                objs = np.zeros(len(obj_names))
                if metamat_prob:
                    new_constrs = np.zeros(len(constr_names))
                else:
                    new_constrs = []

                for i in range(len(obj_names)):
                    objs[i] = float(lines[obj_names[i]])

                if metamat_prob:
                    for j in range(len(constr_names)):
                        new_constrs[j] = float(lines[constr_names[j]])

                obj1_next = objs[0]
                obj2_next = objs[1]

                if math.isnan(obj1_next):
                    obj1_next = 0.0

                if np.all(new_constrs == 0) or (len(new_constrs) == 0):
                    new_color = 'green'
                else:
                    new_color = 'red'

                # Plot final state objectives
                ax.scatter(obj1_next, obj2_next, color=new_color)
                if not one_dec: 
                    ax.text(obj1_next+1e-4*obj1_next, obj2_next+1e-4*obj2_next, str(step))
                else:
                    reward = np.round(float(lines['Reward']), decimals=2)

                    obj_weight0 = np.round(float(lines['Objective Weight0']), decimals=2)
                    obj_weight1 = np.round(1.0 - obj_weight0, decimals=2)
                    ax.text(obj1_next+1e-4*obj1_next, obj2_next+1e-4*obj2_next, "Wts.: [" + str(obj_weight0) + ", " + str(obj_weight1) + "], r: " + str(reward), wrap=True)
                
                if not one_dec:
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

                if not one_dec:
                    current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step) + '.jpeg')
                else:
                    current_img_name = os.path.join(current_save_path, img_name + 'traj' + str(step) + '.jpeg')
                fig.savefig(current_img_name)
                plt.close(fig)

                if not one_dec:
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
