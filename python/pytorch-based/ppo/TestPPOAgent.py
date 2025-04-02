# -*- coding: utf-8 -*-
"""
Testing a trained DQN agent using keras functions instead of tf-agents

@author: roshan94
"""
import os
import sys
import json
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path
import torch 

print("Torch cuda available: ", torch.cuda.is_available())
print("Torch device: ", torch.cuda.get_device_name(0))
device = torch.device(0)
#device = torch.device("cpu")

from torch import nn
from torch.nn import ModuleList

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)

from save.resultsaving import ResultSaver

from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (Compose, RewardSum, TransformedEnv)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.distributions import OneHotCategorical
from envs.metamaterial.arteryproblemenv import ArteryProblemEnv
from envs.metamaterial.equalstiffnessproblemenv import EqualStiffnessProblemEnv
from envs.metamaterial.arteryonedecisionenv import ArteryOneDecisionEnv
from envs.metamaterial.equalstiffnessonedecisionenv import EqualStiffnessOneDecisionEnv
from envs.eoss.assignmentonedecisionenv import AssignmentOneDecisionEnv
from envs.eoss.assignmentproblemenv import AssignmentProblemEnv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

#plt.rcParams['animation.ffmpeg_path'] = 'C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\ffmpeg'

import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

use_plotting = False # Plot in this script if True, or else plot in a separate script (TestAgentPlottingOnly.py)
# TestAgentPlottingOnly.py is run on a different conda environment that contains ffmpeg to save a video of the explored states. 
# The script here stores the explored states in a csv file which is read in TestAgentPlottingOnly.py

run_num = 0

## Get savepath from config file
f = open('pytorch-based\\ppo\\ppo-config.json')
alg_dir = "PPO-H\\"
data = json.load(f)

save_path = data["Savepath"]

actor_fc_layer_params = data["Actor network layer units"]
actor_dropout_layer_params = data["Actor network dropout probabilities"]

new_reward = data["Use new problem formulation"]
include_weights = data["Include weights in state"]

## Define problem environment
problem_choice = 2 # 1 - Metamaterial problem, 2 - EOSS problem

match problem_choice:
    case 1:
        metamat_prob = True

        f_prob = open('.\\envs\\metamaterial\\problem-config.json')
        data_prob = json.load(f_prob)

        artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

        one_dec = data_prob["Use One Decision Environment"] # If true -> {problem}OneDecisionEnvironment.py, false -> {problem}ProblemEnvironment.py

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

    case _:
        print("Invalid problem choice")

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

# Load Keras Network
nfe_agent_ext = "final" # final or "ep" + str(nfe)
model_filename = "learned_actor_network_" + nfe_agent_ext
#seed = 25

if problem_choice == 1:
    if artery_prob:
        model_filename += "_artery"
    else:
        model_filename += "_eqstiff"
else:
    if assign_prob:
        model_filename += "_assign"
    else:
        model_filename += "_partition"
model_filename += ".pth"

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

#################################################################################################
########################################## Key Classes ##########################################
#################################################################################################

class ActorNetwork(nn.Module):

    def __init__(self, environ, hidden_layer_units, hidden_layer_dropout_probs):
        super().__init__()

        # Define layers for actor network
        self.params = ModuleList()
        #self.params = []

        #self.layers = []
        #self.activation_layers = []
        #self.dropout_layers = []

        #self.layers.append(nn.Linear(environ.observation_spec['observation'].shape[0], hidden_layer_units[0], device=device)) # Input layer
        #self.activation_layers.append(nn.ReLU())
        #self.dropout_layers.append(nn.Dropout(hidden_layer_dropout_probs[0]))

        if include_weights:
            self.params.append(nn.Linear(environ.observation_spec['design'].shape[0], hidden_layer_units[0], device=device)) # Input layer
        else:
            self.params.append(nn.Linear(environ.observation_spec['observation'].shape[0], hidden_layer_units[0], device=device)) # Input layer
        self.params.append(nn.ReLU())
        self.params.append(nn.Dropout(hidden_layer_dropout_probs[0]))

        for i in range(len(hidden_layer_units)-1): # Hidden layers
            #self.layers.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            #self.activation_layers.append(nn.ReLU())
            #self.dropout_layers.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))

            self.params.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            self.params.append(nn.ReLU())
            self.params.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))
        
        if one_dec:
            self.params.append(nn.Linear(hidden_layer_units[-1], 2, device=device)) # Output layer
        else:
            #self.output_layer = nn.Linear(hidden_layer_units[-1], environ.action_spec.shape[0], device=device) # Output layer
            self.params.append(nn.Linear(hidden_layer_units[-1], environ.action_spec.shape[0], device=device)) # Output layer
            #self.params.append(nn.Linear(hidden_layer_units[-1], n_actions, device=device)) # Output layer

    def forward(self, x):
        if one_dec:
            x = torch.argmax(x, dim=-1) # Dimension of one-hot decision encoding is squeezed
        x = x.to(torch.float32)
        # for layer, act_layer, drop_layer in zip(self.layers, self.activation_layers, self.dropout_layers):
        #     x = act_layer(layer(x))
        #     x = drop_layer(x)
        # return self.output_layer(x)
        for layer in self.params:
            x = layer(x)

        return x
    
#################################################################################################
######################################### Main Operation ########################################
#################################################################################################

# Initialize result saver
current_save_path = os.path.join(save_path, "run " + str(run_num))
if metamat_prob:
    result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)
else:
    result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=[], heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=0.0)

## Initialize environment
if problem_choice == 1:
    if artery_prob:
        if one_dec:
            base_env = GymWrapper(ArteryOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
        else:
            base_env = GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
    else:
        if one_dec: 
            base_env = GymWrapper(EqualStiffnessOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
        else:
            base_env = GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
else:
    if assign_prob:
        if one_dec:
            base_env = GymWrapper(AssignmentOneDecisionEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
            n_states = env.base_env.unwrapped.get_n_states()
        else:
            base_env = GymWrapper(AssignmentProblemEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), device=device, categorical_action_encoding=False)
            env = TransformedEnv(base_env, RewardSum())
            n_states = env.base_env.unwrapped.get_n_states()
    else:
        print("TBD")

check_env_specs(env)

learned_agent = ActorNetwork(env, actor_fc_layer_params, actor_dropout_layer_params)
learned_agent.load_state_dict(torch.load(os.path.join(current_save_path, model_filename), weights_only=True))

if include_weights:
    policy_module = TensorDictModule(learned_agent, in_keys=["design"], out_keys=["logits"])
else:
    policy_module = TensorDictModule(learned_agent, in_keys=["observation"], out_keys=["logits"])

policy_module = ProbabilisticActor(
module=policy_module,
spec=env.action_spec,
in_keys=["logits"],
distribution_class=OneHotCategorical,
return_log_prob=True,
)

policy_module.eval()

#current_state = env.reset()
explored_states = []

step_counter = 0
current_nfe_val = 0

if not one_dec:
    n_trajs = 1
else:
    n_trajs = 100 # This corresponds to the number of designs generated for the one decision environments

max_steps = 100
if one_dec:
    max_steps = n_states
        
traj_start = True

csv_rows = []
if use_plotting:
    artists = []
    fig, ax = plt.subplots()

with tqdm(total=n_trajs) as pbar_traj:
    pbar_traj.set_description("Trajectory")

    for traj_num in range(n_trajs):

        row = {}

        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad(): # ExplorationType.MEAN is nan for Categorical Distribution, ExplorationType.MODE chooses the action with the highest probability, ExplorationType.RANDOM samples the distribution for an action
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(max_steps, policy_module)

        if not one_dec:    
            with tqdm(total=max_steps) as pbar:
                while step_counter < max_steps:

                    row = {}
                    
                    row['Step Number'] = step_counter

                    print("step number = " + str(step_counter))

                    state_tensor = eval_rollout['observation'][step_counter].detach().cpu().numpy()
                    row['State'] = ''.join(map(str,state_tensor))
                    row['Action'] = eval_rollout['action'][step_counter].argmax().detach().cpu().numpy()

                    if not traj_start:
                        if np.array_equal(state_tensor, next_state):
                            print('True')
                        else:
                            print('False')

                    next_state = eval_rollout['next']['observation'][step_counter].detach().cpu().numpy() 
                    reward = eval_rollout['next']['reward'][step_counter].detach().cpu().numpy()[0]

                    true_objs, objs, constrs, heurs  = result_logger.evaluate_design(metamat_prob=metamat_prob, artery_prob=artery_prob, design=state_tensor)
                    next_true_objs, next_objs, next_constrs, next_heurs  = result_logger.evaluate_design(metamat_prob=metamat_prob, artery_prob=artery_prob, design=next_state)

                    ## Plotting current agent step
                    if np.all(constrs == 0) or (len(constrs) == 0):
                        prev_color = 'green'
                    else:
                        prev_color = 'red'

                    if np.all(next_constrs == 0) or (len(next_constrs) == 0):
                        new_color = 'green'
                    else:
                        new_color = 'red'

                    # Plot initial state objectives
                    obj1_prev = true_objs[0]
                    obj2_prev = true_objs[1]

                    obj1_next = next_true_objs[0]
                    obj2_next = next_true_objs[1]

                    # Deal with nan first objective (in metamaterial design problems)
                    # Since the first objective (stiffness related) must be maximized, a non-nan design will have negative first objective values
                    if math.isnan(obj1_prev):
                        obj1_prev = 0.0

                    if math.isnan(obj1_next):
                        obj1_next = 0.0

                    if traj_start:
                        if use_plotting:
                            ax.scatter(obj1_prev, obj2_prev, color=prev_color)    
                            ax.text(obj1_prev+1e-4*obj1_prev, obj2_prev+1e-4*obj2_prev, str(step_counter))

                            current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step_counter) + '.png')
                            fig.savefig(current_img_name)
                            plt.close(fig)

                            artists.append([plt.imshow(plt.imread(current_img_name))])

                        pbar.update(1)

                    # Plot final state objectives
                    if use_plotting:
                        ax.scatter(obj1_next, obj2_next, color=new_color)
                        ax.text(obj1_next+1e-4*obj1_next, obj2_next+1e-4*obj2_next, str(step_counter))

                    # Plot arrow from initial to final state
                    if use_plotting:
                        dx = obj1_next - obj1_prev
                        dy = obj2_next - obj2_prev
                        ax.arrow(obj1_prev, obj2_prev, dx, dy, width=1e-5)

                    # Add reward text at the center of the line
                    if use_plotting:
                        x_r = (obj1_next + obj1_prev)/2
                        y_r = (obj2_next + obj2_prev)/2
                        text_obj = ax.text(x_r, y_r, 'r='+str(reward))

                        current_img_name = os.path.join(current_save_path, img_name + 'step' + str(step_counter) + '.png')
                        fig.savefig(current_img_name)
                        plt.close(fig)

                        text_obj.set_visible(False)

                        artists.append([plt.imshow(plt.imread(current_img_name))])

                    row['Reward'] = reward

                    for i in range(len(obj_names)):
                        row[obj_names[i]] = true_objs[i]

                    if metamat_prob:
                        for j in range(len(constr_names)):
                            row[constr_names[j]] = constrs[j]

                    for k in range(len(heur_names)):
                        row[heur_names[k]] = heurs[k]
                    
                    csv_rows.append(row)

                    step_counter += 1
                    pbar.update(1)
                    traj_start = False

        else:

            end_step = step_counter + eval_rollout['done'].shape[0]
            row['Trajectory'] = traj_num
            row['Objective Weight0'] = eval_rollout['objective weight0'][-1].detach().cpu().numpy()[0]

            print("step number = " + str(end_step))

            state_tensor = eval_rollout['next']['design'][-1,:,:].detach().cpu().numpy().argmax(axis=1).astype(np.int32)
            row['State'] = ''.join(map(str,state_tensor))
            row['Action'] = eval_rollout['action'][-1].argmax().detach().cpu().numpy()
            reward = eval_rollout['next']['reward'][-1].detach().cpu().numpy()[0]

            true_objs, objs, constrs, heurs  = result_logger.evaluate_design(metamat_prob=metamat_prob, artery_prob=artery_prob, design=state_tensor)

            ## Plotting current agent step
            if np.all(constrs == 0):
                prev_color = 'green'
            else:
                prev_color = 'red'

            # Plot initial state objectives
            obj1 = true_objs[0]
            obj2 = true_objs[1]

            # Deal with nan first objective (in metamaterial design problems)
            # Since the first objective (stiffness related) must be maximized, a non-nan design will have negative first objective values
            if math.isnan(obj1):
                obj1 = 0.0
                
            # Add reward text next to design
            if use_plotting:
                ax.scatter(obj1, obj2, color=prev_color) 
                text_obj = ax.text(obj1+1e-4*obj1, obj2+1e-4*obj2, 'r='+str(reward))

                current_img_name = os.path.join(current_save_path, img_name + 'traj' + str(traj_num) + '.png')
                fig.savefig(current_img_name)
                plt.close(fig)

                artists.append([plt.imshow(plt.imread(current_img_name))])

            row['Reward'] = reward

            for i in range(len(obj_names)):
                row[obj_names[i]] = true_objs[i]

            if metamat_prob:
                for j in range(len(constr_names)):
                    row[constr_names[j]] = constrs[j]

            for k in range(len(heur_names)):
                row[heur_names[k]] = heurs[k]
            
            csv_rows.append(row)

        pbar_traj.update(1)

# Save animation
if use_plotting:
    ani = animation.ArtistAnimation(fig, artists, interval=100, blit=True)
    ani.save(os.path.join(current_save_path, 'trained-agent-testing.png'), writer="pillow")

    plt.xlabel(plot_obj_names[0], fontsize=12)
    plt.ylabel(plot_obj_names[1], fontsize=12)
    plt.savefig(os.path.join(current_save_path, 'agent_test_steps_' + str(nfe_agent_ext) + '.png'), dpi=600)

# write to csv
if one_dec:
    field_names = ['Trajectory', 'State', 'Action', 'Reward', 'Objective Weight0']
else:
    field_names = ['Step Number', 'State', 'Action', 'Reward']
field_names.extend(obj_names)
if metamat_prob:
    field_names.extend(constr_names)
field_names.extend(heur_names)

with open(os.path.join(current_save_path, file_name), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names, lineterminator = '\n')

    writer.writeheader()

    writer.writerows(csv_rows)
