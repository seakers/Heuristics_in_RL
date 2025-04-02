# -*- coding: utf-8 -*-
"""
Training and saving a PPO agent using PyTorch methods - runs on multiple GPUs and separates the actor and critic updates on different GPUs
Reference: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html

TODO:
1. Plotting
2. General debugging

@author: roshan94
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Add path for parent directory with environment classes
import sys
import json
from pathlib import Path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = str(Path(current_path).resolve().parents[1]) # parents[i] is the i-th parent from the current directory
sys.path.append(parent_path)
from itertools import chain
from copy import deepcopy

from collections import defaultdict

from tqdm import tqdm

import matplotlib.pyplot as plt
import torch 
torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tensordict.nn import TensorDictModule
from torch import nn
from torch.nn import ModuleList
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributions import OneHotCategorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, RewardSum, TransformedEnv)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.libs.gym import GymWrapper
from tqdm import tqdm

print("Torch cuda available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        #device = torch.device(0)
        print("Torch device: ", torch.cuda.get_device_name(i))
else:
    print("Torch device: cpu")
    #device = torch.device("cpu")

from envs.metamaterial.arteryproblemenv import ArteryProblemEnv
from envs.metamaterial.arteryonedecisionenv import ArteryOneDecisionEnv
from envs.metamaterial.equalstiffnessproblemenv import EqualStiffnessProblemEnv
from envs.metamaterial.equalstiffnessonedecisionenv import EqualStiffnessOneDecisionEnv
from envs.eoss.assignmentonedecisionenv import AssignmentOneDecisionEnv
from envs.eoss.assignmentproblemenv import AssignmentProblemEnv
from save.resultsaving import ResultSaver

import numpy as np
import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import csv

import time

#################################################################################################
######################################## Parameter Setup ########################################
#################################################################################################

## Setup and train parameters from config file
start_time = time.time()

f_ppo = open('.\\pytorch-based\\ppo\\ppo-config.json')
data_ppo = json.load(f_ppo)

n_runs = data_ppo["Number of runs"]

gamma = data_ppo["Value discount (gamma)"] # discount factor
original_max_train_episodes = data_ppo["Number of training episodes"] # number of training episodes

max_eval_steps = data_ppo["Maximum steps in evaluation episode"] # termination in evaluation environment
episode_training_trajs = data_ppo["Number of trajectories used for training per episode"] # number of trajectories sampled in each iteration to train the actor and critic

max_unique_nfe_run = data_ppo["Maximum unique NFE"]

use_buffer = data_ppo["Buffer used"]

eval_interval = data_ppo["Episode interval for evaluation"] # After how many episodes is the actor being evaluated
new_reward = data_ppo["Use new problem formulation"]
include_weights = data_ppo["Include weights in state"]

sample_minibatch = data_ppo["Sample minibatch"] # Whether to sample minibatch or use the entire set of generated trajectories
trajectory_collect_steps = data_ppo["Number of steps in a collected trajectory"] # number of steps in each trajectory
minibatch_steps = data_ppo["Number of steps in a minibatch"]
replay_buffer_capacity = data_ppo["Replay buffer capacity"] # maximum number of trajectories that can be stored in the buffer

if sample_minibatch:
    if minibatch_steps > trajectory_collect_steps:
        print("Number of steps in a minibatch is greater than the number of collected steps, reduce the minibatch steps")
        sys.exit(0)
    
## NOTE: Total number of designs used for training in each run = episode_training_trajs*minibatch_steps*max_train_episodes

advantage_norm = data_ppo["Normalize advantages"] # whether to normalize advantages for training
discrete_actions = data_ppo["Discrete actions"]
lam = data_ppo["Advantage discount (lambda)"] # advantage discount factor

actor_fc_layer_params = data_ppo["Actor network layer units"]
actor_dropout_layer_params = data_ppo["Actor network dropout probabilities"]

critic_fc_layer_params = data_ppo["Critic network layer units"]
critic_dropout_layer_params = data_ppo["Critic network dropout probabilities"]

## NOTE: At least clipping or adaptive KL penalty must be used
clip_ratio = data_ppo["Clipping ratio threshold"]

use_entropy_bonus = data_ppo["Use entropy loss bonus"] # Use additional entropy of actor distribution term in actor loss
ent_coeff = data_ppo["Entropy coefficient"] # coefficient for the entropy bonus term in actor loss
critic_loss_coeff = data_ppo["Critic loss coefficient"]

clip_gradient_norm = data_ppo["Clip gradient norm"]
max_grad_norm = data_ppo["Max gradient norm"]

train_epochs = data_ppo["Number of training epochs"]

initial_learning_rate = data_ppo["Initial training learning rate"]
decay_rate = data_ppo["Learning rate decay rate"]
decay_steps = data_ppo["Learning rate decay steps"]

alpha = data_ppo["RMSprop optimizer velocity discount rate"]
momentum = data_ppo["RMSprop optimizer momentum"]

compute_periodic_returns = data_ppo["Compute periodic returns"]
use_continuous_minibatch = data_ppo["Continuous minibatch"] # use continuous trajectory slices to generate minibatch for training or random transitions from all trajectories
network_save_intervals = data_ppo["Episode interval to save actor and critic networks"]

save_path = data_ppo["Savepath"]

if network_save_intervals > original_max_train_episodes:
    print("Episode interval to save networks is greater than number of training episodes")
    sys.exit(0)

## Define problem environment
problem_choice = 2 # 1 - Metamaterial problem, 2 - EOSS problem

match problem_choice:
    case 1:
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

        feas_c_target_delta = data_prob["Feasible stiffness delta"] # delta about target stiffness ratio defining satisfying designs (only for equal stiffness problem)

        render_steps = data_prob["Render steps"]

        ## find number of states and actions based on sidenum
        n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
        n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
        n_actions = 1

        metamat_prob = True

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

n_episodes_per_fig = 4 # used for plotting returns and losses 
linestyles = ['solid','dotted','dashed','dashdot']

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()   

#################################################################################################
########################################## Key Classes ##########################################
#################################################################################################

class ActorNetwork(nn.Module):

    def __init__(self, environ, hidden_layer_units, hidden_layer_dropout_probs, device_name):
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
            self.params.append(nn.Linear(environ.observation_spec['design'].shape[0], hidden_layer_units[0], device=device_name)) # Input layer
        else:
            self.params.append(nn.Linear(environ.observation_spec['observation'].shape[0], hidden_layer_units[0], device=device_name)) # Input layer
        self.params.append(nn.ReLU())
        self.params.append(nn.Dropout(hidden_layer_dropout_probs[0]))

        for i in range(len(hidden_layer_units)-1): # Hidden layers
            #self.layers.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            #self.activation_layers.append(nn.ReLU())
            #self.dropout_layers.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))

            self.params.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device_name))
            self.params.append(nn.ReLU())
            self.params.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))
        
        if one_dec:
            self.params.append(nn.Linear(hidden_layer_units[-1], 2, device=device_name)) # Output layer
        else:
            #self.output_layer = nn.Linear(hidden_layer_units[-1], environ.action_spec.shape[0], device=device) # Output layer
            self.params.append(nn.Linear(hidden_layer_units[-1], environ.action_spec.shape[0], device=device_name)) # Output layer
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

class CriticNetwork(nn.Module):

    ## ISSUE: Dropout does not work in the critic network somehow, vmap_randomness flag in the loss module cannot be set for the ClipPPOLoss object currently

    def __init__(self, environ, hidden_layer_units, hidden_layer_dropout_probs, device_name):
        super().__init__()

        # Define layers for critic network
        self.params = ModuleList()

        #self.layers = []
        #self.activation_layers = []
        #self.dropout_layers = []

        #self.layers.append(nn.Linear(environ.observation_spec['observation'].shape[0], hidden_layer_units[0], device=device)) # Input layer
        #self.activation_layers.append(nn.ReLU())
        if include_weights:
            self.params.append(nn.Linear(environ.observation_spec['design'].shape[0], hidden_layer_units[0], device=device_name)) # Input layer
        else:
            self.params.append(nn.Linear(environ.observation_spec['observation'].shape[0], hidden_layer_units[0], device=device_name)) # Input layer
        self.params.append(nn.ReLU())

        #self.dropout_layers.append(nn.Dropout(hidden_layer_dropout_probs[0]))
        for i in range(len(hidden_layer_units)-1): # Hidden layers
            #self.layers.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            #self.activation_layers.append(nn.ReLU())
            #self.dropout_layers.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))

            self.params.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device_name))
            self.params.append(nn.ReLU())
            #self.params.append(nn.Dropout(hidden_layer_dropout_probs[i+1]))
        
        #self.output_layer = nn.Linear(hidden_layer_units[-1], 1, device=device) # Output layer
        self.params.append(nn.Linear(hidden_layer_units[-1], 1, device=device_name)) # Output layer

    def forward(self, x):
        if one_dec:
            x = torch.argmax(x, dim=-1) # Dimension of one-hot decision encoding is squeezed
        x = x.to(torch.float32)

        # for layer, act_layer, drop_layer in zip(self.layers, self.activation_layers, self.dropout_layers):
        #     x = act_layer(layer(x))
        #     x = drop_layer(x)
        # for layer, act_layer in zip(self.layers, self.activation_layers):
        #     x = act_layer(layer(x))

        # return self.output_layer(x)

        for layer in self.params:
            x = layer(x)

        return x
    
    def estimate_value(self, x):
        return self.forward(x)

#################################################################################################
######################################## Worker Function ########################################
#################################################################################################

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def training_function(rank, world_size, data_tensordict, include_weights, action_spec, optim, train_epochs, episode, network_save_intervals, current_save_path):

    setup(rank=rank, world_size=world_size)

    dataloader = DataLoader(data_tensordict, batch_size=int(data_tensordict['next']['reward'].shape[0]/world_size), shuffle=False, collate_fn=lambda x: x, sampler=DistributedSampler(data_tensordict))

    print('Training in device: ', rank)
    print('Batch size: ', int(data_tensordict['next']['reward'].shape[0]/world_size)) 

    ## Initialize actor and critic with loaded weights of previously trained networks
    actor = ActorNetwork(train_env, actor_fc_layer_params, actor_dropout_layer_params)
    critic = CriticNetwork(train_env, critic_fc_layer_params, critic_dropout_layer_params)
    actor.load_state_dict(torch.load(os.path.join(current_save_path, model_filename), weights_only=True))
    critic.load_state_dict(torch.load(os.path.join(current_save_path, model_filename), weights_only=True))

    actor.to(rank)
    critic.to(rank)

    actor_ddp = DDP(actor, device_ids=[rank])
    critic_ddp = DDP(critic, device_ids=[rank])     


    if include_weights:
        policy_module = TensorDictModule(actor_ddp, in_keys=["design"], out_keys=["logits"])
    else:
        policy_module = TensorDictModule(actor_ddp, in_keys=["observation"], out_keys=["logits"])

    policy_module = ProbabilisticActor(
    module=policy_module,
    spec=action_spec,
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
    )

    if include_weights:
        value_module = ValueOperator(
        module=critic_ddp,
        in_keys=["design"],
        )
    else:
        value_module = ValueOperator(
        module=critic_ddp,
        in_keys=["observation"],
        )

    advantage_module = GAE(gamma=gamma, lmbda=lam, value_network=value_module, average_gae=True)

    loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_ratio,
    entropy_bonus=use_entropy_bonus,
    entropy_coef=ent_coeff,
    # these keys match by default but we set this for completeness
    critic_coef=critic_loss_coeff,
    loss_critic_type="l2",
    )
    #loss_module.set_vmap_randomness("same")

    policy_module.to(rank)
    value_module.to(rank)
    advantage_module.to(rank)

    # we now have a batch of data to work with. Let's learn something from it.
    losses_data = []
    for epoch in range(train_epochs):
        optim.zero_grad()

        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(dataloader)
        
        loss_vals = loss_module(dataloader.to(rank))
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        if torch.isnan(loss_vals["loss_objective"]) or torch.isnan(loss_vals["loss_critic"]) or torch.isnan(loss_vals["loss_entropy"]):
            print('Check')

        print('Device rank: ', rank, ' Epoch: ', epoch)
        print('Device rank: ', rank, ' Actor clipping loss: ', loss_vals["loss_objective"].detach().cpu().numpy())
        print('Device rank: ', rank, ' Learning rate: ', optim.param_groups[0]["lr"])
        print('Device rank: ', rank, ' Critic loss: ', loss_vals["loss_critic"].detach().cpu().numpy())
        print('Device rank: ', rank, ' Entropy bonus: ', loss_vals["loss_entropy"].detach().cpu().numpy())
        print('Device rank: ', rank, ' Critic coef.: ', critic_loss_coeff)
        print('Device rank: ', rank, ' Entropy coef.:', ent_coeff)
        print('\n')

        # Optimization: backward, grad clipping and optimization step
        loss_value.backward()
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        optim.step()

        losses_data.append([epoch, loss_vals["loss_objective"].detach().cpu().numpy(), loss_vals["loss_entropy"].detach().cpu().numpy(), loss_vals["loss_critic"].detach().cpu().numpy()])

    loss_data_savepath = os.path.join(current_save_path, "losses_device" + str(rank) + "_episode" + episode + ".csv")
    loss_data_fields = ["Epoch", "Actor Clipping Loss", "Actor Entropy Loss", "Critic Loss"]

    with open(loss_data_savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(loss_data_fields)
        writer.writerows(losses_data)
    
    # mean_clipping_loss = np.mean(actor_clip_loss)
    # mean_entropy_loss = np.mean(actor_entropy_loss)
    # mean_critic_loss = np.mean(critic_loss)

    # mean_reward = tensordict_data["next", "reward"].mean().item()

    ## Save underlying actor and critic networks 





    # Save current trained actor and critic networks at regular intervals
    if (rank == 0) and (episode % network_save_intervals == 0):
        actor_model_filename = "learned_actor_network_ep" + str(episode)
        if problem_choice == 1:
            if artery_prob:
                actor_model_filename += "_artery"
            else:
                actor_model_filename += "_eqstiff"
        else:
            if assign_prob:
                actor_model_filename += "_assign"
            else:
                actor_model_filename += "_partition"
        
        actor_model_filename += ".pth"

        torch.save(actor_ddp.module.state_dict(), os.path.join(current_save_path, actor_model_filename))

        critic_model_filename = "learned_critic_network_ep" + str(episode)
        if problem_choice == 1:
            if artery_prob:
                critic_model_filename += "_artery"
            else:
                critic_model_filename += "_eqstiff"
        else:
            if assign_prob:
                critic_model_filename += "_assign"
            else:
                critic_model_filename += "_partition"
                
        critic_model_filename += ".pth"

        torch.save(critic_ddp.module.state_dict(), os.path.join(current_save_path, critic_model_filename))

    cleanup()

#################################################################################################
######################################### Main Operation ########################################
#################################################################################################

run_logs = defaultdict(lambda: defaultdict(list))
pbar_runs = tqdm(total=n_runs)
pbar_runs.set_description("Runs: ")
for run_num in range(n_runs):

    print('Run ' + str(run_num))

    #global_nfe = 0

    pbar = tqdm(total=original_max_train_episodes)  
    pbar.set_description("Episodes: ")

    current_save_path = save_path + "run " + str(run_num) 

    if not os.path.exists(current_save_path):
        os.mkdir(current_save_path)

    file_name = "RL_training_designs_ppo"
    if problem_choice == 1:
        if artery_prob:
            file_name += "_artery_"
        else:
            file_name += "_eqstiff_"
    else:
        if assign_prob:
            file_name += "_assign_"
        else:
            file_name += "_partition_"

    if n_heurs_used > 0:
        for i in range(len(heur_abbr)):
            if heurs_used[i]:
                file_name += heur_abbr[i]
        
    file_name += str(run_num) + ".csv"

    ## Initialize environment
    if problem_choice == 1:
        if artery_prob:
            if one_dec:
                base_train_env = GymWrapper(ArteryOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(ArteryOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())

            else:
                base_train_env = GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())
        else:
            if one_dec: 
                base_train_env = GymWrapper(EqualStiffnessOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(EqualStiffnessOneDecisionEnv(operations_instance=operations_instance, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())
            else:
                base_train_env = GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())
    else:
        if assign_prob:
            if one_dec:
                base_train_env = GymWrapper(AssignmentOneDecisionEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(AssignmentOneDecisionEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())
            else:
                base_train_env = GymWrapper(AssignmentProblemEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                train_env = TransformedEnv(base_train_env, RewardSum())
                base_eval_env = GymWrapper(AssignmentProblemEnv(operations_instance=operations_instance, resources_path=resources_path, obj_names=obj_names, heur_names=heur_names, heurs_used=heurs_used, consider_feas=consider_feas, dc_thresh=dc_thresh, mass_thresh=mass_thresh, pe_thresh=pe_thresh, ic_thresh=ic_thresh, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights), categorical_action_encoding=False)
                eval_env = TransformedEnv(base_eval_env, RewardSum())
        else:
            print("TBD")

    check_env_specs(train_env)
    check_env_specs(eval_env)
    
    ## Initialize actor and critic with loaded weights of previously trained networks
    actor = ActorNetwork(train_env, actor_fc_layer_params, actor_dropout_layer_params)
    critic = CriticNetwork(train_env, critic_fc_layer_params, critic_dropout_layer_params)

    if include_weights:
        policy_module = TensorDictModule(actor, in_keys=["design"], out_keys=["logits"])
    else:
        policy_module = TensorDictModule(actor, in_keys=["observation"], out_keys=["logits"])

    policy_module = ProbabilisticActor(
    module=policy_module,
    spec=train_env.action_spec,
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
    )

    if include_weights:
        value_module = ValueOperator(
        module=critic,
        in_keys=["design"],
        )
    else:
        value_module = ValueOperator(
        module=critic,
        in_keys=["observation"],
        )

    advantage_module = GAE(gamma=gamma, lmbda=lam, value_network=value_module, average_gae=True)

    loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_ratio,
    entropy_bonus=use_entropy_bonus,
    entropy_coef=ent_coeff,
    # these keys match by default but we set this for completeness
    critic_coef=critic_loss_coeff,
    loss_critic_type="l2",
    )
    #loss_module.set_vmap_randomness("same")

    #print("Running policy:", policy_module(train_env.reset()))
    #print("Running value:", value_module(train_env.reset()))

    ## Initialize result saver
    if metamat_prob:
        result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)
    else:
        result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=[], heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=0.0)


    collector = SyncDataCollector(
        train_env,
        policy_module,
        frames_per_batch=trajectory_collect_steps*episode_training_trajs, # steps per batch
        total_frames=trajectory_collect_steps*episode_training_trajs*original_max_train_episodes, # total steps over entire run
        max_frames_per_traj=trajectory_collect_steps,
        split_trajs=True,
        )

    if sample_minibatch: # Use replay buffer to sample minibatch each episode (replay buffer is purged after each episode so that the algorithm still works in an on-policy manner)
        replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=trajectory_collect_steps*episode_training_trajs),
        # sampler=SamplerWithoutReplacement(),)
        sampler=RandomSampler(),
        )

    #optim = torch.optim.Adam(loss_module.parameters(), lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #optim, total_frames // frames_per_batch, 0.0
    #)

    optim = torch.optim.RMSprop(loss_module.parameters(), lr=initial_learning_rate, alpha=alpha, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_rate, last_epoch=-1)

    #run_logs["run " + str(run_num)].append(defaultdict(list))
    pbar_steps = tqdm(total=trajectory_collect_steps*episode_training_trajs*original_max_train_episodes)
    eval_str = ""
    overall_step_counter = 0 # used for result logger (TODO: Instead of this, use step counter and traj_ids from data_view)
    #update_nfe = 0
    actor_losses_data = []
    critic_losses_data = []
    
    # We iterate over the collector until it reaches the total number of frames it was designed to collect:
    for episode, tensordict_data in enumerate(collector):

        data_view = tensordict_data.reshape(-1)
        if sample_minibatch:
            replay_buffer.extend(data_view.cpu())




        # Log trajectories into the result logger
        if include_weights:
            n_des_episode = data_view['design'].shape[0]
        else:
            n_des_episode = data_view['observation'].shape[0]
        for step in range(n_des_episode): # Saving all generated trajectories
            current_step = overall_step_counter + (data_view['collector']['traj_ids'][step].detach().cpu().numpy()*trajectory_collect_steps) + data_view['step_count'][step].detach().cpu().numpy()[0]
            selected_action = np.argmax(data_view['action'][step].detach().cpu().numpy())
            if include_weights:
                if one_dec:
                    save_des = False
                    if data_view['next']['done'][step]:
                        save_des = True
                        obs = data_view['next']['design'][step].detach().cpu().numpy().argmax(axis=1).astype(np.int32)
                else:
                    save_des = True
                    obs = data_view['design'][step].detach().cpu().numpy()
            
                # if 2 in obs:
                #     obs = np.zeros(n_states, dtype=np.int32)
                if save_des:
                    updated_nfe = result_logger.save_to_logger_pytorch(metamat_prob=metamat_prob, artery_prob=artery_prob, step_number=current_step, action=selected_action, prev_obs=obs, reward=data_view['next']['reward'][step].detach().cpu().numpy()[0], obj_weight=data_view['objective weight0'][step].detach().cpu().numpy()[0])
            else:
                if one_dec:
                    save_des = False
                    if data_view['next']['done'][step]:
                        obs = data_view['next']['observation'][step].detach().cpu().numpy().argmax(axis=1).astype(np.int32)
                else:
                    save_des = True
                    obs = data_view['observation'][step].detach().cpu().numpy()
                
                # if 2 in obs:
                #     obs = np.zeros(n_states, dtype=np.int32)
                if save_des:
                    updated_nfe = result_logger.save_to_logger_pytorch(metamat_prob=metamat_prob, artery_prob=artery_prob, step_number=current_step, action=selected_action, prev_obs=obs, reward=data_view['next']['reward'][step].detach().cpu().numpy()[0])

            #print(updated_nfe)
        
        overall_step_counter += data_view['step_count'].shape[0]

        # Evaluate actor at chosen intervals
        if compute_periodic_returns:
            if episode % eval_interval == 0:
                policy_module.eval()
                # We evaluate the policy once every "eval_interval" batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps.
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.RANDOM), torch.no_grad(): # ExplorationType.MEAN is nan for Categorical Distribution, ExplorationType.MODE chooses the action with the highest probability, ExplorationType.RANDOM samples the distribution for an action
                    # execute a rollout with the trained policy
                    eval_rollout = eval_env.rollout(max_eval_steps, policy_module)
                    run_logs["run " + str(run_num)]["eval episode"].append(episode)
                    run_logs["run " + str(run_num)]["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    run_logs["run " + str(run_num)]["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    #logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_reward_sum = run_logs["run " + str(run_num)]['eval reward (sum)'][-1]
                    init_eval_reward_sum = run_logs["run " + str(run_num)]['eval reward (sum)'][0]
                    eval_str = (
                        f"eval cumulative reward: {eval_reward_sum: 4.4f} "
                        f"(init: {init_eval_reward_sum: 4.4f}) "
                        #f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
                policy_module.train()

        if updated_nfe >= max_unique_nfe_run:
            # Extend the logs to complete them as if all episodes were run
            if episode < original_max_train_episodes:
                run_logs["run " + str(run_num)]["actor clipping loss"].extend([run_logs["run " + str(run_num)]["actor clipping loss"][-1] for i in range((original_max_train_episodes - episode -1)*train_epochs)])
                run_logs["run " + str(run_num)]["critic loss"].extend([run_logs["run " + str(run_num)]["critic loss"][-1] for i in range((original_max_train_episodes - episode - 1)*train_epochs)])
                run_logs["run " + str(run_num)]["actor entropy loss"].extend([run_logs["run " + str(run_num)]["actor entropy loss"][-1] for i in range((original_max_train_episodes - episode - 1)*train_epochs)])
                run_logs["run " + str(run_num)]["mean actor clipping loss"].extend([run_logs["run " + str(run_num)]["mean actor clipping loss"][-1] for i in range(original_max_train_episodes - episode - 1)])
                run_logs["run " + str(run_num)]["mean critic loss"].extend([run_logs["run " + str(run_num)]["mean critic loss"][-1] for i in range(original_max_train_episodes - episode - 1)])
                run_logs["run " + str(run_num)]["mean actor entropy loss"].extend([run_logs["run " + str(run_num)]["mean actor entropy loss"][-1] for i in range(original_max_train_episodes - episode - 1)])
                run_logs["run " + str(run_num)]["episode epochs"].extend([run_logs["run " + str(run_num)]["episode epochs"][-1] for i in range((original_max_train_episodes - episode - 1)*train_epochs)])
                run_logs["run " + str(run_num)]["reward"].extend([run_logs["run " + str(run_num)]["reward"][-1] for i in range(original_max_train_episodes - episode - 1)])
                run_logs["run " + str(run_num)]["lr"].extend([run_logs["run " + str(run_num)]["lr"][-1] for i in range(original_max_train_episodes - episode - 1)])
                run_logs["run " + str(run_num)]["train episode"].extend([(episode + i) for i in range(original_max_train_episodes - episode - 1)])
                if (original_max_train_episodes - episode) >= eval_interval:
                    run_logs["run " + str(run_num)]["eval episode"].extend([i for i in range(episode, original_max_train_episodes, eval_interval)])
                    run_logs["run " + str(run_num)]["eval reward"].extend([run_logs["run " + str(run_num)]["eval reward"][-1] for i in range(int((original_max_train_episodes - episode - 1)/eval_interval))])
                    run_logs["run " + str(run_num)]["eval reward (sum)"].extend([run_logs["run " + str(run_num)]["eval reward"][-1] for i in range(int((original_max_train_episodes - episode - 1)/eval_interval))])
            break

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

        pbar.update(1)

    # Save results to the file
    result_logger.save_to_csv()

    # Save mean losses to file
    actor_losses_fields = ["Episode No.", "Mean Actor Clipping Loss", "Mean Actor Entropy Loss"]
    critic_losses_fields = ["Episode No.", "Mean Critic Loss"]

    actor_losses_savepath = os.path.join(current_save_path, "mean_actor_losses.csv")
    critic_losses_savepath = os.path.join(current_save_path, "mean_critic_losses.csv")

    with open(actor_losses_savepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(actor_losses_fields)
            writer.writerows(actor_losses_data)

    with open(critic_losses_savepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(critic_losses_fields)
            writer.writerows(critic_losses_data)

    # Save trained actor and critic networks
    actor_model_filename = "learned_actor_network_final"
    if problem_choice == 1:
        if artery_prob:
            actor_model_filename += "_artery"
        else:
            actor_model_filename += "_eqstiff"
    else:
        if assign_prob:
            actor_model_filename += "_assign"
        else:
            actor_model_filename += "_partition"

    actor_model_filename += ".pth"

    torch.save(actor.state_dict(), os.path.join(current_save_path, actor_model_filename))

    critic_model_filename = "learned_critic_network_final"
    if problem_choice == 1:
        if artery_prob:
            critic_model_filename += "_artery"
        else:
            critic_model_filename += "_eqstiff"
    else:
        if assign_prob:
            critic_model_filename += "_assign"
        else:
            critic_model_filename += "_partition"

    critic_model_filename += ".pth"

    torch.save(critic.state_dict(), os.path.join(current_save_path, critic_model_filename))

    pbar_runs.update(run_num+1)

end_time = time.time()

print("Total time: ", (end_time - start_time))

#################################################################################################
############################################ Plotting ###########################################
#################################################################################################

# Compute loss statistics across runs
actor_clipping_loss_stats = {}
critic_loss_stats = {}
actor_entropy_loss_stats = {}
overall_loss_stats = {}

mean_actor_clipping_loss_stats = {}
mean_critic_loss_stats = {}
mean_actor_entropy_loss_stats = {}
mean_overall_loss_stats = {}

actor_clipping_loss_runs = np.zeros((n_runs, original_max_train_episodes*train_epochs))
critic_loss_runs = np.zeros((n_runs, original_max_train_episodes*train_epochs))
actor_entropy_loss_runs = np.zeros((n_runs, original_max_train_episodes*train_epochs))
overall_loss_runs = np.zeros((n_runs, original_max_train_episodes*train_epochs))

mean_actor_clipping_loss_runs = np.zeros((n_runs, original_max_train_episodes))
mean_critic_loss_runs = np.zeros((n_runs, original_max_train_episodes))
mean_actor_entropy_loss_runs = np.zeros((n_runs, original_max_train_episodes))
mean_overall_loss_runs = np.zeros((n_runs, original_max_train_episodes))

for run_num in range(n_runs):
    actor_clipping_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["actor clipping loss"]
    critic_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["critic loss"]
    actor_entropy_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["actor entropy loss"]
    overall_loss_runs[run_num, :] = [np.sum(x) for x in zip(run_logs["run " + str(run_num)]["actor clipping loss"], run_logs["run " + str(run_num)]["critic loss"], run_logs["run " + str(run_num)]["actor entropy loss"])]

    mean_actor_clipping_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["mean actor clipping loss"]
    mean_critic_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["mean critic loss"]
    mean_actor_entropy_loss_runs[run_num, :] = run_logs["run " + str(run_num)]["mean actor entropy loss"]
    mean_overall_loss_runs[run_num, :] = [np.sum(x) for x in zip(run_logs["run " + str(run_num)]["mean actor clipping loss"], run_logs["run " + str(run_num)]["mean critic loss"], run_logs["run " + str(run_num)]["mean actor entropy loss"])]
    
actor_clipping_loss_stats['median'] = np.median(actor_clipping_loss_runs, axis=0)
actor_clipping_loss_stats['first quartile'] = np.percentile(actor_clipping_loss_runs, 25, axis=0)
actor_clipping_loss_stats['third quartile'] = np.percentile(actor_clipping_loss_runs, 75, axis=0)

critic_loss_stats['median'] = np.median(critic_loss_runs, axis=0)
critic_loss_stats['first quartile'] = np.percentile(critic_loss_runs, 25, axis=0)
critic_loss_stats['third quartile'] = np.percentile(critic_loss_runs, 75, axis=0)

actor_entropy_loss_stats['median'] = np.median(actor_entropy_loss_runs, axis=0)
actor_entropy_loss_stats['first quartile'] = np.percentile(actor_entropy_loss_runs, 25, axis=0)
actor_entropy_loss_stats['third quartile'] = np.percentile(actor_entropy_loss_runs, 75, axis=0)

overall_loss_stats['median'] = np.median(overall_loss_runs, axis=0)
overall_loss_stats['first quartile'] = np.percentile(overall_loss_runs, 25, axis=0)
overall_loss_stats['third quartile'] = np.percentile(overall_loss_runs, 75, axis=0)

mean_actor_clipping_loss_stats['median'] = np.median(mean_actor_clipping_loss_runs, axis=0)
mean_actor_clipping_loss_stats['first quartile'] = np.percentile(mean_actor_clipping_loss_runs, 25, axis=0)
mean_actor_clipping_loss_stats['third quartile'] = np.percentile(mean_actor_clipping_loss_runs, 75, axis=0)

mean_critic_loss_stats['median'] = np.median(mean_critic_loss_runs, axis=0)
mean_critic_loss_stats['first quartile'] = np.percentile(mean_critic_loss_runs, 25, axis=0)
mean_critic_loss_stats['third quartile'] = np.percentile(mean_critic_loss_runs, 75, axis=0)

mean_actor_entropy_loss_stats['median'] = np.median(mean_actor_entropy_loss_runs, axis=0)
mean_actor_entropy_loss_stats['first quartile'] = np.percentile(mean_actor_entropy_loss_runs, 25, axis=0)
mean_actor_entropy_loss_stats['third quartile'] = np.percentile(mean_actor_entropy_loss_runs, 75, axis=0)

mean_overall_loss_stats['median'] = np.median(mean_overall_loss_runs, axis=0)
mean_overall_loss_stats['first quartile'] = np.percentile(mean_overall_loss_runs, 25, axis=0)
mean_overall_loss_stats['third quartile'] = np.percentile(mean_overall_loss_runs, 75, axis=0)

# Plotting losses
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(run_logs["run 0"]["episode epochs"], actor_clipping_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["episode epochs"], actor_clipping_loss_stats['first quartile'], actor_clipping_loss_stats['third quartile'], alpha=0.5)
for i in range(original_max_train_episodes): # Vertical lines to denote episode demarcations
    plt.plot([i*train_epochs for j in range(100)], np.linspace(0, np.max([np.max(actor_clipping_loss_stats['median']), np.max(actor_clipping_loss_stats['first quartile']), np.max(actor_clipping_loss_stats['third quartile'])]), 100), '--r')
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.title("Actor Clipping Loss")

plt.subplot(2, 2, 2)
plt.plot(run_logs["run 0"]["episode epochs"], critic_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["episode epochs"], critic_loss_stats['first quartile'], critic_loss_stats['third quartile'], alpha=0.5)
for i in range(original_max_train_episodes): # Vertical lines to denote episode demarcations
    plt.plot([i*train_epochs for j in range(100)], np.linspace(0, np.max([np.max(critic_loss_stats['median']), np.max(critic_loss_stats['first quartile']), np.max(critic_loss_stats['third quartile'])]), 100), '--r')
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.title("Critic Loss")

plt.subplot(2, 2, 3)
plt.plot(run_logs["run 0"]["episode epochs"], actor_entropy_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["episode epochs"], actor_entropy_loss_stats['first quartile'], actor_entropy_loss_stats['third quartile'], alpha=0.5)
for i in range(original_max_train_episodes): # Vertical lines to denote episode demarcations
    plt.plot([i*train_epochs for j in range(100)], np.linspace(0, np.max([np.max(actor_entropy_loss_stats['median']), np.max(actor_entropy_loss_stats['first quartile']), np.max(actor_entropy_loss_stats['third quartile'])]), 100), '--r')
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.title("Actor Entropy Loss")

plt.subplot(2, 2, 4)
plt.plot(run_logs["run 0"]["episode epochs"], overall_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["episode epochs"], overall_loss_stats['first quartile'], overall_loss_stats['third quartile'], alpha=0.5)
for i in range(original_max_train_episodes): # Vertical lines to denote episode demarcations
    plt.plot([i*train_epochs for j in range(100)], np.linspace(0, np.max([np.max(overall_loss_stats['median']), np.max(overall_loss_stats['first quartile']), np.max(overall_loss_stats['third quartile'])]), 100), '--r')
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.title("Overall Loss")

actor_losses_filename = "actor_losses"
if problem_choice == 1:
    if artery_prob:
        actor_losses_filename += "_artery"
    else:
        actor_losses_filename += "_eqstiff"
else:
    if assign_prob:
        actor_losses_filename += "_assign"
    else:
        actor_losses_filename += "_partition"

actor_losses_filename += ".png"
plt.savefig(os.path.join(save_path, actor_losses_filename))
#plt.show()

# Plotting mean losses
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(run_logs["run 0"]["train episode"], mean_actor_clipping_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["train episode"], mean_actor_clipping_loss_stats['first quartile'], mean_actor_clipping_loss_stats['third quartile'], alpha=0.5)
plt.xlabel("Training Episode")
plt.ylabel("Loss")
plt.title("Actor Clipping Loss")

plt.subplot(2, 2, 2)
plt.plot(run_logs["run 0"]["train episode"], mean_critic_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["train episode"], mean_critic_loss_stats['first quartile'], mean_critic_loss_stats['third quartile'], alpha=0.5)
plt.xlabel("Training Episode")
plt.ylabel("Loss")
plt.title("Critic Loss")

plt.subplot(2, 2, 3)
plt.plot(run_logs["run 0"]["train episode"], mean_actor_entropy_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["train episode"], mean_actor_entropy_loss_stats['first quartile'], mean_actor_entropy_loss_stats['third quartile'], alpha=0.5)
plt.xlabel("Training Episode")
plt.ylabel("Loss")
plt.title("Actor Entropy Loss")

plt.subplot(2, 2, 4)
plt.plot(run_logs["run 0"]["train episode"], mean_overall_loss_stats['median']) 
plt.fill_between(run_logs["run 0"]["train episode"], mean_overall_loss_stats['first quartile'], mean_overall_loss_stats['third quartile'], alpha=0.5)
plt.xlabel("Training Episode")
plt.ylabel("Loss")
plt.title("Overall Loss")

actor_losses_filename = "actor_mean_losses"
if problem_choice == 1:
    if artery_prob:
        actor_losses_filename += "_artery"
    else:
        actor_losses_filename += "_eqstiff"
else:
    if assign_prob:
        actor_losses_filename += "_assign"
    else:
        actor_losses_filename += "_partition"

actor_losses_filename += ".png"
plt.savefig(os.path.join(save_path, actor_losses_filename))
#plt.show()

# Compute agent evaluation statistics across runs
eval_mean_reward_stats = {}
eval_return_stats = {}

eval_mean_reward_runs = np.zeros((n_runs, int(original_max_train_episodes/eval_interval)))
eval_return_runs = np.zeros((n_runs, int(original_max_train_episodes/eval_interval)))

for run_num in range(n_runs):
    eval_mean_reward_runs[run_num, :] = run_logs["run " + str(run_num)]["eval reward"]
    eval_return_runs[run_num, :] = run_logs["run " + str(run_num)]["eval reward (sum)"]

eval_mean_reward_stats['median'] = np.median(eval_mean_reward_runs, axis=0)
eval_mean_reward_stats['first quartile'] = np.percentile(eval_mean_reward_runs, 25, axis=0)
eval_mean_reward_stats['third quartile'] = np.percentile(eval_mean_reward_runs, 75, axis=0)

eval_return_stats['median'] = np.median(eval_return_runs, axis=0)
eval_return_stats['first quartile'] = np.percentile(eval_return_runs, 25, axis=0)
eval_return_stats['third quartile'] = np.percentile(eval_return_runs, 75, axis=0)

# # Plotting Agent Evaluation Results
plt.figure(figsize=(5, 10))
plt.subplot(2,1,1)
plt.plot(run_logs["run 0"]["eval episode"], eval_mean_reward_stats['median'])
plt.fill_between(run_logs["run 0"]["eval episode"], eval_mean_reward_stats['first quartile'], eval_mean_reward_stats['third quartile'], alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Mean Reward")

plt.subplot(2,1,2)
plt.plot(run_logs["run 0"]["eval episode"], eval_return_stats['median'])
plt.fill_between(run_logs["run 0"]["eval episode"], eval_return_stats['first quartile'], eval_return_stats['third quartile'], alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Evaluation Return")

actor_eval_filename = "actor_evaluation"
if problem_choice == 1:
    if artery_prob:
        actor_eval_filename += "_artery"
    else:
        actor_eval_filename += "_eqstiff"
else:
    if assign_prob:
        actor_eval_filename += "_assign"
    else:
        actor_eval_filename += "_partition"
actor_eval_filename += ".png"
plt.savefig(os.path.join(save_path, actor_eval_filename))
#plt.show()
