# -*- coding: utf-8 -*-
"""
Training and saving a PPO agent using PyTorch methods
Reference: https://keras.io/examples/rl/ppo_cartpole/ and keras-based PPO agent scripts

TODO:
1. Track NFE and stop training when max NFE is reached
2. Plotting
3. General debugging

@author: roshan94
"""
import os

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
import warnings
warnings.filterwarnings("ignore")

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, RewardSum, StepCounter, TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.libs.gym import GymWrapper
from tqdm import tqdm

print("Torch cuda available: ", torch.cuda.is_available())
print("Torch device: ", torch.cuda.get_device_name(0))
device = torch.device(0)
#device = torch.device("cpu")

from envs.metamaterial.ArteryProblemEnv import ArteryProblemEnv
from envs.metamaterial.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from save.ResultSaving import ResultSaver

import numpy as np
import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

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

max_steps = data_ppo["Maximum steps in training episode (for train environment termination)"] 
max_eval_steps = data_ppo["Maximum steps in evaluation episode (for evaluation environment termination)"] # termination in evaluation environment
max_eval_episodes = data_ppo["Number of evaluation episodes"] # number of episodes per evaluation of the actor

max_unique_nfe_run = data_ppo["Maximum unique NFE"]

use_buffer = data_ppo["Buffer used"]

eval_interval = data_ppo["Episode interval for evaluation"] # After how many episodes is the actor being evaluated
new_reward = data_ppo["Use new problem formulation"]
include_weights = data_ppo["Include weights in state"]

sample_minibatch = data_ppo["Sample minibatch"] # Whether to sample minibatch or use the entire set of generated trajectories
trajectory_collect_steps = data_ppo["Number of steps in a collected trajectory"] # number of steps in each trajectory
episode_training_trajs = data_ppo["Number of trajectories used for training per episode"] # number of trajectories sampled in each iteration to train the actor and critic
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
problem_choice = 1 # 1 - Metamaterial problem, 2 - EOSS problem

match problem_choice:
    case 1:
        f_prob = open('.\\envs\\metamaterial\\problem-config.json')
        data_prob = json.load(f_prob)

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

        c_target = data_prob["Target stiffness ratio"]
        # c_target = 1
        # if artery_prob:
        #     c_target = 0.421

        feas_c_target_delta = data_prob["Feasible stiffness delta"] # delta about target stiffness ratio defining satisfying designs

        render_steps = data_prob["Render steps"]

        ## find number of states and actions based on sidenum
        n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
        n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
        n_actions = 1

    case 2:
        print("TBD")
        #print("EOSS - Assigning Problem")

        #print("EOSS - Partitioning Problem")

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

    def __init__(self, environ, hidden_layer_units, hidden_layer_dropout_probs):
        super().__init__()

        # Define layers for actor network
        layers = []
        layers.append(nn.Linear(environ.observation_spec.shape[-1], hidden_layer_units[0], device=device)) # Input layer
        for i in range(len(hidden_layer_units)-1): # Hidden layers
            layers.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            layers.append(nn.ReLU(device=device))
            layers.append(nn.Dropout(hidden_layer_dropout_probs[i], device=device))
        layers.append(nn.Linear(hidden_layer_units[-1], environ.action_spec.shape[-1], device=device)) # Output layer
        
        self.actor_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor_net(x)
    
    def sample_action(self, x):
        action_logits = self.forward(x)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()

        return action, action_logits

class CriticNetwork(nn.Module):

    def __init__(self, environ, hidden_layer_units, hidden_layer_dropout_probs):
        super().__init__()

        # Define layers for critic network
        layers = []
        layers.append(nn.Linear(environ.observation_spec.shape[-1], hidden_layer_units[0], device=device)) # Input layer
        for i in range(len(hidden_layer_units)-1): # Hidden layers
            layers.append(nn.Linear(hidden_layer_units[i], hidden_layer_units[i+1], device=device))
            layers.append(nn.ReLU(device=device))
            layers.append(nn.Dropout(hidden_layer_dropout_probs[i], device=device))
        layers.append(nn.Linear(hidden_layer_units[-1], 1, device=device)) # Output layer
        
        self.critic_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.critic_net(x)
    
    def estimate_value(self, x):
        return self.forward(x)

#################################################################################################
######################################### Main Operation ########################################
#################################################################################################


for run_num in range(n_runs):

    print('Run ' + str(run_num))

    global_nfe = 0

    logs = defaultdict(list)
    pbar = tqdm(total=original_max_train_episodes)

    current_save_path = save_path + "run " + str(run_num) 

    if not os.path.exists(current_save_path):
        os.mkdir(current_save_path)

    file_name = "RL_training_designs_ppo"
    if artery_prob:
        file_name += "_artery_"
    else:
        file_name += "_eqstiff_"

    if n_heurs_used > 0:
        for i in range(len(heur_abbr)):
            if heurs_used[i]:
                file_name += heur_abbr[i]
        
    file_name += str(run_num) + ".csv"

    ## Initialize environment
    if problem_choice == 1:
        if artery_prob:
            base_env = GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights))
            train_env = TransformedEnv(base_env, Compose(RewardSum(), StepCounter()))
            eval_env = GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights))
        else:
            base_env = GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights))
            #base_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)
            train_env = TransformedEnv(base_env, Compose(RewardSum(), StepCounter()))
            #eval_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights)
            eval_env = GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, c_target_delta=feas_c_target_delta, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max, include_wts_in_state=include_weights))
    else:
        print("TBD")
    
    ## Initialize actor and critic
    actor = ActorNetwork(train_env, actor_fc_layer_params, actor_dropout_layer_params)
    critic = CriticNetwork(train_env, critic_fc_layer_params, critic_dropout_layer_params)

    ## Initialize result saver
    result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward, include_weights=include_weights, c_target_delta=feas_c_target_delta)

    if sample_minibatch:
        collector = SyncDataCollector(
        train_env,
        actor,
        frames_per_batch=trajectory_collect_steps, # steps per batch
        total_frames=max_steps, # total steps over entire run
        max_frames_per_traj=minibatch_steps,
        split_trajs=False,
        device=device,)
    else:
        collector = SyncDataCollector(
        train_env,
        actor,
        frames_per_batch=trajectory_collect_steps, # steps per batch
        total_frames=max_steps, # total steps over entire run
        max_frames_per_traj=trajectory_collect_steps,
        split_trajs=False,
        device=device,)

    if sample_minibatch: # Use replay buffer to sample minibatch each episode (replay buffer is purged after each episode so that the algorithm still works in an on-policy manner)
        replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=trajectory_collect_steps),
        sampler=SamplerWithoutReplacement(),)

    advantage_module = GAE(gamma=gamma, lmbda=lam, value_network=critic, average_gae=True)

    loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic,
    clip_epsilon=clip_ratio,
    entropy_bonus=use_entropy_bonus,
    entropy_coef=ent_coeff,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="l2",
    )

    #optim = torch.optim.Adam(loss_module.parameters(), lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #optim, total_frames // frames_per_batch, 0.0
    #)

    optim = torch.optim.RMSprop(loss_module.parameters(), lr=initial_learning_rate, alpha=alpha, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_rate, last_epoch=original_max_train_episodes)

    logs = defaultdict(list)
    pbar = tqdm(total=max_steps)
    eval_str = ""
    overall_step_counter = 0 # used for result logger
    #episode = 0

    train_env.reset()
    eval_env.reset()
    
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for episode, tensordict_data in enumerate(collector):

        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(train_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            if sample_minibatch:
                replay_buffer.extend(data_view.cpu())

            # Log trajectory into the result logger
            if sample_minibatch:
                subdata = replay_buffer.sample(minibatch_steps)
            else:
                subdata = deepcopy(data_view)
                minibatch_steps = trajectory_collect_steps

            for step in range(minibatch_steps):
                result_logger.save_to_logger2(step_number=overall_step_counter, action=subdata["action"][step, 0], truss_design=subdata["Truss Design"][step], reward=subdata["Reward"][step])
                overall_step_counter += 1

            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

            if sample_minibatch:
                replay_buffer.empty()

        #episode += 1

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    # Save current trained actor and critic networks at regular intervals
    if episode % network_save_intervals == 0:
        actor_model_filename = "learned_actor_network_ep" + str(episode)
        if artery_prob:
            actor_model_filename += "_artery"
        else:
            actor_model_filename += "_eqstiff"
        actor_model_filename += ".h5"

        torch.save(actor.state_dict(), os.path.join(current_save_path, actor_model_filename))

        critic_model_filename = "learned_critic_network_ep" + str(episode)
        if artery_prob:
            critic_model_filename += "_artery"
        else:
            critic_model_filename += "_eqstiff"
        critic_model_filename += ".h5"

        torch.save(critic.state_dict(), os.path.join(current_save_path, critic_model_filename))

    # Evaluate actor at chosen intervals
    if compute_periodic_returns:
        if i % eval_interval == 0:
            # We evaluate the policy once every "eval_interval" batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps.
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = eval_env.rollout(1000, actor)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

    # Save results to the file
    result_logger.save_to_csv()

    # Save trained actor and critic networks
    actor_model_filename = "learned_actor_network_final"
    if artery_prob:
        actor_model_filename += "_artery"
    else:
        actor_model_filename += "_eqstiff"
    actor_model_filename += ".h5"

    torch.save(actor.state_dict(), os.path.join(current_save_path, actor_model_filename))

    critic_model_filename = "learned_critic_network_final"
    if artery_prob:
        critic_model_filename += "_artery"
    else:
        critic_model_filename += "_eqstiff"
    critic_model_filename += ".h5"

    torch.save(critic.state_dict(), os.path.join(current_save_path, critic_model_filename))

end_time = time.time()

print("Total time: ", (end_time - start_time))




    










