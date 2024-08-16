# -*- coding: utf-8 -*-
"""
Training a PPO Agent using Pytorch and TorchRL

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

from collections import defaultdict

from tqdm import tqdm

import matplotlib.pyplot as plt
import torch 

print("Torch cuda available: ", torch.cuda.is_available())
print("Torch device: ", torch.cuda.get_device_name(0))

import warnings
warnings.filterwarnings("ignore")

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, ObservationNorm, DoubleToFloat, StepCounter, TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

#device = torch.device(0)
device = torch.device("cpu")

from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from save.ResultSaving import ResultSaver

import numpy as np
import math 

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

import time

#################################################################################################
############################## GYM - Mountain Car (continuous) example ##########################
#################################################################################################

start_time = time.time()

num_cells = 256
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

lr = 3e-4
max_grad_norm = 1.0

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

base_env = GymEnv("MountainCarContinuous-v0", device=device)
env = TransformedEnv(base_env, Compose(ObservationNorm(in_keys=["observation"]), DoubleToFloat(), StepCounter()))

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print("normalization constant shape:", env.transform[0].loc.shape)

print("observation_spec: ", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("input_spec:", env.input_spec)
print("action_spec (as defined by input_spec):", env.action_spec)

print("Checking env specs")
check_env_specs(env)

rollout = env.rollout(3)
print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

actor_net = nn.Sequential(
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
NormalParamExtractor(),)

policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])

policy_module = ProbabilisticActor(
module=policy_module,
spec=env.action_spec,
in_keys=["loc", "scale"],
distribution_class=TanhNormal,
distribution_kwargs={
    "min": env.action_spec.space.low,
    "max": env.action_spec.space.high,
},
return_log_prob=True,
# we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(num_cells, device=device),
nn.Tanh(),
nn.LazyLinear(1, device=device),)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
env,
policy_module,
frames_per_batch=frames_per_batch,
total_frames=total_frames,
split_trajs=False,
device=device,)

replay_buffer = ReplayBuffer(
storage=LazyTensorStorage(max_size=frames_per_batch),
sampler=SamplerWithoutReplacement(),)

advantage_module = GAE(
gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
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

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
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

end_time = time.time()

print("Total time: ", (end_time - start_time))

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()

#################################################################################################
############################## Metamaterial Problems ############################################
#################################################################################################

# ## Setup and train parameters from config file
# f_ppo = open('.\\pytorch-based\\ppo\\ppo-config.json')
# data_ppo = json.load(f_ppo)

# n_runs = data_ppo["Number of runs"]

# gamma = data_ppo["Value discount (gamma)"] # discount factor
# max_train_episodes = data_ppo["Number of training episodes"] # number of training episodes

# max_steps = data_ppo["Maximum steps in training episode (for train environment termination)"] # no termination in training environment
# max_eval_steps = data_ppo["Maximum steps in evaluation episode (for evaluation environment termination)"] # termination in evaluation environment
# max_eval_episodes = data_ppo["Number of evaluation episodes"] # number of episodes per evaluation of the actor

# use_buffer = data_ppo["Buffer used"]

# eval_interval = data_ppo["Episode interval for evaluation"] # After how many episodes is the actor being evaluated
# new_reward = data_ppo["Use new problem formulation"]

# initial_collect_trajs = data_ppo["Initial number of stored trajectories"] # number of trajectories in the driver to populate replay buffer before beginning training (only used if replay buffer is used)
# trajectory_collect_steps = data_ppo["Number of steps in a collected trajectory"] # number of steps in each trajectory
# episode_training_trajs = data_ppo["Number of trajectories used for training per episode"] # number of trajectories sampled in each iteration to train the actor and critic
# minibatch_steps = data_ppo["Number of steps in a minibatch"]
# replay_buffer_capacity = data_ppo["Replay buffer capacity"] # maximum number of trajectories that can be stored in the buffer

# if minibatch_steps > trajectory_collect_steps:
#     print("Number of steps in a minibatch is greater than the number of collected steps, reduce the minibatch steps")
#     sys.exit(0)

# ## NOTE: Total number of designs used for training in each run = episode_training_trajs*minibatch_steps*max_train_episodes

# advantage_norm = data_ppo["Normalize advantages"] # whether to normalize advantages for training
# discrete_actions = data_ppo["Discrete actions"]
# lam = data_ppo["Advantage discount (lambda)"] # advantage discount factor

# actor_fc_layer_params = data_ppo["Actor network layer units"]
# actor_dropout_layer_params = data_ppo["Actor network dropout probabilities"]

# critic_fc_layer_params = data_ppo["Critic network layer units"]
# critic_dropout_layer_params = data_ppo["Critic network dropout probabilities"]

# ## NOTE: At least clipping or adaptive KL penalty must be used
# use_clipping = data_ppo["Use clipping loss"] # Use PPO clipping term in actor loss
# clip_ratio = data_ppo["Clipping ratio threshold"]

# use_adaptive_kl_penalty = data_ppo["Use Adaptive KL penalty loss"] # Use adaptive KL-divergence based penalty term in actor loss
# kl_targ = data_ppo["KL target"]
# beta = data_ppo["Adaptive KL coefficient (beta)"]

# use_entropy_bonus = data_ppo["Use entropy loss bonus"] # Use additional entropy of actor distribution term in actor loss
# ent_coeff = data_ppo["Entropy coefficient"] # coefficient for the entropy bonus term in actor loss
# use_early_stopping = data_ppo["Use early stopping for actor training"] # Stop training epoch early if current KL-divergence crosses 1.5 * kl_targ

# train_policy_iterations = data_ppo["Number of actor training iterations"]
# train_value_iterations = data_ppo["Number of critic training iterations"]

# initial_actor_learning_rate = data_ppo["Initial actor training learning rate"]
# initial_critic_learning_rate = data_ppo["Initial critic training learning_rate"]  
# decay_rate = data_ppo["Learning rate decay rate"]
# decay_steps_actor = data_ppo["Learning rate decay steps (actor)"]
# decay_steps_critic = data_ppo["Learning rate decay steps (critic)"]
# #decay_steps_actor = max_train_episodes*train_policy_iterations 
# #decay_steps_critic = max_train_episodes*train_value_iterations 
# rho = data_ppo["RMSprop optimizer rho"]
# momentum = data_ppo["RMSprop optimizer momentum"]

# compute_periodic_returns = data_ppo["Compute periodic returns"]
# use_continuous_minibatch = data_ppo["Continuous minibatch"] # use continuous trajectory slices to generate minibatch for training or random transitions from all trajectories
# network_save_intervals = data_ppo["Episode interval to save actor and critic networks"]

# save_path = data_ppo["Savepath"]

# if network_save_intervals > max_train_episodes:
#     print("Episode interval to save networks is greater than number of training episodes")
#     sys.exit(0)

# ## Load problem parameters from config file
# f_prob = open('.\\pytorch-based\\problem-config.json')
# data_prob = json.load(f_prob)

# artery_prob = data_prob["Solve artery problem"] # If true -> artery problem, false -> equal stiffness problem

# #print(device_lib.list_local_devices())

# # model_sel = 0 --> Fibre Stiffness Model
# #           = 1 --> Truss Stiffness Model
# #           = 2 --> Beam Model        
# model_sel = data_prob["Model selection"]

# rad = data_prob["Member radius in m"] # in m
# sel = data_prob["Lattice side element length in m"] # in m
# E_mod = data_prob["Young's Modulus in Pa"] # in Pa
# sidenum = data_prob["Lattice number of side nodes"]
# nucFac = data_prob["NucFac"]

# obj_names = data_prob["Objective names"]
# heur_names = data_prob["Heuristic names"] # make sure this is consistent with the order of the heuristic operators in the Java code
# heur_abbr = data_prob["Heuristic abbreviations"]
# heurs_used = data_prob["Heuristics used"] # for [partial collapsibility, nodal properties, orientation, intersection]
# n_heurs_used = heurs_used.count(True)
# constr_names = data_prob["Constraint names"]
# # if not artery_prob:
# #     constr_names = ['FeasibilityViolation','ConnectivityViolation','StiffnessRatioViolation']
# # else:
# #     constr_names = ['FeasibilityViolation','ConnectivityViolation']
# objs_max = data_prob["Objective maximized"]

# c_target = data_prob["Target stiffness ratio"]
# # c_target = 1
# # if artery_prob:
# #     c_target = 0.421

# render_steps = data_prob["Render steps"]

# ## find number of states and actions based on sidenum
# n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
# n_action_vals = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
# n_actions = 1

# n_episodes_per_fig = 4 # used for plotting returns and losses 
# linestyles = ['solid','dotted','dashed','dashdot']

# ## Access java gateway and pass parameters to operations class instance
# gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
# operations_instance = gateway.entry_point.getOperationsInstance()

# for run_num in range(n_runs):

#     print('Run ' + str(run_num))

#     current_save_path = save_path + "run " + str(run_num) 

#     if not os.path.exists(current_save_path):
#         os.mkdir(current_save_path)

#     file_name = "RL_training_designs_ppo"
#     if artery_prob:
#         file_name += "_artery_"
#     else:
#         file_name += "_eqstiff_"

#     if n_heurs_used > 0:
#         for i in range(len(heur_abbr)):
#             if heurs_used[i]:
#                 file_name += heur_abbr[i]
        
#     file_name += str(run_num) + ".csv"

#     # Initialize result saver
#     result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, new_reward=new_reward)

#     if artery_prob:
#         train_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
#         eval_env = ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps)
#     else:
#         train_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max)
#         eval_env = EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_action_vals, n_states=n_states, max_steps=max_eval_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps, new_reward=new_reward, obj_max=objs_max)

#     train_env_transform = TransformedEnv(train_env, Compose(DoubleToFloat(), StepCounter()))

#     print("observation_spec: ", train_env.observation_spec)
#     print("reward_spec:", train_env.reward_spec)
#     print("input_spec:", train_env.input_spec)
#     print("action_spec (as defined by input_spec):", train_env.action_spec)

#     print("Checking env specs")
#     check_env_specs(train_env)

    
    