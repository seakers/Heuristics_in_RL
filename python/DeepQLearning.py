# -*- coding: utf-8 -*-
"""
Deep Q-learning main script for both the metamaterial and satelite problems
Adapted from https://www.tensorflow.org/agents/tutorials/9_c51_tutorial and https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial

@author: roshan94
"""
import math
import numpy as np
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from tqdm import tqdm

import matplotlib.pyplot as plt

import tensorflow as tf
import os

#from tf_agents.metrics import tf_metrics

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments.gym_wrapper import GymWrapper
#from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import tf_py_environment
#from tf_agents.networks import sequential
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
#from tf_agents.trajectories import trajectory
#from tf_agents.specs import tensor_spec
from tf_agents.utils import common

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"

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

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
#max_steps = 1000

n_runs = 5

## DQN Hyperparameters
num_iterations = 420 
max_steps = num_iterations

initial_collect_steps = 100  
collect_steps_per_iteration = 1  
replay_buffer_capacity = 10000 

fc_layer_params = (280, 1000, 1500, 560)
dropout_layer_params = (0, 0.10, 0.20, 0)

batch_size = 32
learning_rate = 1e-3  
gamma = 0.99
log_interval = 50  
prob_eps_greedy = 0.05

num_eval_episodes = 10  
eval_interval = 10 

n_step_update = 4

## Define method to compute average return for DQN training
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

returns_run = []

for run_num in range(n_runs): 

  print('Run ' + str(run_num))

  file_name = "RL_training_designs_dqn"
  if artery_prob:
      file_name += "_artery"
  else:
      file_name += "_eqstiff"

  file_name += str(run_num) + ".csv"

  if artery_prob:
      train_env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      eval_env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #train_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
  else:
      train_env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      eval_env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #train_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, file_name=file_name, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

  ## Define Q-network
  q_net = q_network.QNetwork(
      input_tensor_spec=train_env.observation_spec(),
      action_spec=train_env.action_spec(),
      fc_layer_params=fc_layer_params,
      q_layer_activation_fn=tf.keras.activations.softmax,
      dropout_layer_params=dropout_layer_params)

  # def dense_layer(num_units):
  #   return tf.keras.layers.Dense(
  #       num_units,
  #       activation=tf.keras.activations.relu,
  #       kernel_initializer=tf.keras.initializers.VarianceScaling(
  #           scale=2.0, mode='fan_in', distribution='truncated_normal'))

  # dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
  # q_values_layer = tf.keras.layers.Dense(
  #     n_actions,
  #     activation=None,
  #     kernel_initializer=tf.keras.initializers.RandomUniform(
  #         minval=-0.03, maxval=0.03),
  #     bias_initializer=tf.keras.initializers.Constant(-0.2))
  # q_net = sequential.Sequential(dense_layers + [q_values_layer])

  ## Define optimizer
  optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)

  #train_step_counter = tf.Variable(0)

  global_step = tf.compat.v1.train.get_or_create_global_step()

  ## Define DQN Agent
  agent = dqn_agent.DqnAgent(
      time_step_spec=train_env.time_step_spec(),
      action_spec=train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      n_step_update=n_step_update,
      gamma=gamma,
      td_errors_loss_fn=common.element_wise_squared_loss,
      #train_step_counter=train_step_counter,
      train_step_counter=global_step,
      epsilon_greedy=prob_eps_greedy)

  agent.initialize()

  policy_dir = os.path.join(save_path, ('policy'+str(run_num)))
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)

  ## Define random policy for initial trajectory addition to the replay buffer
  random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

  ## Define Replay Buffer for DQN training
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=1,
      max_length=replay_buffer_capacity)

  ## Add trajectory elements to replay buffer initially using a random policy
  #def collect_step(environment, policy): 

    ## Single step transition passed onto trajectory 
    #time_step = environment.current_time_step()
    #action_step = policy.action(time_step)
    #next_time_step = environment.step(action_step) 

    #traj = trajectory.from_transition(time_step, action_step, next_time_step)

    ## Add trajectory to the replay buffer
    #replay_buffer.add_batch(traj)

  #train_env.reset()
  #eval_env.reset()

  #for _ in range(initial_collect_steps):
    #collect_step(train_env, random_policy)

  ## Adding trajectories using Dynamic Step Driver instead of the manual process above
  train_env.reset()
  eval_env.reset()

  train_obs = [replay_buffer.add_batch]

  collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, random_policy, train_obs, num_steps=initial_collect_steps)

  collect_driver.run()

  ## Dataset generates trajectories with shape [BxTx...] where T = n_step_update + 1.
  dataset = replay_buffer.as_dataset(num_parallel_calls=2, sample_batch_size=batch_size, num_steps=n_step_update + 1).prefetch(3)

  ## Observers
  #num_episodes = tf_metrics.NumberOfEpisodes()
  #env_steps = tf_metrics.EnvironmentSteps()
  #observers = [num_episodes, env_steps]

  #driver = dynamic_step_driver.DynamicStepDriver(train_env, random_policy, observers, num_steps=2)

  ## (Optional) Optimize by wrapping some of the code in a graph using TF function.
  agent.train = common.function(agent.train)

  ## Reset the train step
  agent.train_step_counter.assign(0)

  ## Evaluate the agent's policy once before training.
  avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  returns = [avg_return]

  for _ in tqdm(range(num_iterations)):

    ## Collect a few steps using collect_policy and save to the replay buffer.
    #for _ in range(collect_steps_per_iteration):
      #collect_step(train_env, agent.collect_policy)
    
    ## Using collect_driver instead of collect_step method
    collect_driver.run()

    iterator = iter(dataset)

    ## Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
      returns.append(avg_return)

  # Save results to the file
  train_env.pyenv.envs[0].gym.get_support_result_logger().save_to_csv()

  # Save weights of learned q-model weights
  #model_filename = "learned_Q_Network"
  #if artery_prob:
      #model_filename += "_artery"
  #else:
      #model_filename += "_eqstiff"
  #model_filename += ".h5"

  #agent._q_network.save(model_filename, save_format='h5')

  # Save just the policy
  tf_policy_saver.save(policy_dir)

  # Save run returns
  returns_run.append(returns)

## Visualize Training
returns_array = np.array(returns_run)
steps = range(0, num_iterations + 1, eval_interval)
plt.figure()
plt.plot(steps, np.mean(returns_run, axis=0))
plt.fill_between(steps, np.mean(returns_run, axis=0) - 3*np.std(returns_run, axis=0), np.mean(returns_run, axis=0) + 3*np.std(returns_run, axis=0))
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.show()
