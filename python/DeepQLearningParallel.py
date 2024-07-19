# -*- coding: utf-8 -*-
"""
Deep Q-learning main script for both the metamaterial and satelite problems
Adapted from https://www.tensorflow.org/agents/tutorials/9_c51_tutorial and https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial 
with parellelization based on https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial

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
import keras
from keras import layers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments.gym_wrapper import GymWrapper
#from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import tf_py_environment
#from tf_agents.networks import sequential
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.train.utils import strategy_utils
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
#from tf_agents.trajectories import trajectory
#from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tf_agents.trajectories import trajectory

from save.ResultSaving import ResultSaver

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

#from tensorflow.python.client import device_lib

artery_prob = False # If true -> artery problem, false -> equal stiffness problem

save_path = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results\\"

#print(device_lib.list_local_devices())

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
heur_abbr = ['p','n','o','i']
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

n_runs = 1

use_replay = True
compute_periodic_returns = True

## DQN Hyperparameters
num_unique_des_episode = 1000

initial_collect_steps = 50 # number of steps in the driver to populate replay buffer before beginning training
episode_collect_steps = 8 # number of steps in the driver to populate replay buffer during training
replay_buffer_capacity = 10000 

fc_layer_params = (200, 120)
dropout_layer_params = (0.10, 0.05)

batch_size = 16 # batch size for replay buffer storage and training
learning_rate = 1e-3  
epsilon_decay_steps = 50
gamma = 0.99
prob_eps_greedy = 0.05

num_eval_episodes = 3  
num_train_epsiodes = 10

use_target_q_net = True
target_update_steps = 5 # number of steps to update target Q-network (steps here is steps as computed to train agent)

eval_interval = 4 

n_step_update = 16 # number of transition steps in a trajectory used to train the agent

n_episodes_per_fig = 4 # used for plotting returns and losses 
linestyles = ['solid','dotted','dashed','dashdot']

#if use_replay:
  #max_steps = (num_iterations+1)*initial_collect_steps + 1000 # total number of designs generated in an episode is num_iterations*initial_collect_steps (in each iteration {initial_collect_steps} designs are evaluated and added to the replay buffer)
  ## + the initial number of designs added to the buffer before training (additional 1000 steps added to account for possible repeated designs)
#else:
  #max_steps = num_iterations*(n_step_update+1) + 1000 # total number of designs generated in an episode is num_iterations*n_step_update (additional 1000 steps added to account for possible repeated designs)
max_steps = np.Inf # no termination in training environment

max_steps_eval = 50

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

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

run_train_steps = {}
losses_run = {}
#avg_losses_run = []

if compute_periodic_returns:
  run_eval_steps = {}
  returns_run = {}
  #avg_returns_run = []


## Define method to generate a random trajectory to train agent (if not using replay buffer)
def generate_trajectory(env, first_time_step, num_steps, collect_policy):
    observations = []
    rewards = []
    actions = []
    next_observations = []
    discounts = []
    step_types = []
    next_step_types = []

    step = 0
    time_step = first_time_step # Start from a random observation
    while(step < num_steps):
        observations.append(time_step.observation._numpy()[0])
        action_step = collect_policy.action(time_step)
        actions.append(action_step.action._numpy()[0])
        next_time_step = env.step(action_step.action)
        rewards.append(next_time_step.reward._numpy()[0])
        next_observations.append(next_time_step.observation._numpy()[0])
        discounts.append(next_time_step.discount._numpy()[0])
        step_types.append(time_step.step_type._numpy()[0])
        next_step_types.append(next_time_step.step_type._numpy()[0])

        time_step = next_time_step
        step += 1
    
    return trajectory.Trajectory(observation=tf.expand_dims(tf.convert_to_tensor(observations),0),
                                action=tf.expand_dims(tf.convert_to_tensor(actions,dtype=np.int64),0),
                                reward=tf.expand_dims(tf.convert_to_tensor(rewards),0),
                                policy_info=action_step.info,
                                discount=tf.expand_dims(tf.convert_to_tensor(discounts),0),
                                step_type=tf.expand_dims(tf.convert_to_tensor(step_types),0),
                                next_step_type=tf.expand_dims(tf.convert_to_tensor(next_step_types),0)), time_step

for run_num in range(n_runs): 

  print('Run ' + str(run_num))

  current_save_path = save_path + "run " + str(run_num)

  if not os.path.exists(current_save_path):
    os.mkdir(current_save_path) 

  file_name = "RL_training_designs_dqn"
  if artery_prob:
      file_name += "_artery_"
  else:
      file_name += "_eqstiff_"

  if n_heurs_used > 0:
     for i in range(len(heur_abbr)):
        if heurs_used[i]:
           file_name += heur_abbr[i]
      
  file_name += str(run_num) + ".csv"

  # Initialize result saver
  result_logger = ResultSaver(save_path=os.path.join(current_save_path, file_name), operations_instance=operations_instance, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names)

  if artery_prob:
      train_env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      eval_env = tf_py_environment.TFPyEnvironment(GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps_eval, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #train_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
  else:
      train_env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      eval_env = tf_py_environment.TFPyEnvironment(GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps_eval, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=current_save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #train_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
      #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

  ## initialize MirroredStrategy for distributed training
  strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True) # Set to False if no GPUs

  ## Define Q-network and target Q-network
  with strategy.scope():
    q_net = q_network.QNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        q_layer_activation_fn=tf.keras.activations.linear,
        dropout_layer_params=dropout_layer_params)
    
  if use_target_q_net:
     with strategy.scope():
      target_q_net = q_network.QNetwork(
          input_tensor_spec=train_env.observation_spec(),
          action_spec=train_env.action_spec(),
          fc_layer_params=fc_layer_params,
          q_layer_activation_fn=tf.keras.activations.linear,
          dropout_layer_params=dropout_layer_params)

  def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units, activation=tf.keras.activations.relu)

  save_q_net = keras.Sequential()
  #dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
  #input_layer = tf.keras.layers.Dense(n_states, activation=tf.keras.activations.relu)
  q_values_layer = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.relu)
  
  save_q_net.add(tf.keras.Input(shape=(n_states,))) # Input required to initialize weights, since this model is just used to verify policy saving at the end
  for n_units in fc_layer_params:
    save_q_net.add(dense_layer(n_units))
  save_q_net.add(q_values_layer)
  
  #save_q_net = keras.Sequential([input_layers] + dense_layers + [q_values_layer])

  #train_step_counter = tf.Variable(0)

  global_step = tf.compat.v1.train.get_or_create_global_step()

  ## Define DQN Agent
  with strategy.scope():
    ## Define exponential decay of epsilon
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=epsilon_decay_steps, decay_rate=0.96)

    ## Define optimizer
    #optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.96, momentum=0.001)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.96, momentum=0.001)

    if use_target_q_net:
      agent = dqn_agent.DqnAgent(
          time_step_spec=train_env.time_step_spec(),
          action_spec=train_env.action_spec(),
          q_network=q_net,
          optimizer=optimizer,
          n_step_update=n_step_update,
          gamma=gamma,
          td_errors_loss_fn=common.element_wise_squared_loss,
          #train_step_counter=train_step_counter,
          target_q_network=target_q_net,
          target_update_period=target_update_steps,
          train_step_counter=global_step,
          epsilon_greedy=prob_eps_greedy)
    else:
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

  policy_dir = os.path.join(current_save_path, ('policy'+str(run_num)))
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)
  
  collect_policy_dir = os.path.join(current_save_path, ('collect policy'+str(run_num)))
  tf_collect_policy_saver = policy_saver.PolicySaver(agent.collect_policy)

  ## Define random policy for initial trajectory addition to the replay buffer
  explore_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
  #explore_policy = agent.collect_policy

  ## Define Replay Buffer for DQN training
  if use_replay:
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
  if use_replay:
    #train_env.reset()
    #eval_env.reset()
    train_obs = [replay_buffer.add_batch]
    dataset = replay_buffer.as_dataset(num_parallel_calls=2, sample_batch_size=batch_size, num_steps=n_step_update + 1).prefetch(3)
    
    ## Dataset generates trajectories with shape [BxTx...] where T = n_step_update + 1.
    iterator = iter(dataset)
    
    collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, explore_policy, train_obs, num_steps=initial_collect_steps)

    episode_collect_driver = dynamic_step_driver.DynamicStepDriver(train_env, explore_policy, train_obs, num_steps=episode_collect_steps)

    #collect_driver.run()

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
  #avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  #returns = [avg_return]

  returns_episodes = []
  losses_episodes = []

  episode_step_counts = []
  episode_eval_step_counts = []
  created_designs = set()

  if use_replay:
    collect_driver.run()

  for ep_num in tqdm(range(num_train_epsiodes)):

    first_train_time_step = train_env.reset()
    first_eval_time_step = eval_env.reset()

    returns_steps = []
    losses_steps = []

    return_step_counts = []
    loss_step_counts = []

    #eval_step_count = 0
    step_count = 0

    #with tqdm(total=num_iterations*batch_size*(n_step_update+1)) as pbar:
    with tqdm(total=num_unique_des_episode) as pbar:
      #while (step_count < num_iterations*batch_size*(n_step_update+1)): # Stop current episode if requisite number of steps is taken
      while (step_count < num_unique_des_episode): # Stop current episode if requisite number of steps is taken
      #for step_num in tqdm(range(num_iterations + 100)):

        ## Collect a few steps using collect_policy and save to the replay buffer.
        #for _ in range(collect_steps_per_iteration):
          #collect_step(train_env, agent.collect_policy)
        
        ## Using collect_driver instead of collect_step method
        if use_replay:
          episode_collect_driver.run()     

          ## Sample a batch of data from the buffer and update the agent's network.
          experience, unused_info = next(iterator)
        else:
          experience, last_time_step = generate_trajectory(env=train_env, first_time_step=first_train_time_step, num_steps=n_step_update+1, collect_policy=explore_policy)
          first_train_time_step = last_time_step # continue to generate next trajectory from the last time step
        
        train_loss = agent.train(experience)

        losses_steps.append(train_loss.loss)

        step = agent.train_step_counter.numpy()
        loss_step_counts.append(step)

        # Save results to logger
        actions = experience.action._numpy()
        observations = experience.observation._numpy()
        rewards = experience.reward._numpy()

        #print('Environment is done = ' + str(train_env.pyenv.envs[0].gym.get_isdone()))
        #print('Internal step number = ' + str(train_env.pyenv.envs[0].gym.get_step_counter()))
        #print('Agent step = ' + str(step))
        for i in range(batch_size):
          for j in range(n_step_update + 1):
            current_state = observations[i][j]
            current_state_str = "".join([str(dec) for dec in current_state])
            # Do not count repeated designs towards NFEs (but these designs are used to train the agent)
            if not current_state_str in created_designs:
              result_logger.save_to_logger(step_number=step_count, action=actions[i][j], prev_obs=current_state, reward=rewards[i][j])
              created_designs.add(current_state_str)
              step_count += 1   
              pbar.update(1)

        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if compute_periodic_returns:
          if (step-1) % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns_steps.append(avg_return)
            return_step_counts.append(step)
            #eval_step_count += 1

    episode_step_counts.append(loss_step_counts)
    episode_eval_step_counts.append(return_step_counts)

    losses_episodes.append(losses_steps)
    returns_episodes.append(returns_steps)
      
  # Save results to the file
  #train_env.pyenv.envs[0].gym.get_support_result_logger().save_to_csv()
  result_logger.save_to_csv()

  # Save weights of learned q-model weights
  save_q_net.set_weights(agent._q_network.get_weights())
  model_filename = "learned_Q_Network"
  if artery_prob:
      model_filename += "_artery"
  else:
      model_filename += "_eqstiff"
  model_filename += ".keras"

  save_q_net.save(os.path.join(current_save_path, model_filename))

  #current_internal_step = train_env.pyenv.envs[0].gym.get_step_counter()
  #print('Current Step = ' + str(current_internal_step))

  # Save just the policy
  tf_policy_saver.save(policy_dir)
  tf_collect_policy_saver.save(collect_policy_dir)

  # Save run returns and losses (not applicable if repeat designs are not considered for NFE)
  #avg_losses_run.append(np.mean(losses_episodes, axis=0)) # averaging over episodes
  #if compute_periodic_returns:
    #avg_returns_run.append(np.mean(returns_episodes, axis=0)) # averaging over episodes

  run_train_steps['run' + str(run_num)] = episode_step_counts
  if compute_periodic_returns:
     run_eval_steps['run' + str(run_num)] = episode_eval_step_counts

  losses_run['run' + str(run_num)] = losses_episodes
  if compute_periodic_returns:
    returns_run['run' + str(run_num)] = returns_episodes

## Visualize Training
if compute_periodic_returns:
  for i in range(n_runs):
    returns_current_run = returns_run['run' + str(i)]
    eval_steps_current_run = run_eval_steps['run' + str(i)]
    episode_counter = 0
    figure_counter = 0
    while episode_counter < num_train_epsiodes:
      plt.figure()
      for k in range(n_episodes_per_fig):
        if episode_counter == num_train_epsiodes: # in case last figure must have less than "n_episodes_per_fig" episodes
          break
        returns_episode = returns_current_run[episode_counter]
        steps = eval_steps_current_run[episode_counter]
        plt.plot(steps, returns_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
        episode_counter += 1
      plt.ylabel('Average Return')
      plt.xlabel('Evaluation Step #')
      plt.grid()
      plt.title('Agent Evaluation: Run ' + str(i))
      plt.legend(loc='best')
      #plt.show()
      plt.savefig(save_path + "dqn_run" + str(i) + "_fig" + str(figure_counter) + "avg_returns.png", dpi=600)
      figure_counter += 1

for i in range(n_runs):
  losses_current_run = losses_run['run' + str(i)]
  train_steps_current_run = run_train_steps['run' + str(i)]
  episode_counter = 0
  figure_counter = 0
  while episode_counter < num_train_epsiodes:
    plt.figure()
    for k in range(n_episodes_per_fig): 
      if episode_counter == num_train_epsiodes: # in case last figure must have less than "n_episodes_per_fig" episodes
          break
      losses_episode = losses_current_run[episode_counter]
      loss_steps = train_steps_current_run[episode_counter] 
      plt.plot(loss_steps, losses_episode, linestyle=linestyles[k], label='Ep. ' + str(episode_counter+1))
      episode_counter += 1
    plt.ylabel('Loss')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.title('Agent Training Loss: Run ' + str(i))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(save_path + "dqn_run" + str(i) + "_fig" + str(figure_counter) + "avg_losses.png", dpi=600)
    figure_counter += 1
