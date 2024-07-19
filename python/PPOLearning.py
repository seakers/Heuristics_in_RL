# -*- coding: utf-8 -*-
"""
PPO training main script for both the metamaterial and satelite problems
Adapted from https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial,https://www.tensorflow.org/agents/tutorials/9_c51_tutorial and 
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial 
with parellelization based on https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial

@author: roshan94
"""
import math
import numpy as np
from envs.ArteryProblemEnv import ArteryProblemEnv
from envs.EqualStiffnessProblemEnv import EqualStiffnessProblemEnv
from tqdm import tqdm

import matplotlib.pyplot as plt

import keras

import tensorflow as tf
import os

from tf_agents.environments.gym_wrapper import GymWrapper
#from tf_agents.environments.suite_gym import wrap_env
from tf_agents.specs import tensor_spec

from tf_agents.train import actor
#from tf_agents.train import learner
from custom.custom_learner import CustomLearner
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils

from tf_agents.agents import PPOKLPenaltyAgent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

#from tf_agents.agents.ppo import ppo_actor_network

#from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.train.utils import strategy_utils
#from tf_agents.drivers import dynamic_step_driver
#from tf_agents.policies import random_tf_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common

from tf_agents.environments import batched_py_environment

#from tf_agents.trajectories import trajectory

from save.ResultSaving import ResultSaver

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import GatewayParameters

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

render_steps = True

## find number of states and actions based on sidenum
n_states = math.comb(sidenum**2, 2) - 2*math.comb(sidenum, 2) # number of states = number of design variables 
n_actions = n_states + n_heurs_used # number of actions = number of design variables (an action corresponds to flipping the corresponding bit of the binary design decision)
#max_steps = 1000

n_runs = 1

compute_periodic_returns = True

## PPO Hyperparameters
num_unique_des_episode = 1000

initial_collect_steps = 750 # number of steps in the driver to populate replay buffer
episode_collect_steps = 8 # number of steps in the driver to populate replay buffer during training
replay_buffer_capacity = 10000 

batch_size = 32 # batch size for replay buffer storage and training
n_step_update = 32 # number of transition steps in a trajectory used to train the agent

n_epochs = 5 # Number of epochs for computing policy updates, also equals agent step increment

# Default values
initial_adaptive_kl_beta = 1.0
adaptive_kl_target = 0.01
adaptive_kl_tolerance = 0.5 # heuristically chosen in the original paper
log_prob_clipping = 0.05 
kl_cutoff_coef = 0.2
kl_cutoff_factor = 0.01

ppo_name = 'ppokl_'

actor_layer_params = (200, 120)
actor_dropout_layer_params = (0.10, 0.05)

critic_layer_params = (100, 60)
critic_dropout_layer_params = (0.10, 0.05)

learning_rate = 1e-3  
epsilon_decay_steps = 50
gamma = 0.99
prob_eps_greedy = 0.05

num_eval_episodes = 3  
num_train_epsiodes = 10

eval_interval = 4

policy_save_interval = 4

n_episodes_per_fig = 4 # used for plotting returns and losses 
linestyles = ['solid','dotted','dashed','dashdot']

#if use_replay:
    #max_steps = (num_iterations+1)*initial_collect_steps # total number of designs generated in an episode is num_iterations*initial_collect_steps (in each iteration {initial_collect_steps} designs are evaluated and added to the replay buffer)
    # + the initial number of designs added to the buffer before training
#else:
    #max_steps = num_iterations*(n_step_update+1) # total number of designs generated in an episode is num_iterations*n_step_update

max_steps = np.Inf # No termination in training environment

max_steps_eval = 50

## Access java gateway and pass parameters to operations class instance
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
operations_instance = gateway.entry_point.getOperationsInstance()

run_train_steps = {}
losses_run = {}
#avg_losses_run = []

if compute_periodic_returns:
    run_eval_steps = {}
    returns_run = {}
    #avg_returns_run = []

for run_num in range(n_runs): 
    print('Run ' + str(run_num))

    current_save_path = save_path + "run " + str(run_num)

    if not os.path.exists(current_save_path):
        os.mkdir(current_save_path)

    file_name = "RL_training_designs_" + str(ppo_name)
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
        train_env = batched_py_environment.BatchedPyEnvironment([GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma)])
        eval_env = batched_py_environment.BatchedPyEnvironment([GymWrapper(ArteryProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps_eval, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma)])
        #train_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
        #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(ArteryProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
    else:
        train_env = batched_py_environment.BatchedPyEnvironment([GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma)])
        eval_env = batched_py_environment.BatchedPyEnvironment([GymWrapper(EqualStiffnessProblemEnv(operations_instance=operations_instance, n_actions=n_actions, n_states=n_states, max_steps=max_steps_eval, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma)])
        #train_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))
        #eval_env = tf_py_environment.TFPyEnvironment(wrap_env(EqualStiffnessProblemEnv(n_actions=n_actions, n_states=n_states, max_steps=max_steps, model_sel=model_sel, sel=sel, sidenum=sidenum, rad=rad, E_mod=E_mod, c_target=c_target, nuc_fac=nucFac, save_path=save_path, obj_names=obj_names, constr_names=constr_names, heur_names=heur_names, heurs_used=heurs_used, render_steps=render_steps), discount=gamma))

    ## initialize MirroredStrategy for distributed training
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False) # Set to False if no GPUs

    ## Define Actor and Critic Network
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
           input_tensor_spec=train_env.observation_spec(),
           output_tensor_spec=tensor_spec.from_spec(train_env.action_spec()),  # weird issue where output_tensor_spec has to be an instance of TensorSpec instead of ArraySpec, but Actor doesn't support TFEnviornments (PyEnvironments output ArraySpecs as default)
           fc_layer_params=actor_layer_params,
           dropout_layer_params=actor_dropout_layer_params,
           activation_fn=tf.keras.activations.softmax,)
        # actor_net_builder = ppo_actor_network.PPOActorNetwork()
        # actor_net = actor_net_builder.create_sequential_actor_net(fc_layer_units=actor_layer_params, action_tensor_spec=tensor_spec.from_spec(train_env.action_spec()))

    with strategy.scope():
        critic_net = value_network.ValueNetwork(
            input_tensor_spec=train_env.observation_spec(),
            fc_layer_params=critic_layer_params,
            dropout_layer_params=critic_dropout_layer_params,)
        
    def dense_layer(num_units):
        return tf.keras.layers.Dense(num_units, activation=tf.keras.activations.relu)

    save_actor_net = keras.Sequential()
    #dense_layers = [dense_layer(num_units) for num_units in actor_layer_params]
    #input_layer = tf.keras.layers.Dense(n_states, activation=tf.keras.activations.relu)
    action_dist_layer = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.relu)
    
    save_actor_net.add(tf.keras.Input(shape=(n_states,))) # Input required to initialize weights, since this model is just used to verify policy saving at the end
    for n_units in actor_layer_params:
        save_actor_net.add(dense_layer(n_units))
    save_actor_net.add(action_dist_layer)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    ## Define PPO Agent
    with strategy.scope():

        train_step = train_utils.create_train_step()
       
        ## Define optimizer
        ## Define exponential decay of epsilon
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=epsilon_decay_steps, decay_rate=0.96)

        #optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
        #optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.96, momentum=0.001)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.96, momentum=0.001)

        agent = PPOKLPenaltyAgent(
              time_step_spec=tensor_spec.from_spec(train_env.time_step_spec()),
              action_spec=tensor_spec.from_spec(train_env.action_spec()), # weird issue where output_tensor_spec has to be an instance of TensorSpec instead of ArraySpec, but Actor doesn't support TFEnviornments (PyEnvironments output ArraySpecs as default)
              actor_net=actor_net,
              value_net=critic_net,
              num_epochs=n_epochs,
              initial_adaptive_kl_beta=initial_adaptive_kl_beta,
              adaptive_kl_target=adaptive_kl_target,
              adaptive_kl_tolerance=adaptive_kl_tolerance,
              log_prob_clipping=log_prob_clipping,
              kl_cutoff_coef=kl_cutoff_coef,
              kl_cutoff_factor=kl_cutoff_factor,
              optimizer=optimizer)
           
        agent.initialize()
    
    tf_eval_policy = agent.policy
    tf_collect_policy = agent.collect_policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)

    ## Define random policy for initial trajectory addition to the replay buffer 
    #explore_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    ## Define Replay Buffer for PPO training
    # replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    #     data_spec=agent.collect_data_spec,
    #     batch_size=1,
    #     max_length=replay_buffer_capacity)
    replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        capacity=replay_buffer_capacity,
        data_spec=tensor_spec.to_nest_array_spec(agent.collect_data_spec))
        
    ## Adding trajectories using Dynamic Step Driver instead of the manual process above
    train_obs = [replay_buffer.add_batch]
    dataset = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=n_step_update + 1).prefetch(3)

    ## Dataset generates trajectories with shape [BxTx...] where T = n_step_update + 1.
    experience_dataset_fn = lambda: dataset

    ## Create actors for initial replay buffer population, episodic training and evaluation
    initial_collect_actor = actor.Actor(
        env=train_env,
        policy=collect_policy,
        train_step=train_step,
        steps_per_run=initial_collect_steps,
        observers=train_obs)
    
    episodic_collect_actor = actor.Actor(
        env=train_env,
        policy=collect_policy, 
        train_step=train_step,
        steps_per_run=episode_collect_steps,
        metrics=actor.collect_metrics(buffer_size=5),
        summary_dir=os.path.join(current_save_path,"train"),
        observers=train_obs)
    
    eval_actor = actor.Actor(
        env=eval_env,
        policy=eval_policy,
        train_step=train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(buffer_size=num_eval_episodes),
        summary_dir=os.path.join(current_save_path,"eval"))
    
    ## Create learner to train the agent and triggers to save agent policy checkpoints
    saved_model_dir = os.path.join(current_save_path, "savedPolicy")
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir=saved_model_dir,
            agent=agent,
            train_step=train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step=train_step, interval=num_unique_des_episode/100)
    ]

    # Using traditional tensorflow learner
    #agent_learner = learner.Learner(
        #root_dir=current_save_path,
        #train_step=train_step,
        #agent=agent,
        #experience_dataset_fn=experience_dataset_fn,
        #triggers=learning_triggers,
        #strategy=strategy)

    # Using custom tensorflow learner
    agent_learner = CustomLearner(
        root_dir=current_save_path,
        train_step=train_step,
        agent=agent,
        experience_dataset_fn=experience_dataset_fn,
        triggers=learning_triggers,
        direct_sampling=True, # Sampling from replay buffer dataset iterator doesn't produce sample info for some reason
        strategy=strategy)
    
    ## Evaluation method
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()

        return results

    ## (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    ## Reset the train step
    agent.train_step_counter.assign(0)

    returns_episodes = []
    losses_episodes = []

    episode_step_counts = []
    episode_eval_step_counts = []
    created_designs = set()

    ## Initial replay buffer population
    initial_collect_actor.run()

    for ep_num in tqdm(range(num_train_epsiodes)):
       
        first_train_time_step = train_env.reset()
        first_eval_time_step = eval_env.reset()

        returns_steps = []
        losses_steps = []

        return_step_counts = []
        loss_step_counts = []

        #eval_step_count = 0
        step_count = 0

        current_eval_step_count = 0

        with tqdm(total=num_unique_des_episode) as pbar:
            while (step_count < num_unique_des_episode): # Stop current episode if requisite number of steps is taken

                ## Training step
                episodic_collect_actor.run()
                train_loss = agent_learner.run(iterations=1)
                experience = agent_learner.current_experience_sample

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
                    if current_eval_step_count == eval_interval: 
                        # Evaluate agent's policy 
                        avg_return = get_eval_metrics()["AverageReturn"]
                        print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                        returns_steps.append(avg_return)
                        return_step_counts.append(step)
                        current_eval_step_count = 0
                        #eval_step_count += 1
                    else:
                        current_eval_step_count += 1

        episode_step_counts.append(loss_step_counts)
        episode_eval_step_counts.append(return_step_counts)

        losses_episodes.append(losses_steps)
        returns_episodes.append(returns_steps)

    # Save results to the file
    result_logger.save_to_csv()

    # Save weights of learned q-model weights
    save_actor_net.set_weights(agent.actor_net.get_weights())
    model_filename = "learned_actor_Network"
    if artery_prob:
        model_filename += "_artery"
    else:
        model_filename += "_eqstiff"
    model_filename += ".keras"

    save_actor_net.save(os.path.join(current_save_path, model_filename))

    # Save run returns and losses
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
      plt.savefig(save_path + "ppo_run" + str(i) + "_fig" + str(figure_counter) + "avg_returns.png", dpi=600)
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
    plt.savefig(save_path + "ppo_run" + str(i) + "_fig" + str(figure_counter) + "avg_losses.png", dpi=600)
    figure_counter += 1