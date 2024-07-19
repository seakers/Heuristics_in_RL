# -*- coding: utf-8 -*-
"""
Comparing two methods to compute KL divergence between two action distributions by ratio of selecting a particular action
in each distribution
1. Based on logits (Reference: https://keras.io/examples/rl/ppo_cartpole/)
2. Based on probabilities (in math, the logit function maps probabilities [0,1] to the real space [-inf,inf])

@author: roshan94
"""
import numpy as np
import tensorflow as tf

# Assuming 3 actions for current state
n_actions = 3
logits1 = [0.3, 0.1, -0.4] # action distribution for first policy
logits2 = [-0.4, -0.1, 0.5] # action distribution for second policy
action_selected = 1

# Computing policy ratio using logits directly (from reference)
def log_probs(logits, a):
    logprobs_all = tf.nn.log_softmax(logits)
    logprobs = tf.math.reduce_sum(tf.one_hot(a, n_actions) * logprobs_all)
    
    return logprobs

log_probabilities1 = log_probs(logits1, action_selected) # log probability of selecting action in policy 1
log_probabilities2 = log_probs(logits2, action_selected) # log probability of selecting action in policy 2

ratio_log = tf.math.exp(log_probabilities1 - log_probabilities2)

print("Ratio by logits = " + str(ratio_log))

# Computing policy ratio using probabilities
probs1 = np.divide(1, 1 + np.exp(np.multiply(logits1, -1))) # convert logits to corresponding probabilities (based on p = 1/(1+exp(-L)))
probs2 = np.divide(1, 1 + np.exp(np.multiply(logits2, -1)))

ratio_prob = probs1[action_selected]/probs2[action_selected]
#ratio_prob = np.exp(probs1[action_selected] - probs2[action_selected])

print("Ratio by probabilities = " + str(ratio_prob))


