"""
Utils
---
Supporting functions for NaRLA
"""

import argparse
import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def pg_action_loss(prob, action, reward):
    dist = tfp.distributions.Categorical(
        probs=prob,
        dtype=tf.float32
    )

    log_prob = dist.log_prob(action)
    loss = -log_prob*reward

    return loss

def prepare_state(state):
    return tf.convert_to_tensor(
        state.reshape(1, -1),
        dtype=tf.float32
    )

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--repeat_num', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=50000)

    parser.add_argument('--num_nodes', type=int, default=15)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--receptive_window', type=int, default=1)

    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--reward_type', default='task', choices=['task','bio'])

    return parser.parse_args()
