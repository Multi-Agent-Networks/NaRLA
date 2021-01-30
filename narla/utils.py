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
    parser.add_argument('--update_every', type=int, default=25)
    parser.add_argument('--num_episodes', type=int, default=100000)

    parser.add_argument('--num_nodes', type=int, default=15)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--reward_type', default='task', choices=['all','task','bio','bio_then_all'])

    return parser.parse_args()
