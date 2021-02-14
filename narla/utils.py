"""
Utils
---
Supporting functions for NaRLA
"""

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def create_connectivity_matrix(rows, cols, max_connections=3):
    all_diag_idxs = []
    connectivity = np.zeros((rows, cols))

    for sum_ in range(rows + cols - 1):
        diag_idxs = []
        for k in range(sum_ + 1):
            if (sum_ - k) < rows and k < cols:
                diag_idxs.append((sum_-k, k))
        all_diag_idxs.append(diag_idxs)

    for i, diag_idxs in enumerate(all_diag_idxs):
        if len(diag_idxs) > max_connections:
            extra = int(
                (len(diag_idxs) - 2) / 2
            )

            new_diag_idxs = diag_idxs[extra:-extra]
        else:
            new_diag_idxs = diag_idxs

        for diag_idx in new_diag_idxs:
            connectivity[diag_idx] = 1

    return connectivity

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

    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--reward_type', default='task', choices=['task','bio'])

    return parser.parse_args()
