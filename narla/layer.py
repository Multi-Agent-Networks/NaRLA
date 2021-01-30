"""
Layer
---
Implements a holder for a group of Nodes
Forward pass through the layer computes input to all nodes
Learning step updates the parameters of each node in the layer
"""

import numpy as np
import tensorflow as tf
from . import utils
from .node import Node
from .settings import DISPLAY_REWARD_NAMES


class Layer:
    def __init__(self, num_nodes=1, layer_id=0, prev_layer_size=1, start_or_end_layer=False, w=3):
        self.num_nodes = num_nodes
        self.layer_id = layer_id
        self.layer_name = f'Layer{layer_id}'
        self.nodes = []
        self.layer_opts  = []

        connectivity = np.zeros((num_nodes, prev_layer_size))
        prev_layer_idxs = np.arange(prev_layer_size)

        for node_id in range(num_nodes):

            # DETERMINE RECEPTIVE FIELD FROM INPUT SHAPE
            s_idx = max(node_id-w, 0)
            e_idx = min(node_id+w+1, prev_layer_size)

            input_idxs = prev_layer_idxs[s_idx:e_idx]
            if start_or_end_layer:
                input_idxs = prev_layer_idxs

            connectivity[node_id, input_idxs] = 1

            self.nodes.append(
                Node(
                    node_id=node_id,
                    name=f'l{layer_id}_n{node_id}',
                    input_idxs=input_idxs,
                )
            )
            self.layer_opts.append(
                tf.keras.optimizers.Adam(
                    learning_rate=0.01,
                    # clipvalue=0.5
                )
            )
        self.connectivity = tf.convert_to_tensor(
            connectivity,
            dtype=tf.float32
        )

        self.reset()

    def store_bio_reward(self, **kwargs):
        for k,r in kwargs.items():
            self.bio_rewards[k].append(r)

    def reset(self):
        self.bio_rewards = {
            k:[] for k in DISPLAY_REWARD_NAMES
        }

    def init_cumm_grads(self):
        cummulative_gradients = []

        for node in self.nodes:
            cummulative_gradients.append([
                tf.zeros_like(var) for var in node.trainable_variables
            ])

        return cummulative_gradients

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, X):
        layer_outputs = []

        with tf.name_scope(self.layer_name) as scope:

            for i, node in enumerate(self.nodes):
                x = tf.gather(
                    params=X,
                    indices=node.input_idxs,
                    axis=-1
                )

                x, _ = node(x)
                x = tf.identity(
                    x, node.name + '_output'
                )

                layer_outputs.append(x)

            output = tf.reshape(
                tf.concat(layer_outputs, axis=-1),
                (1, -1)
            )

        return output

    @tf.function(experimental_relax_shapes=True)
    def learn(self, layer_input, layer_output, rewards):
        layer_losses = 0
        layer_grads = []

        for i, node in enumerate(self.nodes):

            with tf.GradientTape() as tape:
                x = tf.gather(
                    params=layer_input,
                    indices=node.input_idxs,
                    axis=-1
                )

                _, prob = node(x)
                action = tf.gather_nd(layer_output, [0, i])
                reward = tf.gather_nd(rewards, [0, i])
                loss = utils.pg_action_loss(prob, action, reward)

            layer_losses += loss
            layer_grads.append(
                tape.gradient(loss, node.trainable_variables)
            )

        return layer_losses, layer_grads

    @tf.function(experimental_relax_shapes=True)
    def update(self, grads):
        for node_idx, node in enumerate(self.nodes):
            self.layer_opts[node_idx].apply_gradients(
                zip(grads[node_idx], node.trainable_variables)
            )
