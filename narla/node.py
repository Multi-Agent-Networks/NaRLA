"""
Node
---
Implements a single agent Node
Each node contains its own neural network
"""

import tensorflow as tf
import tensorflow_probability as tfp

class Node(tf.Module):
    def __init__(self, node_id=0, name=None, input_idxs=[]):
        super(Node, self).__init__(name=name)
        self.node_id = node_id
        self.node_name = name
        self.input_idxs = tf.convert_to_tensor(input_idxs)

        with self.name_scope:
            self.layers = [
                tf.keras.layers.Dense(128),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(
                    2, activation='softmax'
                )
            ]

    @tf.function
    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        dist = tfp.distributions.Categorical(
            probs=x,
            dtype=tf.float32
        )

        return dist.sample(), x
