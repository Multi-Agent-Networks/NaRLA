"""
Multi Agent Network (MAN)
---
Construct a MAN with specified number of layers
The whole structure is implemented as a single TensorFlow graph
Train each agent in the network with computed reward signals
"""

import numpy as np
import tensorflow as tf
from .layer import Layer

class MultiAgentNetwork:
    def __init__(self, input_size, num_layers=3, num_nodes_per_layer=10):
        self.layers = []
        self.num_layers = num_layers
        self.input_size = prev_layer_size = input_size
        self.architecture = [num_nodes_per_layer]*(num_layers - 1) + [1]

        for layer_id, num_nodes in enumerate(self.architecture):
            cur_layer = Layer(
                num_nodes=num_nodes,
                layer_id=layer_id,
                prev_layer_size=prev_layer_size,
                start_or_end_layer=layer_id == 0 or (layer_id+1) == num_layers,
                w=3,
            )

            self.layers.append(cur_layer)
            prev_layer_size = num_nodes

        self.reset()

    def reset(self):
        if hasattr(self, 'layer_inputs'):
            del self.layer_inputs
        self.layer_inputs = []

        if hasattr(self, 'layer_outputs'):
            del self.layer_outputs
        self.layer_outputs = []

        if hasattr(self, 'rewards'):
            del self.rewards
        self.rewards = []

    def __call__(self, x):
        layer_tensors = self._compute_forward(x)

        self.layer_inputs.append(layer_tensors[:-1])
        self.layer_outputs.append(layer_tensors[1:])

        return layer_tensors[-1]

    @tf.function
    def _compute_forward(self, x):
        layer_tensors = [x]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            layer_tensors.append(x)

        return layer_tensors

    def record_reward(self, R):
        self.rewards.append(R)

    def discount_rewards(self):
        sum_reward = 0
        discnt_rewards = []

        for r in self.rewards[::-1]:
            sum_reward = r + .99*sum_reward
            discnt_rewards.insert(0, sum_reward)

        if len(discnt_rewards) > 3:
            discnt_rewards = np.array(discnt_rewards)
            discnt_rewards = (discnt_rewards - discnt_rewards.mean()) / discnt_rewards.std()
            discnt_rewards = discnt_rewards.tolist()

        self.returns = discnt_rewards# [::-1]

    def learn(self):
        self.discount_rewards()

        cummulative_gradients = self._compute_backward(
            self.layer_inputs,
            self.layer_outputs,
            self.returns
        )

        for layer, grads in zip(self.layers, cummulative_gradients):
            layer.update(grads)

        self.reset()
        return cummulative_gradients

    def _compute_backward(self, all_layer_inputs, all_layer_outputs, all_returns):
        num_samples = len(self.returns)

        cummulative_gradients = []
        for layer in self.layers:
            cummulative_gradients.append(
                layer.init_cumm_grads()
            )

        zipped_up_1 = zip(
            range(num_samples),
            all_layer_inputs,
            all_layer_outputs,
            all_returns,
        )

        for sample_i, layer_inputs, layer_outputs, reward in zipped_up_1:

            zipped_up_2 = zip(
                range(self.num_layers),
                layer_inputs,
                layer_outputs,
            )

            for layer_idx, layer_input, layer_output in zipped_up_2:

                layer_losses, layer_grads = self.layers[layer_idx].learn(
                    layer_input=layer_input,
                    layer_output=layer_output,
                    reward=tf.constant(reward, dtype=tf.float32)
                )

                for node_idx in range(len(cummulative_gradients[layer_idx])):

                    cummulative_gradients[layer_idx][node_idx] = [
                        cgrad + (grad / num_samples) for cgrad, grad in zip(
                                cummulative_gradients[layer_idx][node_idx],
                                layer_grads[node_idx]
                            )
                    ]

        return cummulative_gradients
