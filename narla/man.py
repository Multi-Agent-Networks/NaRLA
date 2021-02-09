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
from .settings import DISPLAY_REWARD_NAMES

class MultiAgentNetwork:
    def __init__(self, args, input_size):
        self.layers = []
        self.num_layers = args.num_layers
        self.input_size = prev_layer_size = input_size
        self.architecture = [args.num_nodes]*(args.num_layers - 1) + [1]
        self.use_bio_rewards = args.reward_type == 'bio'
        self.sparsity_goal   = 0.2

        self.fire_s = 1. #/ num_nodes_per_layer
        self.pred_s = 1. #/ num_nodes_per_layer
        self.sparse_s = 1. #/ num_nodes_per_layer
        self.trace_s = 1.

        self.gamma = tf.constant(.99)

        for layer_id, num_nodes in enumerate(self.architecture):
            cur_layer = Layer(
                num_nodes=args.num_nodes,
                layer_id=layer_id,
                prev_layer_size=prev_layer_size,
                start_or_end_layer=layer_id == 0 or (layer_id+1) == args.num_layers,
                w=args.receptive_window,
            )

            self.layers.append(cur_layer)
            prev_layer_size = args.num_nodes

        self.reset()
        self.layer_connectivity = [
            layer.connectivity for layer in self.layers
        ]

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

        for layer in self.layers:
            layer.reset()

    def __call__(self, x):
        layer_tensors = self._compute_forward(x)

        self.layer_inputs.append(layer_tensors[:-1])
        self.layer_outputs.append(layer_tensors[1:])

        self._compute_bio_rewards()

        return int(
            layer_tensors[-1].numpy()[0]
        )

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
        discounted_rewards = []
        for layer in self.layers:
            layer_rewards = []
            layer_sum_r = tf.zeros((1, layer.num_nodes))

            for sample_i in range(len(self.rewards)-1, -1, -1):

                reward = self.rewards[sample_i]
                sample_r = tf.ones((1, layer.num_nodes)) * reward

                # USE BIO REWARDS ; ADD TO TOTAL
                if self.use_bio_rewards:
                    pred_r = layer.bio_rewards['pred_reward'][sample_i]
                    fire_r = layer.bio_rewards['fire_reward'][sample_i]
                    trace_r = layer.bio_rewards['trace_reward'][sample_i]
                    sparse_r = layer.bio_rewards['sparse_reward'][sample_i]
                    # print(fire_r + pred_r + sparse_r + trace_r)
                    sample_r = fire_r + pred_r + sparse_r + trace_r + reward

                # DISCOUNT REWARDS
                layer_sum_r = sample_r + self.gamma * layer_sum_r
                layer_rewards.insert(0, layer_sum_r)

            layer_rewards = tf.concat(
                layer_rewards, axis=0
            )

            # NORMALIZE REWARDS
            if layer_rewards.shape[0] > 3:
                mean = tf.reduce_mean(layer_rewards, axis=0, keepdims=True)
                std  = tf.math.reduce_std(layer_rewards, axis=0, keepdims=True)

                layer_rewards = (layer_rewards - mean) / std

            discounted_rewards.append(layer_rewards)

        return discounted_rewards

    def learn(self):
        returns = self.discount_rewards()

        cummulative_gradients = self._compute_backward(
            all_layer_inputs=self.layer_inputs,
            all_layer_outputs=self.layer_outputs,
            all_returns=returns
        )

        for layer, grads in zip(self.layers, cummulative_gradients):
            layer.update(grads)

        self.reset()
        return cummulative_gradients

    def _compute_backward(self, all_layer_inputs, all_layer_outputs, all_returns):
        num_samples = len(all_returns[0])

        cummulative_gradients = []
        for layer in self.layers:
            cummulative_gradients.append(
                layer.init_cumm_grads()
            )

        zipped_up_1 = zip(
            range(num_samples),
            all_layer_inputs,
            all_layer_outputs,
        )

        for sample_i, layer_inputs, layer_outputs in zipped_up_1:

            zipped_up_2 = zip(
                range(self.num_layers),
                self.layers,
                layer_inputs,
                layer_outputs,
                all_returns,
            )

            for layer_idx, layer, layer_input, layer_output, layer_returns in zipped_up_2:

                rewards = tf.expand_dims(
                    tf.gather(layer_returns, sample_i),
                    axis=0
                )

                layer_losses, layer_grads = layer.learn(
                    layer_input=layer_input,
                    layer_output=layer_output,
                    rewards=rewards
                )

                for node_idx in range(len(cummulative_gradients[layer_idx])):

                    cummulative_gradients[layer_idx][node_idx] = [
                        cgrad + (grad / num_samples) for cgrad, grad in zip(
                                cummulative_gradients[layer_idx][node_idx],
                                layer_grads[node_idx]
                            )
                    ]

        return cummulative_gradients

    def _compute_bio_rewards(self):
        firing_reward = []
        sparsity_reward = []
        prediction_reward = []
        activity_trace_reward = []

        zipped_up = zip(
            self.layers,
            self.layer_inputs[-1],
            self.layer_outputs[-1],
            self.layer_connectivity
        )

        # CALCULATE BIO REWARDS
        for i, (layer, inputs, outputs, con) in enumerate(zipped_up):
            on_first_layer = i == 0
            on_last_layer  = i == (self.num_layers - 1)

            # ACTIVITY TRACE
            if len(layer.bio_rewards['fire_reward']) == 0:
                trace = tf.zeros_like(outputs)
            else:
                trace = tf.abs(
                    tf.reduce_mean(
                        tf.concat(
                            layer.bio_rewards['fire_reward'],
                            axis=0
                        ), axis=0, keepdims=True
                    ) - .5
                )
                trace = -1746463 + (0.9256812 - -1746463)/(1 + (trace/236.1357)**2.160334)

            activity_trace_reward.append(trace)

            if not on_first_layer:
                # CALCULATE PREDICTION REWARD
                activity_interaction = (tf.transpose(outputs) @ inputs) * con
                activity_rewards     = tf.reduce_sum(activity_interaction, axis=0, keepdims=True)

                inactive_inputs      = tf.where(inputs == 0, -1., 0.)
                inactive_interaction = (tf.transpose(outputs) @ inactive_inputs) * con
                inactive_penalties   = tf.reduce_sum(inactive_interaction, axis=0, keepdims=True)

                prediction_reward.append(
                    (activity_rewards + inactive_penalties)
                )

            if not on_last_layer:
                # CALCULATE SPARISTY REWARD
                sparsity_error = tf.reduce_sum(outputs) - self.sparsity_goal
                sparse_rewards = tf.where(outputs == 0., -1., 1.)

                if sparsity_error > 0:
                    sparsity_rewards = -sparsity_error * sparse_rewards
                elif sparsity_error < 0:
                    sparsity_rewards =  sparsity_error * sparse_rewards
                elif sparsity_error == 0:
                    sparsity_rewards = tf.abs(sparse_rewards) * self.sparsity_goal

                sparsity_reward.append(sparsity_rewards)

                firing_reward.append(outputs)

        firing_reward.append(tf.zeros_like(outputs))
        prediction_reward.append(tf.zeros_like(outputs))
        sparsity_reward.append(tf.zeros_like(outputs))

        zipped_up = zip(
            self.layers,
            firing_reward,
            prediction_reward,
            sparsity_reward,
            activity_trace_reward,
        )

        # DISTRIBUTE BIO REWARDS TO LAYERS
        for layer, fire_r, pred_r, sparse_r, trace_r in zipped_up:
            layer.store_bio_reward(
                fire_reward=self.fire_s * fire_r,
                pred_reward=self.pred_s * pred_r,
                sparse_reward=self.sparse_s * sparse_r,
                trace_reward=self.trace_s * trace_r,
            )
