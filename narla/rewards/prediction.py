from __future__ import annotations

import torch

import narla
from narla.rewards.biological_reward import BiologicalReward


class Prediction(BiologicalReward):
    """
    Rewards Neurons in the layer for predicting next layer activity
    """

    def compute(self, network: narla.multi_agent_network.MultiAgentNetwork, layer_index: int) -> torch.Tensor:
        current_layer = network.layers[layer_index]
        # Can't reward last layer
        if layer_index == narla.experiment_settings.multi_agent_network_settings.number_of_layers - 1:
            return current_layer.layer_output * 0

        next_layer = network.layers[layer_index + 1]
        # Non-active Neurons in next_layer provide current_layer Neurons with a negative reward if they were active
        next_layer_output = next_layer.layer_output.clone()
        next_layer_output[next_layer_output == 0] = -1

        # Map the output from current_layer through next_layer's connectivity
        # Each column (e.g. next_layer_input[:, i]) will correspond to a Neuron's input in next_layer
        next_layer_input = current_layer.layer_output.T * next_layer.connectivity
        # Sum over columns to get per reward in current_layer
        prediction_reward = torch.sum(next_layer_input * next_layer_output, dim=-1)

        return prediction_reward[None]
