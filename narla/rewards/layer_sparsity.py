from __future__ import annotations

import torch

import narla
from narla.rewards.biological_reward import BiologicalReward


class LayerSparsity(BiologicalReward):
    """
    Rewards Neurons in the layer for appropriate sparsity
    """

    def __init__(self, desired_sparsity: float = 0.2):

        self.desired_sparsity = desired_sparsity

    def compute(self, network: narla.multi_agent_network.MultiAgentNetwork, layer_index: int) -> torch.Tensor:
        current_layer = network.layers[layer_index]

        number_of_active_neurons = torch.sum(current_layer.layer_output)
        sparsity = number_of_active_neurons / current_layer.number_of_neurons
        sparsity_error = sparsity - self.desired_sparsity

        if sparsity_error > 0:
            sparsity_rewards = -sparsity_error * current_layer.layer_output

        elif sparsity_error < 0:
            sparsity_rewards = sparsity_error * current_layer.layer_output

        else:
            sparsity_rewards = current_layer.layer_output.abs() * self.desired_sparsity

        return sparsity_rewards
