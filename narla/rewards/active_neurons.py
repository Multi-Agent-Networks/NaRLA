from __future__ import annotations

import torch

import narla
from narla.rewards.biological_reward import BiologicalReward


class ActiveNeurons(BiologicalReward):
    """
    Rewards Neurons for becoming active
    """

    def compute(self, network: narla.multi_agent_network.MultiAgentNetwork, layer_index: int) -> torch.Tensor:
        return network.layers[layer_index].layer_output
