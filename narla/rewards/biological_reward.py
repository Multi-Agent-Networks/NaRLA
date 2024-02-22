from __future__ import annotations

import abc

import torch

import narla
from narla.rewards import Reward


class BiologicalReward(Reward):
    """
    BiologicalRewards are based on activity and dynamics within the Network
    """

    @abc.abstractmethod
    def compute(self, network: narla.multi_agent_network.MultiAgentNetwork, layer_index: int) -> torch.Tensor:
        """
        Compute the reward for the specified Layer

        :param network: Network
        :param layer_index: Index of the Layer to computer rewards for
        """
