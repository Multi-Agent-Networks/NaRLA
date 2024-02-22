from __future__ import annotations

import abc
import torch
import narla


class Reward(metaclass=abc.ABCMeta):
    """
    Reward
    """

    @abc.abstractmethod
    def compute(
        self,
        current_layer: narla.multi_agent_network.Layer,
        next_layer: narla.multi_agent_network.Layer
    ) -> torch.Tensor:
        """
        Compute the reward based on the Layer's activity

        :param current_layer: Current layer
        :param next_layer: Next layer, is ``None`` if ``current_layer`` is the last layer
        """
        pass
