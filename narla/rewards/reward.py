from __future__ import annotations

import abc

import torch


class Reward(metaclass=abc.ABCMeta):
    """
    Reward
    """

    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the reward
        """
        pass
