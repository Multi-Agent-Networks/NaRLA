from __future__ import annotations

import abc
import copy
import torch


class Network(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
