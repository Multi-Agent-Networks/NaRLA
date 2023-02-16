from __future__ import annotations

import abc
import copy
import torch


class Network(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

        self._neural_network: torch.nn.Module = None

    @staticmethod
    @abc.abstractmethod
    def _build_network(input_size: int, output_size: int, embedding_size: int = 32) -> torch.nn.Sequential:
        pass

    def clone(self) -> Network:
        cloned_network = copy.deepcopy(self)
        cloned_network._neural_network.load_state_dict(self._neural_network.state_dict())

        return cloned_network

    @abc.abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
