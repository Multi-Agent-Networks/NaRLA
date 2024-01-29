from __future__ import annotations

import copy
import torch
from narla.neurons import Network as BaseNetwork


class Network(BaseNetwork):
    def __init__(self, input_size: int, output_size: int, embedding_size: int = 128):
        super().__init__()

        self._neural_network = self._build_network(
            input_size=input_size,
            output_size=output_size,
            embedding_size=embedding_size
        )

    @staticmethod
    def _build_network(input_size: int, output_size: int, embedding_size) -> torch.nn.Sequential:
        layers = [
            torch.nn.Linear(input_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, output_size)
        ]

        return torch.nn.Sequential(*layers)

    def clone(self) -> Network:
        cloned_network = copy.deepcopy(self)
        cloned_network._neural_network.load_state_dict(self._neural_network.state_dict())

        return cloned_network

    def forward(self, X: torch.Tensor):
        output = self._neural_network(X)

        action_probability = torch.nn.functional.softmax(output, dim=-1)

        return action_probability
