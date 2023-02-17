from __future__ import annotations

import torch
from typing import Tuple
from narla.neurons import Network as BaseNetwork


class Network(BaseNetwork):
    def __init__(self, input_size: int, output_size: int, embedding_size: int = 128):
        super().__init__()

        self._backbone = self._build_backbone(
            input_size=input_size,
            embedding_size=embedding_size
        )

        self._value_head = torch.nn.Linear(embedding_size, 1)
        self._action_head = torch.nn.Linear(embedding_size, output_size)

    @staticmethod
    def _build_backbone(input_size: int, embedding_size: int = 128) -> torch.nn.Sequential:
        layers = [
            torch.nn.Linear(input_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU()
        ]

        return torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone_embedding = self._backbone(X)

        value = self._value_head(backbone_embedding)

        action_embedding = self._action_head(backbone_embedding)
        action_probability = torch.nn.functional.softmax(action_embedding, dim=-1)

        return action_probability, value
