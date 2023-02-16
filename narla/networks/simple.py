import torch
import narla
from narla.networks import Network


class Simple(Network):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self._neural_network = self._build_network(
            input_size=input_size,
            output_size=output_size
        )

    @staticmethod
    def _build_network(input_size: int, output_size: int, embedding_size: int = 128) -> torch.nn.Sequential:
        layers = [
            torch.nn.Linear(input_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, output_size)
        ]

        return torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        output = self._neural_network(X)

        return output
