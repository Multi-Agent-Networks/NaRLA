from __future__ import annotations

import torch
import narla


class Neuron:
    def __init__(self, observation_size: int, number_of_actions: int):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._history = narla.history.History()

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        pass

    def learn(self):
        pass

    def record(self, **kwargs):
        self._history.record(**kwargs)
