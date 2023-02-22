from __future__ import annotations

import abc
import torch
import narla


class Neuron(metaclass=abc.ABCMeta):
    def __init__(self, observation_size: int, number_of_actions: int):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._history = narla.history.History()

    @abc.abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    def record(self, **kwargs):
        self._history.record(**kwargs)
