from __future__ import annotations

import abc
import torch
import narla


class Neuron(metaclass=abc.ABCMeta):
    """
    Neuron base class

    :param observation_size: Size of the observation which the Neuron will receive
    :param number_of_actions: Number of actions available to the Neuron
    """
    def __init__(self, observation_size: int, number_of_actions: int, learning_rate: float):
        self.observation_size = observation_size
        self.number_of_actions = number_of_actions

        self._history = narla.history.History()

    @abc.abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Take an action based on the observation

        :param observation: Observation from the Neuron's local environment
        """
        pass

    @property
    def history(self) -> narla.history.History:
        """
        Access the History of the Neuron
        """
        return self._history

    @abc.abstractmethod
    def learn(self):
        """
        Update the Neuron's internal Network
        """
        pass

    def record(self, **kwargs):
        """
        Record information in the Neuron's internal History. This stored data will be used for learning

        :param kwargs: Keyword arguments
        """
        self._history.record(**kwargs)
