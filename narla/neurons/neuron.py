from __future__ import annotations

import torch
import narla
import numpy as np


class Neuron:
    def __init__(self):
        super().__init__()

        self._history = narla.history.History()
        self._environment: narla.environments.Environment = None

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def environment(self) -> narla.environments.Environment:
        return self._environment

    @environment.setter
    def environment(self, environment: narla.environments.Environment):
        self._environment = environment

    def learn(self):
        pass

    def record(self, **kwargs):
        self._history.record(**kwargs)
