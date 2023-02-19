import torch
import narla
import numpy as np
from typing import Tuple


class ActionSpace:
    def __init__(self, number_of_actions: int):
        self._number_of_actions = number_of_actions

    @property
    def number_of_actions(self) -> int:
        return self._number_of_actions

    def sample(self) -> torch.Tensor:
        action = np.random.randint(self._number_of_actions)

        return torch.tensor([action], device=narla.settings.device)

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()
