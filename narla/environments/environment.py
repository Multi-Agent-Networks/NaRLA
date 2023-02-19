from __future__ import annotations

import torch
import narla
import numpy as np
from typing import Tuple


class Environment:
    def __init__(self, name: str, render: bool = False):
        self._name = name
        self._render = render

        self._action_space: narla.environments.ActionSpace = None

    @property
    def action_space(self) -> narla.environments.ActionSpace:
        return self._action_space

    @staticmethod
    def _cast_observation(observation: np.ndarray) -> torch.Tensor:
        observation = torch.tensor([observation.tolist()], device=narla.Settings.device)

        return observation

    @staticmethod
    def _cast_reward(reward: float) -> torch.Tensor:
        reward = torch.tensor([reward], device=narla.Settings.device)

        return reward

    @property
    def observation_size(self) -> int:
        pass

    def reset(self):
        pass

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        pass
