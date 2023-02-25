from __future__ import annotations

import torch
import narla
import numpy as np
from typing import Tuple


class Environment:
    """
    Environment base class

    :param name: Name of the environment
    :param render: If ``True`` will visualize the environment
    """
    def __init__(self, name: narla.environments.AvailableEnvironments, render: bool = False):
        self._name = name
        self._render = render

        self._episode_reward = 0
        self._action_space: narla.environments.ActionSpace = None

    @property
    def action_space(self) -> narla.environments.ActionSpace:
        """
        Get the Environment's ActionSpace
        """
        return self._action_space

    @staticmethod
    def _cast_observation(observation: np.ndarray) -> torch.Tensor:
        observation = torch.tensor([observation.tolist()], device=narla.settings.device)

        return observation

    @staticmethod
    def _cast_reward(reward: float) -> torch.Tensor:
        reward = torch.tensor([reward], device=narla.settings.device)

        return reward

    @property
    def episode_reward(self) -> float:
        """
        Get the total reward from the current episode
        """
        return self._episode_reward

    @property
    def observation_size(self) -> int:
        """
        Access the size of the observation that the Environment produces
        """
        pass

    def reset(self) -> torch.Tensor:
        """
        Reset the environment

        :return: Observation
        """
        pass

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Take a single action and advance the Environment one time step

        :param action: Action to be taken
        :return: Observation, reward, terminated
        """
        pass
