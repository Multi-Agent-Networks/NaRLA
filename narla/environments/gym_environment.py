from __future__ import annotations

import torch
import narla
import gymnasium as gym
from typing import Tuple
from narla.environments import Environment


class GymEnvironment(Environment):
    def __init__(self, name: narla.environments.AvailableEnvironments, render: bool = False):
        super().__init__(
            name=name,
            render=render
        )

        self._gym_environment = self._build_gym_environment(
            name=name,
            render=render
        )
        self._action_space = narla.environments.ActionSpace(
            number_of_actions=self._gym_environment.action_space.n
        )

    @staticmethod
    def _build_gym_environment(name: narla.environments.AvailableEnvironments, render: bool) -> gym.Env:
        render_mode = None
        if render:
            render_mode = "human"

        gym_environment = gym.make(
            id=name.value,
            render_mode=render_mode
        )

        return gym_environment

    @property
    def observation_size(self) -> int:
        observation = self.reset()

        return observation.shape[-1]

    def reset(self) -> torch.Tensor:
        self._episode_reward = 0

        observation, info = self._gym_environment.reset()
        observation = self._cast_observation(observation)

        return observation

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        action = int(action.item())
        observation, reward, terminated, truncated, info = self._gym_environment.step(action)

        self._episode_reward += reward

        observation = self._cast_observation(observation)
        reward = self._cast_reward(reward)

        return observation, reward, terminated or truncated
