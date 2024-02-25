from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

import narla
from narla.neurons.neuron import Neuron as BaseNeuron

TAU = 0.005
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000


class Neuron(BaseNeuron):
    """
    The DeepQ Neuron is based on this `PyTorch example <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>`_

    :param observation_size: Size of the observation which the Neuron will receive
    :param number_of_actions: Number of actions available to the Neuron
    :param learning_rate: Learning rate for the Neuron's Network
    """

    def __init__(self, observation_size: int, number_of_actions: int, learning_rate: float = 1e-4):
        super().__init__(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            learning_rate=learning_rate,
        )

        network = narla.neurons.deep_q.Network(
            input_size=observation_size,
            output_size=number_of_actions,
        ).to(narla.experiment_settings.trial_settings.device)

        self._policy_network = network
        self._target_network = network.clone()

        self._number_of_steps = 0

        self._loss_function = torch.nn.SmoothL1Loss()
        self._optimizer = torch.optim.AdamW(self._policy_network.parameters(), lr=learning_rate, amsgrad=True)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * self._number_of_steps / EPSILON_DECAY)
        self._number_of_steps += 1

        if np.random.rand() > eps_threshold:

            with torch.no_grad():
                output = self._policy_network(observation)
                action = output.max(1)[1].view(1, 1)

        else:
            action = torch.tensor(
                data=[[np.random.randint(0, self.number_of_actions)]],
                device=narla.experiment_settings.trial_settings.device,
                dtype=torch.long,
            )

        self._history.record(
            observation=observation,
            action=action,
        )

        return action

    def learn(self, *reward_types: narla.rewards.RewardTypes):
        if len(self._history) < narla.experiment_settings.trial_settings.batch_size:
            return

        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = self.sample_history(*reward_types)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the
        # actions which would've been taken for each batch state according to policy_net
        state_action_values = self._policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting
        # their best reward with max(1)[0]. This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(
            narla.experiment_settings.trial_settings.batch_size,
            device=narla.experiment_settings.trial_settings.device,
        )

        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_network(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = self._loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._policy_network.parameters(), 100)

        self._optimizer.step()

        # Update the weights of the target network
        self.update_target_network()

    def sample_history(self, *reward_types: narla.rewards.RewardTypes) -> Tuple[torch.Tensor, ...]:
        *rewards, observations, actions, next_observations, terminated = self._history.sample(
            names=[
                *reward_types,
                narla.history.saved_data.OBSERVATION,
                narla.history.saved_data.ACTION,
                narla.history.saved_data.NEXT_OBSERVATION,
                narla.history.saved_data.TERMINATED,
            ],
            sample_size=narla.experiment_settings.trial_settings.batch_size,
        )

        # Combine the all the rewards into a Tensor with shape (batch_size, number_of_rewards)
        rewards = torch.stack([torch.stack(reward_type).squeeze() for reward_type in rewards], dim=-1)
        # Then sum the rewards along the samples
        reward_batch = torch.sum(rewards, dim=-1)

        observation_batch = torch.cat(observations)
        action_batch = torch.cat(actions)

        non_final_next_states = torch.cat([observation for observation, done in zip(next_observations, terminated) if not done])
        non_final_mask = torch.tensor(
            data=[not done for done in terminated],
            device=narla.experiment_settings.trial_settings.device,
            dtype=torch.bool,
        )

        return observation_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_network_state_dict = self._target_network.state_dict()
        policy_network_state_dict = self._policy_network.state_dict()

        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key] * TAU + target_network_state_dict[key] * (1 - TAU)

        self._target_network.load_state_dict(target_network_state_dict)
