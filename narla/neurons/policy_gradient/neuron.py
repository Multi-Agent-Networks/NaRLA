from __future__ import annotations

import torch

import narla
from narla.neurons import Neuron as BaseNeuron

GAMMA = 0.99


class Neuron(BaseNeuron):
    """
    PolicyGradient Neuron

    :param observation_size: Size of the observation which the Neuron will receive
    :param number_of_actions: Number of actions available to the Neuron
    :param learning_rate: Learning rate for the Neuron's Network
    """

    def __init__(self, observation_size: int, number_of_actions: int, learning_rate: float = 1e-2):
        super().__init__(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            learning_rate=learning_rate,
        )

        self._network = narla.neurons.policy_gradient.Network(
            input_size=observation_size,
            output_size=number_of_actions,
        ).to(narla.experiment_settings.trial_settings.device)

        self._number_of_steps = 0

        self._loss_function = torch.nn.SmoothL1Loss()
        self._optimizer = torch.optim.AdamW(self._network.parameters(), lr=learning_rate, amsgrad=True)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        action_probabilities = self._network(observation)

        distribution = torch.distributions.Categorical(action_probabilities)
        action = distribution.sample()

        self._history.record(
            observation=observation,
            log_probability=distribution.log_prob(action),
            action=action,
        )

        return action

    def learn(self):
        policy_losses = []

        returns = self.get_returns()
        log_probabilities = self._history.get(narla.history.saved_data.LOG_PROBABILITY)

        for return_value, log_probability in zip(returns, log_probabilities):
            # calculate actor (policy) loss
            policy_losses.append(-log_probability * return_value)

        # reset gradients
        self._optimizer.zero_grad()

        # sum up all the values of policy_losses
        loss = torch.stack(policy_losses).sum()

        # perform backprop
        loss.backward()
        self._optimizer.step()

        self._history.clear()

    def get_returns(self) -> torch.Tensor:
        rewards = self._history.get(narla.rewards.RewardTypes.TASK_REWARD)

        returns = []
        return_value = 0
        for reward in reversed(rewards):
            return_value = reward + GAMMA * return_value
            returns.insert(0, return_value)

        returns = torch.tensor(returns, device=narla.experiment_settings.trial_settings.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        return returns
