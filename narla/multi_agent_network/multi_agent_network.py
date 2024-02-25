from __future__ import annotations

from typing import List

import torch

import narla


class MultiAgentNetwork:
    """
    A MultiAgentNetwork contains a list of Layers

    :param observation_size: Size of the observation which the MultiAgentNetwork will receive
    :param number_of_actions: Number of actions available to the MultiAgentNetwork
    :param network_settings: Settings for the MultiAgentNetwork
    """

    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        network_settings: narla.multi_agent_network.MultiAgentNetworkSettings,
    ):
        self._network_settings = network_settings

        self._history = narla.history.History(storage_size=1_000_000)
        self._layers: List[narla.multi_agent_network.Layer] = self._build_layers(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            network_settings=network_settings,
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Take an action based on the observation

        :param observation: Observation from the MultiAgentNetwork environment
        """
        previous_layer_action = observation
        for layer in self._layers:
            previous_layer_action = layer.act(previous_layer_action)

        return previous_layer_action

    @staticmethod
    def _build_layers(
        observation_size: int,
        number_of_actions: int,
        network_settings: narla.multi_agent_network.MultiAgentNetworkSettings,
    ) -> List[narla.multi_agent_network.Layer]:
        layers: List[narla.multi_agent_network.Layer] = []
        for layer_index in range(network_settings.number_of_layers):
            is_last_layer = layer_index == network_settings.number_of_layers - 1

            if is_last_layer:
                # On the last layer there's just a single output neuron
                last_layer_settings = network_settings.layer_settings.clone()
                last_layer_settings.number_of_neurons_per_layer = 1

                layer = narla.multi_agent_network.Layer(
                    observation_size=observation_size,
                    number_of_actions=number_of_actions,
                    layer_settings=last_layer_settings,
                )

            else:
                layer = narla.multi_agent_network.Layer(
                    observation_size=observation_size,
                    number_of_actions=2,
                    layer_settings=network_settings.layer_settings,
                )
                observation_size = network_settings.layer_settings.number_of_neurons_per_layer

            # The first and last Layers should be fully connected
            layer.connectivity = narla.multi_agent_network.Layer.build_connectivity(
                observation_size=layer.observation_size,
                number_of_neurons=layer.number_of_neurons,
                local_connectivity=False if layer_index == 0 or is_last_layer else network_settings.local_connectivity,
            )

            layers.append(layer)

        return layers

    def compute_biological_rewards(self):
        """
        Compute BiologicalRewards and distribute to the Neurons
        """
        for layer_index, layer in enumerate(self._layers):
            for reward_type in self._network_settings.reward_types:
                if reward_type not in narla.rewards.RewardTypes.biological_reward_types():
                    continue

                reward = reward_type.to_reward()
                rewards = reward.compute(network=self, layer_index=layer_index)

                for neuron_index, neuron in enumerate(layer.neurons):
                    neuron.record(**{reward_type: rewards[0, neuron_index]})

    def distribute_to_layers(self, **kwargs):
        """
        Distribute data to the Layers

        :param kwargs: Key word arguments to be distributed
        """
        for layer in self._layers:
            layer.distribute_to_neurons(**kwargs)

    @property
    def history(self) -> narla.history.History:
        """
        Access the History of the MultiAgentNetwork
        """
        return self._history

    @property
    def layers(self) -> List[narla.multi_agent_network.Layer]:
        """
        Access the Layers of the MultiAgentNetwork
        """
        return self._layers

    def learn(self):
        """
        Execute learning phase for Layers
        """
        for layer in self._layers:
            layer.learn(*self._network_settings.reward_types)

    def record(self, **kwargs):
        """
        Record data into the MultiAgentNetwork's History

        :param kwargs: Key word arguments to be stored in the History
        """
        self._history.record(**kwargs)
