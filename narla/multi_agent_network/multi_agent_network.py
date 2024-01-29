from __future__ import annotations

import torch
import narla
from typing import List


class MultiAgentNetwork:
    """
    A MultiAgentNetwork contains a list of Layers

    :param observation_size: Size of the observation which the MultiAgentNetwork will receive
    :param number_of_actions: Number of actions available to the MultiAgentNetwork
    :param number_of_layers: Number of Layers in the MultiAgentNetwork
    :param number_of_neurons_per_layer: Number of Neurons each Layer will have
    """
    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        number_of_layers: int,
        number_of_neurons_per_layer: int,
        learning_rate: float,
    ):
        self._history = narla.history.History(storage_size=1_000_000)
        self._layers: List[narla.multi_agent_network.Layer] = self._build_layers(
            observation_size=observation_size,
            learning_rate=learning_rate,
            number_of_actions=number_of_actions,
            number_of_layers=number_of_layers,
            number_of_neurons_per_layer=number_of_neurons_per_layer,
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
        learning_rate: float,
        number_of_actions: int,
        number_of_layers: int,
        number_of_neurons_per_layer: int
    ) -> List[narla.multi_agent_network.Layer]:
        layers: List[narla.multi_agent_network.Layer] = []
        for layer_index in range(number_of_layers):

            if layer_index == number_of_layers - 1:
                # On the last layer there's just a single output neuron
                layer = narla.multi_agent_network.Layer(
                    observation_size=observation_size,
                    learning_rate=learning_rate,
                    number_of_actions=number_of_actions,
                    number_of_neurons=1
                )

            else:
                layer = narla.multi_agent_network.Layer(
                    observation_size=observation_size,
                    learning_rate=learning_rate,
                    number_of_actions=2,
                    number_of_neurons=number_of_neurons_per_layer
                )
                observation_size = number_of_neurons_per_layer

            layers.append(layer)

        return layers

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

    def learn(self):
        """
        Execute learning phase for Layers
        """
        for layer in self._layers:
            layer.learn()

    def record(self, **kwargs):
        """
        Record data into the MultiAgentNetwork's History

        :param kwargs: Key word arguments to be stored in the History
        """
        self._history.record(**kwargs)
