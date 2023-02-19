from __future__ import annotations

import torch
import narla
from typing import List


class MultiAgentNetwork:
    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        number_of_layers: int,
        number_of_neurons_per_layer: int
    ):
        self._layers: List[narla.multi_agent_network.Layer] = self._build_layers(
            observation_size=observation_size,
            number_of_actions=number_of_actions,
            number_of_layers=number_of_layers,
            number_of_neurons_per_layer=number_of_neurons_per_layer
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        previous_layer_action = observation
        for layer in self._layers:
            previous_layer_action = layer.act(previous_layer_action)

        return previous_layer_action

    @staticmethod
    def _build_layers(
        observation_size: int,
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
                    number_of_actions=number_of_actions,
                    number_of_neurons=1
                )

            else:
                layer = narla.multi_agent_network.Layer(
                    observation_size=observation_size,
                    number_of_actions=2,
                    number_of_neurons=number_of_neurons_per_layer
                )
                observation_size = number_of_neurons_per_layer

            layers.append(layer)

        return layers

    def learn(self):
        for layer in self._layers:
            layer.learn()

    def record(self, **kwargs):
        for layer in self._layers:
            layer.record(**kwargs)
